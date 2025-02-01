from itertools import chain
from typing import List, NamedTuple, Optional, Union, Any, Tuple
from functools import partial
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import optax
import optax.tree_utils as otu
from chex import Numeric
from jaxtyping import Array
from optax import GradientTransformation, Updates
from optax._src.utils import canonicalize_dtype
from levanter.optim.config import HessianOptConfig, OptimizerConfig
from dataclasses import dataclass


class SOAPState(NamedTuple):
    count: jnp.ndarray  # type: ignore
    exp_avg: Updates
    exp_avg_sq: Updates
    GG: Any
    Q: Any

@OptimizerConfig.register_subclass("soap")
@dataclass
class SoapConfig(OptimizerConfig):
    weight_decay: float = 0.0
    beta1: float = 0.95
    beta2: float = 0.95
    shampoo_beta: float = 0.95
    eps: float = 1e-8
    max_grad_norm: Optional[float] = 1.0
    haps: Optional[list[int]] = None
    schedule_list: Optional[list[str]] = None
    precondition_frequency: int = 10
    scanned_layers: Optional[optax.Params] = None
    max_precond_dim: int = 10000
    mu_dtype: Optional[Any] = None
    precond_dtype: Optional[Any] = None
    partition_grads_into_blocks: bool = True
    block_size: int = 256
    def build(self, num_train_steps):
        """Creates the optimizer"""
        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))
            components.append(scale_by_soap(
            b1=self.beta1,
            b2=self.beta2,
            shampoo_beta=self.shampoo_beta,
            eps=self.eps,
            precondition_frequency=self.precondition_frequency,
            max_precond_dim=self.max_precond_dim,
            mu_dtype=self.mu_dtype,
            precond_dtype=self.precond_dtype,
            partition_grads_into_blocks=self.partition_grads_into_blocks,
            block_size=self.block_size,
            ))

            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))

            # - learning rate for descent
            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)

            return optimizer

        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))


from jax import vmap
import haliax as hax
from jax.sharding import PartitionSpec
from jax.lax import with_sharding_constraint

def scale_by_soap(
    b1: float = 0.95,
    b2: float = 0.95,
    shampoo_beta: float = -1,
    eps: float = 1e-8,
    precondition_frequency: int = 1,
    max_precond_dim: int = 10000,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
    mu_dtype: Optional[Any] = None,
    precond_dtype: Optional[Any] = None,
    partition_grads_into_blocks: Optional[Any] = True,
    block_size: Optional[Any] = 256
) -> GradientTransformation:
    """
    Implements SOAP algorithm (https://arxiv.org/abs/2409.11321). Based on the original implementation at https://github.com/nikhilvyas/SOAP.

    Args:
        b1 (float, optional): Adam's beta1 parameter. Defaults to 0.95.
        b2 (float, optional): Adam's beta2 parameter. Defaults to 0.95.
        shampoo_beta (float, optional): If >= 0, use this beta for the preconditioner (`L` and `R` in paper, `GG` below)
            moving average instead of b2. Defaults to -1.
        eps (float, optional): Adam's epsilon for numerical stability. Defaults to 1e-8.
        precondition_frequency (int, optional): How often to update the preconditioner. Defaults to 10.
        max_precond_dim (int, optional): Maximum dimension of the preconditioner.
            Set to 10000 to exclude most common vocab sizes while including layers. Defaults to 10000.
        precision (jax.lax.PrecisionLike, optional): Precision to use. Defaults to jax.lax.Precision.H

    Returns:
        optax.GradientTransformationExtraArgs: The SOAP optimizer.
    """
    mu_dtype = canonicalize_dtype(mu_dtype) if mu_dtype is not None else None
    precond_dtype = canonicalize_dtype(precond_dtype) if precond_dtype is not None else None
    shampoo_beta = shampoo_beta if shampoo_beta >= 0 else b2
    def map_fn(do_map, fn, *args):
        """Maybe map a fn along first axis."""
        if do_map:
            return vmap(fn)(*args)
        else:
            return fn(*args)
        
    def init_fn(params: Updates) -> SOAPState:
        scanned_layers_ = jax.tree.map(
            lambda x: (
                jax.tree.map(lambda _: True, x, is_leaf=lambda x: isinstance(x, jax.Array))
                if isinstance(x, hax.nn.Stacked)
                else jax.tree.map(lambda _: False, x, is_leaf=lambda x: isinstance(x, jax.Array))
            ),
            params,
            is_leaf=lambda x: isinstance(x, hax.nn.Stacked),
        )        
        exp_avg = otu.tree_zeros_like(params, dtype=mu_dtype)
        
        
        # if partition_grads_into_blocks:
        #     partitioners = jax.tree.map(
        #         lambda _, ps, dd: BlockPartitioner(ps, block_size, dd),
        #         params,
        #         merged_shapes,
        #         dim_diag,
        #     )
        #     # we can grab resulting shapes from partitioners
        #     partitioned_shapes = jax.tree.map(
        #         lambda _, p_cls: p_cls._padded_stacked_shape, params, partitioners
        #     )

        
        
        
        exp_avg_sq = otu.tree_zeros_like(params, dtype=mu_dtype)
        GG = [
            init_conditioner(
                t[0] if s else t,
                max_precond_dim,
                precond_dtype
            ) 
            for t, s in zip(
                jax.tree.leaves(params), jax.tree.leaves(scanned_layers_)
            )
        ]

        GG = [
                (jax.tree.map(
                    lambda d: jnp.repeat(jnp.expand_dims(d, 0), t.shape[0], axis=0) if d is not None else None, q
                )
                if s
                else q
            ) 
            for q, t, s in zip(GG, jax.tree.leaves(params), jax.tree.leaves(scanned_layers_))
        ]

        Q = [
            init_conditioner(
                t[0] if s else t,
                max_precond_dim,
                precond_dtype
            ) 
            for t, s in zip(
                jax.tree.leaves(params), jax.tree.leaves(scanned_layers_)
            )
        ]

        Q = [
                (jax.tree.map(
                    lambda d: jnp.repeat(jnp.expand_dims(d, 0), t.shape[0], axis=0) if d is not None else None, q
                )
                if s
                else q
            ) 
            for q, t, s in zip(Q, jax.tree.leaves(params), jax.tree.leaves(scanned_layers_))
        ]


        return SOAPState(
            count=jnp.zeros([], jnp.int32),
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            GG=GG,
            Q=Q,
        )

    def init_step(
        updates: Updates,
        state: SOAPState,
        scanned_layers_: Updates
    ) -> tuple[Updates, SOAPState]:
        new_GG = [
            map_fn(s, 
                   partial(update_preconditioner, beta = shampoo_beta),
                   grad,
                   gg
            )
            for s, grad, gg in zip(jax.tree.leaves(scanned_layers_), jax.tree.leaves(updates), state.GG)
        ]

    

        new_Q = [map_fn(s, 
                    partial(get_orthogonal_matrix, eps = eps),
                   gg) 
                   for s, gg in zip(jax.tree.leaves(scanned_layers_), state.GG)]

        new_GG = otu.tree_cast(new_GG, precond_dtype)
        new_Q = otu.tree_cast(new_Q, precond_dtype)
        

        # Replace updates with zeros
        new_updates = otu.tree_zeros_like(updates)

        return new_updates, state._replace(GG=new_GG, Q=new_Q)

    def update_step(
        updates: Updates,
        state: SOAPState,
        scanned_layers_: Updates
    ) -> tuple[Updates, SOAPState]:
        # Project gradients
        _, grads_structure = jax.tree.flatten(updates, is_leaf=lambda x: isinstance(x, jax.Array))
        grad_projected = [
            map_fn(s, partial(project, precision=precision), grad, q)
            for s, grad, q in zip(
                jax.tree.leaves(scanned_layers_),
                jax.tree.leaves(updates),
                state.Q
            )
        ]
        grad_projected = grads_structure.unflatten(grad_projected)

        # Update moments
        exp_avg = otu.tree_update_moment(updates, state.exp_avg, b1, 1)
        exp_avg_sq = otu.tree_update_moment_per_elem_norm(grad_projected, state.exp_avg_sq, b2, 2)

        exp_avg_projected = [
            map_fn(s, partial(project, precision=precision), e, q) for s, e, q in 
            zip(jax.tree.leaves(scanned_layers_),
            jax.tree.leaves(exp_avg),
            state.Q)
        ]
        exp_avg_projected = grads_structure.unflatten(exp_avg_projected)

        # Project back
        norm_updates = [
            map_fn(s, partial(project_back, precision = precision), (e_avg / (jnp.sqrt(e_avg_sq) + eps)), q)
            for  s, e_avg, e_avg_sq, q in 
            zip(jax.tree.leaves(scanned_layers_),
            jax.tree.leaves(exp_avg_projected),
            jax.tree.leaves(exp_avg_sq),
            state.Q)
        ]
        norm_updates = grads_structure.unflatten(norm_updates)

        bc1 = 1 - b1**state.count
        bc2 = 1 - b2**state.count
        corr = jnp.sqrt(bc2) / bc1

        # Bias correction on the updates
        norm_updates = jtu.tree_map(
            lambda p: p * corr,
            norm_updates,
        )

        # Update the preconditioner
        new_GG = [
            map_fn(s, 
                   partial(update_preconditioner, beta = shampoo_beta),
                   grad,
                   gg
            )
            for s, grad, gg in zip(jax.tree.leaves(scanned_layers_), jax.tree.leaves(updates), state.GG)
        ]


        # Update the orthogonal matrix / exp_avg_sq
        new_Q_and_exp_avg_sq = jax.lax.cond(
            state.count % precondition_frequency == 0,
            lambda: [map_fn(s,  partial(get_orthogonal_matrix_QR, precision = precision, precond_dtype = precond_dtype, mu_dtype = mu_dtype), 
                                           gg, q, e) for s, e, gg, q in
                zip(jax.tree.leaves(scanned_layers_),
                jax.tree.leaves(exp_avg_sq),
                new_GG,
                state.Q)
            ],
            lambda: [(q,e) for e, q in
                zip(jax.tree.leaves(state.exp_avg_sq),
                state.Q)
            ]
        )
        ## Unpack the results
        new_Q = [x[0] for x in new_Q_and_exp_avg_sq]
        exp_avg_sq = [x[1] for x in new_Q_and_exp_avg_sq]
        exp_avg_sq = grads_structure.unflatten(exp_avg_sq)
        exp_avg = otu.tree_cast(exp_avg, mu_dtype)
        exp_avg_sq = otu.tree_cast(exp_avg_sq, mu_dtype)
        new_GG = otu.tree_cast(new_GG, precond_dtype)
        new_Q = otu.tree_cast(new_Q, precond_dtype)
        new_state = SOAPState(
            count=state.count,
            exp_avg=exp_avg,
            exp_avg_sq=exp_avg_sq,
            GG=new_GG,
            Q=new_Q,
        )
        

        return norm_updates, new_state

    def update_fn(updates: Updates, state: SOAPState, params: Optional[Updates] = None) -> tuple[Updates, SOAPState]:
        count_inc = jnp.asarray(optax.safe_int32_increment(state.count))
        state = state._replace(count=count_inc)
        scanned_layers_ = jax.tree.map(
            lambda x: (
                jax.tree.map(lambda _: True, x, is_leaf=lambda x: isinstance(x, jax.Array))
                if isinstance(x, hax.nn.Stacked)
                else jax.tree.map(lambda _: False, x, is_leaf=lambda x: isinstance(x, jax.Array))
            ),
            params,
            is_leaf=lambda x: isinstance(x, hax.nn.Stacked),
        )
        updates, new_state = jax.lax.cond(
            count_inc == 1,
            lambda: init_step(updates, state, scanned_layers_),
            lambda: update_step(updates, state, scanned_layers_),
        )

        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore


def update_preconditioner(
    grad: Array,
    GG: List[Union[Array, None]],
    beta: float,
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,

) -> List[Union[Array, None]]:
    if grad.ndim == 1:
        return [lerp(GG[0], jnp.matmul(grad[:, None], grad[None, :], precision=precision), 1 - beta)]  # type: ignore

    new_GG = []
    for idx, gg in enumerate(GG):
        if gg is None:
            new_GG.append(None)
            continue

        outer_product = jnp.tensordot(
            grad,
            grad,
            axes=[[*chain(range(idx), range(idx + 1, len(grad.shape)))]] * 2,
            precision=precision,
        )
        new_GG.append(lerp(gg, outer_product, 1 - beta))

    return new_GG


def project(
    grad: Array,
    Q: List[Union[Array, None]],
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> Array:
    for mat in Q:
        if mat is not None:  # noqa: SIM108
            grad = jnp.tensordot(
                grad,
                mat,
                axes=((0,), (0,)),
                precision=precision,
            )
        else:
            permute_order = list(range(1, len(grad.shape))) + [0]
            grad = jnp.transpose(grad, permute_order)

    return grad


def project_back(
    grad: Array,
    Q: List[Union[Array, None]],
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> Array:
    for mat in Q:
        if mat is not None:  # noqa: SIM108
            grad = jnp.tensordot(
                grad,
                mat,
                axes=((0,), (1,)),
                precision=precision,
            )
        else:
            grad = jnp.moveaxis(grad, 0, -1)

    return grad


def get_orthogonal_matrix(GG: List[Union[Array, None]], eps: float) -> List[Union[Array, None]]:
    Q = []
    for gg in GG:
        if gg is None:
            Q.append(None)
        else:
            _, eigh = jnp.linalg.eigh(gg + eps * jnp.eye(gg.shape[0]))
            Q.append(jnp.flip(eigh, axis=1))
    return Q


def get_orthogonal_matrix_QR(
    GG: List[Union[Array, None]],
    Q: List[Union[Array, None]],
    exp_avg_sq: Array,
    precond_dtype: Optional[Any],
    mu_dtype: Optional[Any],
    precision: jax.lax.PrecisionLike = jax.lax.Precision.HIGHEST,
) -> tuple[List[Union[Array, None]], Array]:
    final_Q = []
    for ind, (m, o) in enumerate(zip(GG, Q)):
        if m is None or o is None:
            final_Q.append(None)
            continue

        est_eig = jnp.diag(
            jnp.matmul(
                jnp.matmul(o.T, m, precision=precision),
                o,
                precision=precision,
            )
        )
        sort_idx = jnp.argsort(est_eig, descending=True)
        exp_avg_sq = jnp.take(exp_avg_sq, sort_idx, axis=ind)
        o = o[:, sort_idx]
        power_iter = jnp.matmul(m, o, precision=precision)
        Q_new, _ = jnp.linalg.qr(power_iter)
        final_Q.append(Q_new)
    final_Q = otu.tree_cast(final_Q, precond_dtype)
    exp_avg_sq = otu.tree_cast(exp_avg_sq, mu_dtype)
    return final_Q, exp_avg_sq


def lerp(
    start: Array,
    end: Array,
    weight: Numeric,
):
    return start + weight * (end - start)


def _get_preconditioner_types(
    shape: Tuple[int, ...], max_precond_dim: int
) -> List[bool]:
    if len(shape) == 0:
        return True
    
    if len(shape) == 1:
        return [True]
    
    return [s <= max_precond_dim for s in shape]



def init_conditioner(p: Array, max_precond_dim: int, dtype: Optional[Any]) -> List[Union[Array, None]]:
    if p.ndim == 1:
        return [jnp.zeros((p.shape[0], p.shape[0]), dtype = dtype)]

    return [jnp.zeros((s, s), dtype = dtype) if s <= max_precond_dim else None for s in p.shape]

import numpy as np

class BlockPartitioner:
    """Partitions a tensor into smaller tensors.

    Modified from distributed_shampoo.
    https://github.com/google-research/google-research/blob/master/scalable_shampoo/optax/distributed_shampoo.py
    Scalable Second Order Optimization for Deep Learning,
    Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer
    https://arxiv.org/abs/2002.09018
    """

    def __init__(self, param_shape, block_size, dim_diag):
        self._shape = param_shape
        self._splits = []
        split_sizes = []
        # We split params into smaller blocks. Here we store the metadata to make
        # that split.
        for i, d in enumerate(param_shape):
            if 0 < block_size < d and not dim_diag[i]:
                # d-1, otherwise split appends a 0-size array.
                nsplit = (d - 1) // block_size
                indices = (np.arange(nsplit, dtype=np.int32) + 1) * block_size
                sizes = np.ones(nsplit + 1, dtype=np.int32) * block_size
                sizes[-1] = d - indices[-1]
                self._splits.append((i, indices))
                split_sizes.append(sizes)
            else:
                split_sizes.append(np.array([d], dtype=np.int32))
        self._split_sizes = split_sizes

        # TODO (evanatyourservice)
        # this might fail with scalar params but for now we're reshaping those
        single_shape = [a[0] for a in split_sizes]
        padded_single_shape = [-(-dim // block_size) * block_size for dim in single_shape]
        stack_size = max(1, np.prod([max(1, len(s)) for s in split_sizes]))
        self._padded_stacked_shape = tuple([stack_size] + padded_single_shape)

    def split_sizes(self):
        return self._split_sizes

    def partition(self, tensor):
        """Partition tensor into blocks."""

        assert tensor.shape == self._shape
        tensors = [tensor]
        for i, indices in self._splits:
            tensors_local = []
            for t in tensors:
                tensors_local.extend(jnp.split(t, indices_or_sections=indices, axis=i))
            tensors = tensors_local
        return tuple(tensors)

    def merge_partitions(self, partitions):
        """Merge partitions back to original shape."""

        for i, indices in reversed(self._splits):
            n = len(indices) + 1
            partial_merged_tensors = []
            ind = 0
            while ind < len(partitions):
                partial_merged_tensors.append(
                    jnp.concatenate(partitions[ind : ind + n], axis=i)
                )
                ind += n
            partitions = partial_merged_tensors
        assert len(partitions) == 1
        return partitions[0]


def _partitions(lst):
    """Generate all partitions of a list."""
    if not lst:
        yield [[]]
    else:
        for i in range(len(lst)):
            for part in _partitions(lst[i + 1 :]):
                yield [lst[: i + 1]] + part


def _merge_small_dims(
    shape_to_merge, max_dim, dim_diag, sharding_to_merge=None
) -> Tuple[List[int], List[bool], Optional[Tuple]]:
    if not shape_to_merge:  # handles scalar shape ()
        return [], [True], PartitionSpec() if sharding_to_merge is not None else None
    if np.all(np.array(shape_to_merge) == 1):  # handles shape (1,)
        return (
            [1],
            [True],
            PartitionSpec(None) if sharding_to_merge is not None else None,
        )

    def dim2loss(d, dim0=max_dim):
        """A heuristic map from dim to loss with the least loss occurs at dim0."""
        loss = 0
        if d < dim0:
            loss += np.log2(dim0 / d)
            too_small = dim0 / 8
            if d < too_small:
                loss += 100 * np.log2(too_small / d)
        else:
            loss += 10 * np.log2(d / dim0)
            too_large = 8 * dim0
            if d > too_large:
                loss += 1000 * np.log2(d / too_large)
        return loss

    best_loss = float("inf")
    best_partition = None

    for p in _partitions(list(range(len(shape_to_merge)))):
        loss = 0
        merged = []
        for group in p:
            if not group:
                continue
            d = np.prod([shape_to_merge[i] for i in group])
            loss += dim2loss(d)
            merged.append(group)

        if loss < best_loss:
            best_loss = loss
            best_partition = merged

    merged_shape = []
    merged_diag = []
    merged_sharding = []

    for group in best_partition:
        merged_shape.append(np.prod([shape_to_merge[i] for i in group]))
        merged_diag.append(all(dim_diag[i] for i in group))
        if sharding_to_merge:
            group_shardings = [sharding_to_merge[i] for i in group]
            valid_shardings = [s for s in group_shardings if s is not None]

            if len(valid_shardings) > 1:
                merged_sharding.append(tuple(valid_shardings))
            elif len(valid_shardings) == 1:
                merged_sharding.append(valid_shardings[0])
            else:
                merged_sharding.append(None)

    return (
        merged_shape,
        merged_diag,
        PartitionSpec(*merged_sharding) if sharding_to_merge else None,
    )


def _pad_and_stack_matrices(array_list, block_size):
    # Handle scalar arrays by adding a dummy dimension
    is_scalar = len(array_list[0].shape) == 0
    if is_scalar:
        array_list = [arr[None] for arr in array_list]

    shapes = [arr.shape for arr in array_list]
    max_dims = [max(shape[i] for shape in shapes) for i in range(len(shapes[0]))]
    padded_shape = [-(-dim // block_size) * block_size for dim in max_dims]
    padded_arrays = []
    for arr in array_list:
        pad_width = [(0, padded_shape[i] - arr.shape[i]) for i in range(arr.ndim)]
        padded = jnp.pad(arr, pad_width)
        padded_arrays.append(padded)

    stacked = jnp.stack(padded_arrays)
    return stacked


def _unstack_and_unpad_matrices(stacked_array, original_shapes):
    # Handle scalar arrays
    is_scalar = len(original_shapes[0]) == 0

    unstacked = jnp.split(stacked_array, stacked_array.shape[0], axis=0)
    unpadded = []
    for arr, orig_shape in zip(unstacked, original_shapes):
        arr = jnp.squeeze(arr, axis=0)
        if is_scalar:
            # For scalars, just take the first element
            arr = arr[0]
        else:
            # For non-scalars, slice to original shape
            slices = tuple(slice(0, dim) for dim in orig_shape)
            arr = arr[slices]
        unpadded.append(arr)
    return tuple(unpadded)

from collections import defaultdict

# unused fns (can be used for stacking partitions without padding):
def _sort_and_group_matrices(matrix_shapes: List[Tuple[int, ...]]):
    indexed_list = list(enumerate(matrix_shapes))
    sorted_indexed = sorted(indexed_list, key=lambda x: x[1])
    sorted_shapes = [shape for _, shape in sorted_indexed]
    change_indices = [original_index for original_index, _ in sorted_indexed]
    revert_indices = [0] * len(matrix_shapes)
    for new_pos, (original_index, _) in enumerate(sorted_indexed):
        revert_indices[original_index] = new_pos
    shape_groups = defaultdict(list)
    for i, shape in enumerate(sorted_shapes):
        shape_groups[shape].append(i)
    unique_sorted_shapes = list(shape_groups.keys())
    return unique_sorted_shapes, dict(shape_groups), change_indices, revert_indices


def _stack_matrices(array_list):
    in_tuple = isinstance(array_list, tuple)
    shapes = [arr.shape for arr in array_list]
    unique_shapes, shape_groups, change_indices, _ = _sort_and_group_matrices(shapes)
    sorted_arrays = [array_list[i] for i in change_indices]
    stacked_arrays = []
    for shape in unique_shapes:
        indices = shape_groups[shape]
        stacked = jnp.stack([sorted_arrays[i] for i in indices])
        stacked_arrays.append(stacked)
    if in_tuple:
        return tuple(stacked_arrays)
    return stacked_arrays


def _unstack_matrices(stacked_arrays, revert_indices):
    in_tuple = isinstance(stacked_arrays, tuple)
    unstacked = []
    for arr in stacked_arrays:
        unstacked.extend(jnp.split(arr, arr.shape[0]))
    array_list = [jnp.squeeze(unstacked[i], axis=0) for i in revert_indices]
    if in_tuple:
        return tuple(array_list)
    return array_list