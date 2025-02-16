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
from levanter.utils.jax_utils import leaf_key_paths
from haliax.nn import Linear
from dataclasses import dataclass
import dataclasses


class MuonBState(NamedTuple):
    count: jnp.ndarray  # type: ignore
    exp_avg: Updates

@OptimizerConfig.register_subclass("muonb")
@dataclass
class MuonBConfig(OptimizerConfig):
    weight_decay: float = 0.0
    beta1: float = 0.95
    beta2: float = 0.95
    shampoo_beta: float = 0.95
    eps: float = 1e-8
    adam_eps: float = 1e-15
    max_grad_norm: Optional[float] = 1.0
    muon_to_adam_lr: float = 0.18 
    haps: Optional[list[int]] = None
    schedule_list: Optional[list[str]] = None
    precondition_frequency: int = 10
    max_precond_dim: int = 10000
    merge_small_dims: bool = True
    one_diag: bool = False
    target_merged_dim_size: int = 2048
    mu_dtype: Optional[Any] = None
    precond_dtype: Optional[Any] = None
    partition_grads_into_blocks: bool = True
    block_size: int = 256
    def build(self, num_train_steps):
        """Creates the optimizer"""
        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def _optimizer(learning_rate):
            adam_lr = learning_rate * self.muon_to_adam_lr
            def muon_transform():
                components = []
                # Muon seems incompatible with gradient clipping, need to investigate
                # if self.max_grad_norm:
                #     components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(scale_by_muon(
                b1=self.beta1,
                b2=self.beta2,
                shampoo_beta=self.shampoo_beta,
                eps=self.eps,
                precondition_frequency=self.precondition_frequency,
                max_precond_dim=self.max_precond_dim,
                merge_small_dims=self.merge_small_dims,
                target_merged_dim_size=self.target_merged_dim_size,
                mu_dtype=self.mu_dtype,
                precond_dtype=self.precond_dtype,
                partition_grads_into_blocks=self.partition_grads_into_blocks,
                block_size=self.block_size,
                one_diag=self.one_diag
                ))
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                optimizer = optax.chain(*components)
                return optimizer

            def adamw_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.adam_eps))
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-adam_lr))
                optimizer = optax.chain(*components)
                return optimizer

            transformations = {
                "muon": muon_transform(),
                "adamw": adamw_transform(),
            }

            return optax.multi_transform(transformations, self.create_mask)
        return optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))
    

    def create_mask(self, params):
        """
        Creates a mask that labels parameters as 'muon' or 'adamw' based on their
        dimensionality and module path, using AdamW for Embedding and lm_head parameters.
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adamw"
            elif isinstance(param, Linear):
                # muon for linear layers
                return dataclasses.replace(param, weight="muon", bias="adamw" if param.bias is not None else None)
            else:
                return "adamw"

        return jax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


from jax import vmap
import haliax as hax
from jax.sharding import PartitionSpec
from jax.lax import with_sharding_constraint
def _map_fn(lax_map, bs, n_maps, fn, *args):
    """Maybe map a fn along multiple leading axes."""
    if n_maps <= 0:
        return fn(*args)

    if lax_map:
        mapped_fn = lambda xs: _map_fn(lax_map, bs, n_maps - 1, fn, *xs)
        return jax.lax.map(mapped_fn, xs=args, batch_size=bs if bs > 1 else None)
    else:
        mapped_fn = lambda *xs: _map_fn(lax_map, bs, n_maps - 1, fn, *xs)
        return vmap(mapped_fn)(*args)
def scale_by_muon(
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
    block_size: Optional[Any] = 256,
    lax_map_scanned_layers: Optional[bool] = True,
    lax_map_batch_size: Optional[int] = 4,
    merge_small_dims: bool = False,
    target_merged_dim_size: int = 2048,
    one_diag: bool = False
) -> GradientTransformation:
    """
    Implements MuonB algorithm (https://arxiv.org/abs/2409.11321). Based on the original implementation at https://github.com/nikhilvyas/MuonB.

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
        optax.GradientTransformationExtraArgs: The MuonB optimizer.
    """
    mu_dtype = canonicalize_dtype(mu_dtype) if mu_dtype is not None else None
    precond_dtype = canonicalize_dtype(precond_dtype) if precond_dtype is not None else None
    shampoo_beta = shampoo_beta if shampoo_beta >= 0 else b2

        
    def init_fn(params: Updates) -> MuonBState:            
        exp_avg = otu.tree_zeros_like(params, dtype=mu_dtype)        
        return MuonBState(
            count=jnp.zeros([], jnp.int32),
            exp_avg=exp_avg,
        )

    def update_step(
        updates: Updates,
        state: MuonBState,
        scanned_layers_: Updates
    ) -> tuple[Updates, MuonBState]:
        # Update moments
        exp_avg_buf = jax.tree.map(
            lambda m, g: None if g is None else b1 * m + g,
            state.exp_avg,
            updates,
            is_leaf=lambda x: x is None,
        )
        exp_avg = jax.tree.map(
            lambda m, g: None if g is None else b1 * m + g,
            exp_avg_buf,
            updates,
            is_leaf=lambda x: x is None,
        )
        shapes = jax.tree.map(
            lambda p, s: p.shape[int(s) :], updates, scanned_layers_
        )                   
        # block gradients, exp_avg, exp_avg_sq
        n_dims_to_map = jax.tree.map(lambda s: int(s), scanned_layers_)
        dummy_updates_tree = jax.tree.map(lambda _: jnp.zeros([]), updates)
        null_dims = jax.tree.map(
            lambda p, s: _get_preconditioner_types(
                p.shape[int(s) :],
                max_precond_dim,
                one_diag
            ),
            updates,
            scanned_layers_,
        )
        # merge small dims
        merged_shapes = shapes
        if merge_small_dims:
            original_shapes = jax.tree.map(
                lambda g, s: g.shape[int(s) :], updates, scanned_layers_
            )
            output = jax.tree.map(
                lambda g, dd, s: _merge_small_dims(
                    g.shape[int(s) :], target_merged_dim_size, dd
                ),
                updates,
                null_dims,
                scanned_layers_)
            merged_shapes, null_dims = [
                jax.tree.map(lambda _, x: x[i], updates, output)
                for i in range(2)
            ]
            exp_avg = jax.tree.map(
                lambda g, s, ns: _map_fn(
                    False, 0, int(s), lambda x, shape=ns: jnp.reshape(x, shape), g
                ),
                exp_avg,
                scanned_layers_,
                merged_shapes,
            )            
        # partition
        partitioned_shapes = merged_shapes
        if partition_grads_into_blocks:
            null_dims = jax.tree.map(
                lambda p, s: _get_preconditioner_types(
                    p.shape[int(s) :],
                    max_precond_dim,
                    one_diag
                ),
                updates,
                scanned_layers_,
            )
            partitioners = jax.tree.map(
                lambda _, ps, dd: BlockPartitioner(ps, block_size, dd),
                updates,
                partitioned_shapes,
                null_dims,
            ) 
            blocked_exp_avg = jax.tree.map(
                lambda g, p_cls, s: _map_fn(False, 0, int(s), p_cls.partition, g),
                exp_avg,
                partitioners,
                scanned_layers_,
            )
            # get shapes
            partitioned_shapes = jax.tree.map(
            lambda _, g, s: jax.tree.map(lambda x: x.shape[int(s) :], g),
            dummy_updates_tree,
            blocked_exp_avg,
            scanned_layers_,)
            # padding
            blocked_exp_avg = jax.tree.map(
                lambda _, g, s: _map_fn(
                    False,
                    0,
                    int(s),
                    lambda x, bs=block_size: _pad_and_stack_matrices(x, bs),
                    g,
                ),
                dummy_updates_tree,
                blocked_exp_avg,
                scanned_layers_,
            )
            n_dims_to_map = jax.tree.map(lambda x: x + 1, n_dims_to_map)
        else:
            blocked_exp_avg = exp_avg

        
        
        
        # Project back
        blocked_norm_updates = jax.tree.map(
            lambda _, nm, e_avg: _map_fn(
                False,
                0,
                nm,
                partial(zeropower_via_newtonschulz5, eps = eps, steps = 10),
                e_avg
            ),
            dummy_updates_tree,
            n_dims_to_map,
            blocked_exp_avg,
        )        
        
        if partition_grads_into_blocks:
            norm_updates = jax.tree.map(
                lambda g, s, ps: _map_fn(
                    False,
                    0,
                    int(s),
                    lambda p, shapes=ps: _unstack_and_unpad_matrices(p, shapes),
                    g,
                ),
                blocked_norm_updates,
                scanned_layers_,
                partitioned_shapes,
            )
            norm_updates = jax.tree.map(
                lambda _, g, s, p_cls: _map_fn(
                    False, 0, int(s), p_cls.merge_partitions, g
                ),
                dummy_updates_tree,
                norm_updates,
                scanned_layers_,
                partitioners,
            )
            exp_avg = jax.tree.map(
                lambda g, s, ps: _map_fn(
                    False,
                    0,
                    int(s),
                    lambda p, shapes=ps: _unstack_and_unpad_matrices(p, shapes),
                    g,
                ),
                blocked_exp_avg,
                scanned_layers_,
                partitioned_shapes,
            )
        else:
            norm_updates = blocked_norm_updates
        
        # unmerge
        if merge_small_dims:
            norm_updates = jax.tree.map(
                lambda g, s, os: _map_fn(False, 0, int(s), lambda p, shape=os: jnp.reshape(p, shape), g),
                norm_updates,
                scanned_layers_,
                original_shapes,
            )
        
        # precision
        exp_avg_buf = otu.tree_cast(exp_avg_buf, mu_dtype)
        new_state = MuonBState(
            count=state.count,
            exp_avg=exp_avg_buf,
        )
        

        return norm_updates, new_state

    def update_fn(updates: Updates, state: MuonBState, params: Optional[Updates] = None) -> tuple[Updates, MuonBState]:
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
        updates, new_state = update_step(updates, state, scanned_layers_)

        return updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)  # type: ignore

import chex

def zeropower_via_newtonschulz5(X, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    chex.assert_rank(X, 2)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X /= jnp.linalg.norm(X) + eps  # Ensure top singular value <= 1
    transpose = False
    if X.shape[0] > X.shape[1]:
        X = X.T
        transpose = True
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transpose:
        X = X.T
    scale = jnp.sqrt(jnp.maximum(1, X.shape[0] / X.shape[1]))
    X *= scale
    # https://x.com/leloykun/status/1874358290093924849
    return X


def lerp(
    start: Array,
    end: Array,
    weight: Numeric,
):
    return start + weight * (end - start)


def _get_preconditioner_types(
    shape: Tuple[int, ...], max_precond_dim: int, one_diag: bool
) -> List[bool]:
    if len(shape) == 0:
        return False
    
    if len(shape) == 1:
        return [False]
    
    result =  [s >= max_precond_dim for s in shape]
    new_result = result
    if one_diag:
        new_result = []
        flag = True
        for i in range(len(result)):
            if not result[i] and flag:
                new_result.append(False)
                flag = False
            else:
                new_result.append(True)
        
    
    
    return new_result



def init_conditioner(p_shape: Any, max_precond_dim: int, dtype: Optional[Any], one_diag: bool) -> List[Union[Array, None]]:
    if len(p_shape) == 1:
        return [jnp.zeros((p_shape[0], p_shape[0]), dtype = dtype)]
    
    if not one_diag:
        return [jnp.zeros((s, s), dtype = dtype) if s <= max_precond_dim else None for s in p_shape]
    else:
        flag = True
        output = []
        for i in range(len(p_shape)):
            s = p_shape[i]
            if s < max_precond_dim and flag:
                output.append(jnp.zeros((s, s), dtype = dtype))
                flag = False
            else:
                output.append(None)
        return output

import numpy as np

class BlockPartitioner:
    """Partitions a tensor into smaller tensors.

    Modified from distributed_shampoo.
    https://github.com/google-research/google-research/blob/master/scalable_shampoo/optax/distributed_shampoo.py
    Scalable Second Order Optimization for Deep Learning,
    Rohan Anil, Vineet Gupta, Tomer Koren, Kevin Regan, Yoram Singer
    https://arxiv.org/abs/2002.09018
    """

    def __init__(self, param_shape, block_size, null_dims):
        self._shape = param_shape
        self._shape = tuple(int(_) for _ in self._shape) # jnp value refuse to be equal to integer, manually convert
        self._splits = []
        split_sizes = []
        # We split params into smaller blocks. Here we store the metadata to make
        # that split.
        for i, d in enumerate(param_shape):
            if 0 < block_size < d and not null_dims[i]:
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
        print('difference')
        print(tensor.shape, self._shape)
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


def _unstack_and_unpad_matrices(stacked_array, shapes):
    # Handle scalar arrays
    is_scalar = len(shapes[0]) == 0

    unstacked = jnp.split(stacked_array, stacked_array.shape[0], axis=0)
    unpadded = []
    for arr, orig_shape in zip(unstacked, shapes):
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


def _merge_small_dims(
    shape_to_merge, max_dim, null_dims
) -> Tuple[List[int], List[bool], Optional[Tuple]]:
    if not shape_to_merge:  # handles scalar shape ()
        return [], [True]
    if np.all(np.array(shape_to_merge) == 1):  # handles shape (1,)
        return (
            [1],
            [True],
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

    for group in best_partition:
        merged_shape.append(np.prod([shape_to_merge[i] for i in group]))
        merged_diag.append(all(null_dims[i] for i in group))

    return (
        merged_shape,
        merged_diag
    )