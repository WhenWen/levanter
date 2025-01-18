import dataclasses
from dataclasses import dataclass
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
import jaxtyping
from optax import tree_utils as otu
from typing import Any
import haliax
from haliax.nn import Linear
import levanter.tracker
from levanter.optim.config import OptimizerConfig
from levanter.optim.util import map_flattened_linear_layers
from levanter.utils.jax_utils import leaf_key_paths


@OptimizerConfig.register_subclass("muadam2")
@dataclass
class MuAdam2Config(OptimizerConfig):
    """
    MuAdam2 optimizer configuration: Momentum Orthogonalized by Newton-Schulz.
    """

    muadam2_to_adam_lr: float = 0.18  # Scaling factor between AdamW and MuAdam2 learning rates
    momentum: float = 0.95
    nesterov: bool = True
    backend_steps: int = 10  # Number of steps for Newton-Schulz orthogonalization
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    # adam_modules: Optional[list[str] | str] = None
    # """A regex or a list of strings to identify where to mask weight.
    # For nano-GPT, this field can be set as `r".*attn.*weight|.*mlp.*weight|.*token_embeddings|.*position_embeddings"`"""
    # default_adam_mask: Optional[bool] = None
    # """Whether to apply a default reasonable weight decay to modules not explicitly masked. None means it will if
    # no weight_decay_modules are set. False means it will not. True means it will regardless of weight_decay_modules."""

    def build(self, num_train_steps):
        """
        Creates the optimizer.
        """
        learning_rate_schedule = self.lr_scheduler(num_train_steps)

        def optimizer(learning_rate):
            adam_lr = learning_rate * self.muadam2_to_adam_lr

            def muadam2_transform():
                components = []
                components.append(scale_with_muadam2(self.momentum, self.beta2, self.nesterov, self.backend_steps))
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-learning_rate))
                optimizer = optax.chain(*components)
                return optimizer

            def adamw_transform():
                components = []
                if self.max_grad_norm:
                    components.append(optax.clip_by_global_norm(self.max_grad_norm))
                components.append(optax.scale_by_adam(self.beta1, self.beta2, self.epsilon))
                if self.weight_decay > 0:
                    components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))
                components.append(optax.scale(-adam_lr))
                optimizer = optax.chain(*components)
                return optimizer

            transformations = {
                "muadam2": muadam2_transform(),
                "adamw": adamw_transform(),
            }

            return optax.multi_transform(transformations, self.create_mask)

        return optax.inject_hyperparams(optimizer)(learning_rate=learning_rate_schedule)

    def create_mask(self, params):
        """
        Creates a mask that labels parameters as 'muadam2' or 'adamw' based on their
        dimensionality and module path, using AdamW for Embedding and lm_head parameters.
        """
        paths = leaf_key_paths(params)

        def mask_fn(param, path):
            path_str = ".".join(path) if isinstance(path, (list, tuple)) else str(path)
            if "Embedding" in path_str or "lm_head" in path_str:
                return "adamw"
            elif isinstance(param, Linear):
                # muadam2 for linear layers
                return dataclasses.replace(param, weight="muadam2", bias="adamw" if param.bias is not None else None)
            else:
                return "adamw"

        return jax.tree_util.tree_map(mask_fn, params, paths, is_leaf=lambda x: isinstance(x, Linear))


class ScaleByMuAdam2State(NamedTuple):
    """State for the Mars algorithm."""
    count: jaxtyping.Array  # shape=(), dtype=jnp.int32.
    momentum_buffer: optax.Updates
    nu: optax.Updates



def scale_with_muadam2(momentum=0.95, b2 = 0.95, nesterov=True, steps=5, eps = 1e-7):
    def init_fn(params):
        momentum_buffer = otu.tree_zeros_like(params)  # First moment
        nu = otu.tree_zeros_like(params)  # Second moment
        return ScaleByMuAdam2State(count=jnp.zeros([], jnp.int32), momentum_buffer=momentum_buffer, nu = nu )

    def update_fn(updates, state, params=None):
        buf = state.momentum_buffer
        buf = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + g,
            buf,
            updates,
            is_leaf=lambda x: x is None,
        )
        updates = jax.tree.map(
            lambda m, g: None if g is None else momentum * m + g,
            buf,
            updates,
            is_leaf=lambda x: x is None,
        )
        buf_leaves = jax.tree_util.tree_leaves(buf)
        h_leaves = jax.tree_util.tree_leaves(updates)

        stats: dict[str, Any] = {
            "optim/param_norm_L2": jnp.sqrt(sum(jnp.sum(p**2) for p in jax.tree_util.tree_leaves(params))),
            "optim/momentum_norm_L2": jnp.sqrt(sum(jnp.sum(m**2) for m in buf_leaves)),
            "optim/update_norm_L2": jnp.sqrt(sum(jnp.sum(h**2) for h in h_leaves))}
        levanter.tracker.jit_log(stats, step=state.count)
        def transform_linear_layer(layer: haliax.nn.Linear):
            assert layer.weight.ndim == 2

            updated_weight_array = zeropower_via_newtonschulz5(layer.weight.array, steps=steps)

            scale = jnp.sqrt(jnp.maximum(1, updated_weight_array.shape[0] / updated_weight_array.shape[1]))
            updated_weight_array *= scale

            updated_weight = dataclasses.replace(layer.weight, array=updated_weight_array)

            return dataclasses.replace(layer, weight=updated_weight)  # type: ignore

        updates = map_flattened_linear_layers(transform_linear_layer, updates)
        nu = otu.tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
        nu_hat = otu.tree_bias_correction(nu, b2, state.count + 1)
        adam_updates = jax.tree.map(lambda m, v: None if m is None else m / (jnp.sqrt(v + eps) + eps),
        updates,
        nu_hat,
        is_leaf=lambda x: x is None,
        )

        return updates, ScaleByMuAdam2State(count = state.count + 1, momentum_buffer=buf, nu = nu)

    return optax.GradientTransformation(init_fn, update_fn)


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
    return X