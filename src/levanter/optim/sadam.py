import dataclasses
from dataclasses import dataclass
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import optax
from optax import tree_utils as otu
from typing import Any, Optional

import haliax
from haliax.nn import Linear

from levanter.optim.config import OptimizerConfig
from levanter.utils.jax_utils import leaf_key_paths
@OptimizerConfig.register_subclass("sadam")
@dataclass
class SAdamConfig(OptimizerConfig):
    weight_decay: float = 0.1
    beta1: float = 0.9
    # cf https://docs.mosaicml.com/projects/composer/en/latest/api_reference/generated/composer.optim.DecoupledAdamW.html
    # https://x.com/giffmana/status/1692641748445438301
    beta2: float = 0.95
    epsilon: float = 1e-8
    weight_lr_power: float = 2.0 # no idea what is this but this is in official implementation
    max_grad_norm: Optional[float] = 1.0
    haps: Optional[list[int]] = None
    schedule_list: Optional[list[str]] = None


    def build(self, num_train_steps):
        """Creates the optimizer"""
        # indirection makes it work with optax.inject_hyperparams so we can log the learning rate
        def _optimizer(learning_rate):
            components = []

            if self.max_grad_norm:
                components.append(optax.clip_by_global_norm(self.max_grad_norm))

            components.append(optax.scale_by_adam(0, self.beta2, self.epsilon))

            if self.weight_decay > 0:
                components.append(optax.add_decayed_weights(self.weight_decay, self.build_weight_decay_mask()))

            # - learning rate for descent
            components.append(optax.scale(-learning_rate))

            optimizer = optax.chain(*components)
            optimizer = optax.contrib.schedule_free(optimizer, learning_rate, b1 = self.beta1,  weight_lr_power=self.weight_lr_power)
            return optimizer

        optimizer =  optax.inject_hyperparams(_optimizer)(learning_rate=self.lr_scheduler(num_train_steps))
        return optimizer
