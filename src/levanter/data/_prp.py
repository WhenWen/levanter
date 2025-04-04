import typing

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np


class Permutation:
    # Pseudo-Random Permutation Code
    """A stateless pseudo-random permutation.

    This class generates a pseudo-random permutation of a given length. The permutation is generated using a PRNG
    with a fixed key. The permutation is generated by finding a random `a` and `b` such that `gcd(a, length) == 1` and
    then computing the permutation as `p(x) = (a * x + b) % length`.

    This is not a very good PRP, but it is probably good enough for our purposes.
    """
    # TODO: is it actually good enough for our purposes?

    def __init__(self, length, prng_key):
        self.length = length
        # Convert jax.random.PRNGKey to numpy.random.Generator
        self.rng = np.random.Generator(np.random.PCG64(jrandom.randint(prng_key, (), 0, 2**30).item()))
        self.a, self.b = self._generate_permutation_params()  # Generate a and b in init

    def _generate_permutation_params(self):
        length = self.length
        rng = self.rng

        if length == 1:
            return 1, 0

        while True:
            a = rng.integers(1, length)
            if np.gcd(a, length) == 1:
                break

        b = rng.integers(0, length)  # b can be in [0, length-1]
        return a, b

    @typing.overload
    def __call__(self, indices: int) -> int:
        ...

    @typing.overload
    def __call__(self, indices: np.ndarray) -> np.ndarray:
        ...

    def __call__(self, indices):
        a = self.a
        b = self.b
        length = self.length

        was_int = False
        if isinstance(indices, np.ndarray | jnp.ndarray):
            if np.any(indices < 0) or np.any(indices >= self.length):
                raise IndexError(f"index {indices} is out of bounds for length {self.length}")
        else:
            if indices < 0 or indices >= self.length:
                raise IndexError(f"index {indices} is out of bounds for length {self.length}")

            indices = np.array(indices)
            was_int = True

        out = (a * indices + b) % length  # Compute permutation on-the-fly

        if was_int:
            return int(out)
        return out
