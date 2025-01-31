from typing import Any, Union
from netket.utils.types import NNInitFunc

import jax
import jax.numpy as jnp
import jax.nn.initializers as init

import flax.linen as nn
from flax.linen.dtypes import promote_dtype


def vec_to_matrix(kernel, shape: tuple, indices):
    r'''
    Maps a kernel vector to a lower-triangular matrix of indicated shape.
    This function basically does jnp.zeros(shape).at[indices].set(kernel) in an optimized way
    Args:
        kernel: vector with given parameters
        shape: resulting shape of the matrix
        indices: mapping between kernel and final matrix
    Returns: 
        parameter matrix with `kernel` at `indices` and zeros otherwise.
    '''

    if jnp.issubdtype(kernel.dtype, jnp.complex128):
        Wr = (
            jnp.zeros(shape, dtype=jnp.float64)
            .at[indices]
            .set(kernel.real, unique_indices=True, indices_are_sorted=True)
        )
        Wi = (
            jnp.zeros(shape, dtype=jnp.float64)
            .at[indices]
            .set(kernel.imag, unique_indices=True, indices_are_sorted=True)
        )
        W = Wr + 1j * Wi

    else:
        W = (
            jnp.zeros(shape, dtype=kernel.dtype)
            .at[indices]
            .set(kernel, unique_indices=True, indices_are_sorted=True)
        )
    return W
