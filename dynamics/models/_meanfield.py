from typing import Any, Union
from netket.utils.types import NNInitFunc

import jax.numpy as jnp
import jax.nn.initializers as init
from jax import vmap

import flax.linen as nn


class MeanField(nn.Module):
    r"""
    log \psi(z) = \sum_i \log( \phi(z_i) )
    Notice that any one-body Jastrow term can be absorbed in the mean-field part, so we do not need it
    """

    param_dtype: Any = jnp.complex128
    """The dtype of the weights."""

    kernel_init: NNInitFunc = init.constant(1)
    """Initializer for the mean-field parameters."""

    @nn.compact
    def __call__(self, x):
        """
        x : (Ns,N)
        """
        N = x.shape[-1]

        phi = self.param("phi", self.kernel_init, (N, 2), self.param_dtype)

        # compute the mean field part
        def one_mf(state):
            indices = jnp.array(
                (1 + state) / 2, dtype=int
            )  # convert spins to indices -1,+1 -> 0,1

            # each sigma selects its phi
            def one_phi(i, index):
                return phi[i, index]

            # now for all components
            return vmap(one_phi)(jnp.arange(N), indices)

        # finally, differently for all samples
        mf = vmap(one_mf)(x)

        return jnp.sum(jnp.log(mf), axis=-1)
