from typing import Any, Union
from netket.utils.types import NNInitFunc

import jax
import jax.numpy as jnp
import jax.nn.initializers as init

import flax.linen as nn
from flax.linen.dtypes import promote_dtype

from ._vec_to_matrix import vec_to_matrix as vec_to_tril

def JastrowNBody(n, *args, **kwargs):
    """
    Factory function to create Jastrow interaction objects.

    Args:
        n: The number of Jastrow interactions. Must be a strictly positive integer.

    Returns:
        An instance of the appropriate Jastrow interaction class based on the input parameters.
    """
    if not (isinstance(n, int) and n > 0):
        raise ValueError(
            f"The number of Jastrow interactions should be a stricly positive integer, instead got {n}."
        )

    if n == 1:
        from ._dense_jastrows import JasOneBody

        return JasOneBody(*args, **kwargs)
    else:
        return JasNBody(n=n, *args, **kwargs)


class JasNBody(nn.Module):
    r"""
    n-body Jastrow implementation
    ..math::
        log \psi(z) = \sum_i z_i \sum_j W_ij z_j \sum_k W_jk z_k \sum_l W_kl z_l ...

    where the n-body matrix is represented as a product of two bodies:
    ..math::
        W_ijkl... = W_ij W_jk W_kl ...

    where each :math:`W` is a 2-body lower triangular matrix.

    Warning: do not initialize at 0, otherwise the gradients are also 0 and you get stuck
    """

    n: int = 2
    """The number of interactions."""

    param_dtype: Any = jnp.complex128
    """The dtype of the weights."""

    kernel_init: NNInitFunc = init.normal(1e-2)
    """Initializer for the jastrow parameters."""

    @nn.compact
    def __call__(self, x_in):
        """
        x_in : (Ns, N)
        """
        N = x_in.shape[-1]
        il = jnp.triu_indices(N, k=+1)

        # 4 bodies jastrow
        kernel = self.param(
            f"kernel", self.kernel_init, (N * (N - 1) // 2,), self.param_dtype
        )
        W = vec_to_tril(kernel, (N, N), il )

        W, x_in = promote_dtype(W, x_in, dtype=None)

        # initialize
        z = jnp.einsum("ni,ji->nj", x_in, W)

        z = jax.lax.fori_loop(
            0, self.n - 2, lambda i, z: jnp.einsum("nj,kj,nj->nk", z, W, x_in), z
        )

        return jnp.einsum("nl,nl->n", z, x_in)
