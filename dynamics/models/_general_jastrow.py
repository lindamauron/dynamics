from typing import Any, Union
from netket.utils.types import NNInitFunc

import jax
import jax.numpy as jnp
import jax.nn.initializers as init

import flax.linen as nn
from flax.linen.dtypes import promote_dtype

class JastrowManyBody(nn.Module):
    r"""
    n-body Jastrow implementation, as the n-body matrix is represented as a product of two bodies:
    ..math::
        W_ijkl... = W_ij W_jk W_kl ...

    where each :math:`W` is a 2-body lower triangular matrix. The log_value is then obtained as
    ..math::
        logpsi(z) = \sum_i zi \sum_j W_ij z_j \sum_k W_jk z_k \sum_l W_kl z_l ...

    Warning: do not initialize at 0, otherwise the gradient are also 0 and you get stuck
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
        x : (Ns,N)
        """
        N = x_in.shape[-1]
        il = jnp.tril_indices(N, k=-1)

        # 4 bodies jastrow
        kernel = self.param(
            f"kernel", self.kernel_init, (N * (N - 1) // 2,), self.param_dtype
        )
        W = jnp.zeros((N, N), dtype=self.param_dtype).at[il].set(kernel)

        W, x_in = promote_dtype(W, x_in, dtype=None)

        # initialize
        z = jnp.einsum("ni,ji->nj", x_in, W)

        z = jax.lax.fori_loop(
            0, self.n - 2, lambda i, z: jnp.einsum("nj,kj,nj->nk", z, W, x_in), z
        )

        return jnp.einsum("nl,nl->n", z, x_in)


def JastrowNBody(n, *args, **kwargs):
    r"""
    Generalizes the Jastrow formalism to any n>0.
    """
    if not (isinstance(n, int) and n > 0):
        raise ValueError(
            f"The number of Jastrow interactions should be a stricly positive integer, instead got {n}."
        )

    if n == 1:
        return JastrowOneBody(n=n, *args, **kwargs)
    else:
        return JastrowManyBody(n=n, *args, **kwargs)


class JastrowOneBody(nn.Module):
    r"""
    1-body Jastrow implementation, with the log_value obtained as
    ..math::
        logpsi(z) = \sum_i zi W_i

    This ansatz then works as a mean-field ansatz, but respects the Jastrow formalism.
    """

    n: int = 1
    """The number of interactions."""

    param_dtype: Any = jnp.complex128
    """The dtype of the weights."""

    kernel_init: NNInitFunc = init.normal(1e-2)
    """Initializer for the jastrow parameters."""

    @nn.compact
    def __call__(self, x_in):
        """
        x : (Ns,N)
        """
        N = x_in.shape[-1]

        W = self.param("kernel", self.kernel_init, (N,), self.param_dtype)

        return jnp.einsum("i,ni->n", W, x_in)


class JasMultipleBodies(nn.Module):
    r"""
    Sum of n-body Jastrow implementation. For each n, the n-body matrix is represented as a product of two bodies:
    ..math::
        W_ijkl... = W_ij W_jk W_kl ...

    where each :math:`W` is a 2-body lower triangular matrix. The log_value is then obtained as
    ..math::
        logpsi_n(z) = \sum_i zi \sum_j W_ij z_j \sum_k W_jk z_k \sum_l W_kl z_l ...

    and the total log-value is obtained as the sum over all features (i.e. all interactions)
    ..math:
        logpsi(z) = \sum_n logpsi_n(z)
    """
    features: Union[tuple[int, ...], int] = (1, 2)
    """The featues of the Jastrow object, i.e. the number of bodies."""

    param_dtype: Any = jnp.complex128
    """The dtype of the weights."""

    kernel_init: NNInitFunc = init.normal(1e-2)
    """Initializer for the jastrow parameters."""

    field_init: NNInitFunc = init.normal(1e-2)
    """Initializer for the mean-field parameters."""

    @nn.compact
    def __call__(self, x_in):
        """
        x : (Ns,N)
        """
        init = lambda n: self.field_init if n == 1 else self.kernel_init

        J = 0
        for n in self.features:
            J += JastrowNBody(
                n=n, param_dtype=self.param_dtype, kernel_init=init(n), name=f"W{n}"
            )(x_in)

        return J
