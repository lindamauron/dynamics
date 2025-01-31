from typing import Any, Union
from netket.utils.types import NNInitFunc

import jax
import jax.numpy as jnp
import jax.nn.initializers as init

import flax.linen as nn
from flax.linen.dtypes import promote_dtype

from ._vec_to_matrix import vec_to_matrix

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

    match n:
        case 1:
            return JasOneBody(*args, **kwargs)
        case 2:
            return JasTwoBody(*args, **kwargs)
        case 3:
            return JasThreeBody(*args, **kwargs)
        case 4:
            return JasFourBody(*args, **kwargs)
        case 5:
            return JasFiveBody(*args, **kwargs)
        case _:
            return JasNBody(n=n, *args, **kwargs)


class JasOneBody(nn.Module):
    r"""
    1-body Jastrow implementation, with the log_value obtained as
    ..math::
        log \psi(z) = \sum_i zi W_i

    This ansatz then works as a mean-field ansatz, but respects the Jastrow formalism.
    """

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

        W = self.param("kernel", self.kernel_init, (N,), self.param_dtype)

        return jnp.einsum("i,ni->n", W, x_in)


class JasTwoBody(nn.Module):
    r"""
    log \psi(z) = \sum_ij W_ij z_i z_j
    """

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
        kernel2 = self.param(
            "kernel", self.kernel_init, (N * (N - 1) // 2,), self.param_dtype
        )
        W2 = vec_to_matrix(kernel2, (N, N), il)

        J2 = jnp.einsum("ij,ni,nj->n", W2, x_in, x_in)

        return J2


class JasThreeBody(nn.Module):
    r"""
    log \psi(z) = \sum_ijk W_ijk z_i z_j z_k
    """

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
        # 3 bodies jastrow
        triu = jnp.triu(jnp.ones((N, N, N), dtype=self.param_dtype), k=1)
        indices = jnp.nonzero(
            triu * triu.transpose(1, 2, 0), size=N * (N - 1) * (N - 2) // 6
        )
        kernel3 = self.param(
            "kernel", self.kernel_init, (N * (N - 1) * (N - 2) // 6,), self.param_dtype
        )
        W3 = vec_to_matrix(kernel3, (N, N, N), indices)

        J3 = jnp.einsum("ijk,ni,nj,nk->n", W3, x_in, x_in, x_in)

        return J3


class JasFourBody(nn.Module):
    r"""
    log \psi(z) = \sum_ijkl W_ijkl z_i z_j z_k z_l
    """

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

        # 4 bodies jastrow
        triu = jnp.triu(jnp.ones((N, N, N, N), dtype=self.param_dtype), k=1)
        indices = jnp.nonzero(
            triu * (triu.transpose(1, 2, 3, 0)) * (triu.transpose(2, 3, 0, 1)),
            size=N * (N - 1) * (N - 2) * (N - 3) // 24,
        )
        kernel4 = self.param(
            "kernel",
            self.kernel_init,
            (N * (N - 1) * (N - 2) * (N - 3) // 24,),
            self.param_dtype,
        )
        W4 = vec_to_matrix(kernel4, (N, N, N, N), indices)

        J4 = jnp.einsum("ijkl,ni,nj,nk,nl->n", W4, x_in, x_in, x_in, x_in)

        return J4


class JasFiveBody(nn.Module):
    r"""
    log \psi(z) = \sum_ijklm W_ijklm z_i z_j z_k z_l z_m
    """

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

        # 5 bodies jastrow
        triu = jnp.triu(jnp.ones((N, N, N, N, N), dtype=self.param_dtype), k=1)
        indices = jnp.nonzero(
            triu
            * (triu.transpose(1, 2, 3, 4, 0))
            * (triu.transpose(2, 3, 4, 0, 1))
            * (triu.transpose(3, 4, 0, 1, 2)),
            size=N * (N - 1) * (N - 2) * (N - 3) * (N - 4) // 120,
        )
        kernel5 = self.param(
            "kernel",
            self.kernel_init,
            N * (N - 1) * (N - 2) * (N - 3) * (N - 4) // 120,
            self.param_dtype,
        )
        W5 = vec_to_matrix(kernel5, (N, N, N, N, N), indices)

        J5 = jnp.einsum("ijklm,ni,nj,nk,nl,nm->n", W5, x_in, x_in, x_in, x_in, x_in)

        return J5


class JasNBody(nn.Module):
    r"""
    log \psi(z) = \sum_ijk... W_ijk... z_i z_j z_k ...
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
        raise NotImplementedError
        N = x_in.shape[-1]

        triu = jnp.triu(jnp.ones((N,) * self.n, dtype=self.param_dtype), k=-1)
        idxs = jnp.arange(self.n, dtype=jnp.int32)
        size = N
        for i in range(1, self.n):
            idxs = jnp.roll(idxs, -1)
            triu = triu * triu.transpose(*idxs)
            size = size * (N - i) // (i + 1)
        size = size * (N - self.n + 1) // (self.n)
        indices = jnp.nonzero(triu, size=size)

        # n bodies jastrow
        kernel = self.param(f"W{self.n}", self.kernel_init, size, self.param_dtype)
        W = vec_to_matrix(kernel, (N,) * self.n, indices)

        W, x_in = promote_dtype(W, x_in, dtype=None)

        # initialize
        z = jax.vmap(jnp.dot, in_axes=(None, 0))(W, x_in)

        for i in range(self.n - 1):
            z = jax.vmap(jnp.dot, in_axes=(0, 0))(z, x_in)

        return z
