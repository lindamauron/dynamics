from typing import Any, Union
from netket.utils.types import NNInitFunc

import jax.numpy as jnp
import jax.nn.initializers as init
from jax import vmap

import flax.linen as nn
from ._general_jastrow import JastrowManyBody


class MF(nn.Module):
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

        phi = self.param("ϕ", self.kernel_init, (N, 2), self.param_dtype)

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


class JMFMultipleBodies(nn.Module):
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


def JastrowNBody(n, *args, **kwargs):
    r"""
    Generalizes the Jastrow formalism to any n>0.
    """
    if not (isinstance(n, int) and n > 0):
        raise ValueError(
            f"The number of Jastrow interactions should be a stricly positive integer, instead got {n}."
        )

    if n == 1:
        return MF(*args, **kwargs)
    else:
        return JastrowManyBody(n=n, *args, **kwargs)
