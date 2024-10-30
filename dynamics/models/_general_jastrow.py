from typing import Any, Union
from netket.utils.types import NNInitFunc

import jax.numpy as jnp
import jax.nn.initializers as init

import flax.linen as nn


class JastrowSum(nn.Module):
    r"""
    Sum of n-body Jastrow implementation. The total log-value is obtained as the sum over all features (i.e. all interactions)
    ..math:
        log \psi(z) = \sum_n log \psi_n(z)

    where each n-body Jastrow correlator is defined as
    ..math::
        log \psi_n(z) = \sum_ijkl... W_ijkl... z_i z_j z_k z_l ...

    In the case of `factorized` Jastrows, the n-body correlator is defined as
    ..math::
        W_ijkl... = W_ij W_jk W_kl ...
    and the rest stays the same.

    Finally, one can choose to replace the 1-body Jastrow by a mean-field Ansatz which is defined as
    ..math::
        log \psi_1(z) = \sum_i \log( \phi(z_i) )
    """

    features: Union[tuple[int, ...], int] = (1, 2)
    """The features of the Jastrow object, i.e. the number of bodies."""

    factorized: Union[tuple[bool], bool] = True
    """Boolean indicating whether to use the factorized representation of the Jastrow kernel."""

    mean_field: bool = False
    """Boolean indicating whether to use a mean-field Ansatz instead of the 1-body Jastrow kernel."""

    param_dtype: Any = jnp.complex128
    """The dtype of the weights."""

    kernel_init: NNInitFunc = init.normal(1e-2)
    """Initializer for the jastrow parameters."""

    field_init: NNInitFunc = init.normal(1e-2)
    """Initializer for the one-body parameters."""

    def setup(self):
        ## modify `factorized` to be a tuple
        if isinstance(self.factorized, bool):
            factorized = (self.factorized,) * len(self.features)
        else:
            factorized = tuple(self.factorized)
        if len(factorized) != len(self.features):
            raise ValueError(
                f"`factorized` should be a tuple of booleans of the same length as features, instead got {self.factorized} and {self.features}."
            )

        ## make things callable
        init = lambda n: self.field_init if n == 1 else self.kernel_init
        if self.mean_field:
            name = lambda n: f"W{n}" if n > 1 else "MF"
        else:
            name = lambda n: f"W{n}"

        Js = []
        for i, n in enumerate(self.features):
            Js.append(
                JastrowNBody(
                    n=n,
                    factorized=factorized[i],
                    param_dtype=self.param_dtype,
                    kernel_init=init(n),
                    name=name(n),
                )
            )

        self.jastrow_modules = Js

        return

    def __call__(self, x_in):
        """
        x : (Ns, N)
        """

        return sum([J(x_in) for J in self.jastrow_modules])


def JastrowNBody(n, factorized, *args, **kwargs):
    """
    Factory function to create Jastrow interaction objects.

    Args:
        n: The number of Jastrow interactions. Must be a strictly positive integer.
        factorized: If True, use the factorized Jastrow interaction. Otherwise, use the dense Jastrow interaction.

    Returns:
        An instance of the appropriate Jastrow interaction class based on the input parameters.
    """
    if not (isinstance(n, int) and n > 0):
        raise ValueError(
            f"The number of Jastrow interactions should be a stricly positive integer, instead got {n}."
        )

    if n == 1:
        if kwargs["name"] == "MF":
            from ._meanfield import MeanField

            return MeanField(*args, **kwargs)
        else:
            from ._dense_jastrows import JasOneBody

            return JasOneBody(*args, **kwargs)
    else:
        if factorized:
            from ._factorized_jastrow import JastrowNBody as JasNBody
        else:
            from ._dense_jastrows import JastrowNBody as JasNBody

        return JasNBody(n=n, *args, **kwargs)
