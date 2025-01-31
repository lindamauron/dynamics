from typing import Any

import numpy as np

import jax
from jax import numpy as jnp
from flax import linen as nn
from jax.nn.initializers import normal
import netket as nk

from netket.utils import HashableArray
from netket.utils.types import NNInitFunc
from netket import nn as nknn

from ._dense_jastrows import JasTwoBody

default_kernel_init = normal(stddev=0.01)


class JRBM(nn.Module):
    r"""A restricted boltzman Machine, equivalent to a 2-layer FFNN with a
    nonlinear activation function in between, coupled to a Jastrow factor as external layer.
    ..math::
        log \psi(z) = \sum_i log \cosh(\sum_j W_{ij} x_j + b_i) + \sum_{ij} x_i J_{ij} x_j
    """

    param_dtype: Any = np.float64
    """The dtype of the weights."""
    activation: Any = nknn.log_cosh
    """The nonlinear activation function."""
    alpha: float | int = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    use_visible_jastrow: bool = True
    """if True adds a jastrow factor to the output."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""
    visible_jastrow_init: NNInitFunc = default_kernel_init
    """Initializer for the visible jastrow factor."""

    @nn.compact
    def __call__(self, input):
        z = nk.models.RBM(self.param_dtype, self.activation, self.alpha, self.use_hidden_bias, self.use_visible_bias, self.use_visible_jastrow, self.precision, self.kernel_init, self.hidden_bias_init, self.visible_bias_init, self.visible_jastrow_init)(input)

        if self.use_visible_jastrow:
            j = JasTwoBody(
                self.param_dtype, self.visible_jastrow_init
                )
            
            z = z + j(input)

        return z
