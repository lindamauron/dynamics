import numpy as np

from typing import Any, Tuple
from netket.utils.types import NNInitFunc

import jax
import jax.numpy as jnp
import jax.nn.initializers as init

import flax.linen as nn
from flax.linen.dtypes import promote_dtype

from ._vec_to_matrix import vec_to_matrix as vec_to_tril
from ._general_jastrow import JastrowNBody

class Rotation(nn.Module):
    r"""
    Defines an ansatz:
    ..math::
        |\psi> = cos(theta/2) |0> + exp(i*phi) sin(theta/2) |1>
    where the angles theta and phi are the variational parameters.
    """
    
    angles_init : Tuple[NNInitFunc] = (init.constant(0.0),init.constant(0.0))
    """Initializer for the angles."""
    
    param_dtype : Any = np.float64
    """The dtype of the angles."""
 
    @nn.compact
    def __call__(self, x):
        '''
        x : (...,N)
        '''
        N = x.shape[-1]

        # Define the variational parameters
        theta = self.param(
            "θ", self.angles_init[0], (N,), self.param_dtype
        )/2
        
        phi = self.param(
            "φ", self.angles_init[1], (N,), self.param_dtype
        )
        mask = jnp.array((1-x)//2, dtype=jnp.int8)
        R = jnp.cos(theta) * mask + jnp.exp(1j*phi)*jnp.sin(theta) * (1-mask)
        
        return jnp.sum(jnp.log(R), axis=-1)


class JasRot(nn.Module):
    features: Tuple[int] = (1,2)
    """The number of interactions."""

    rotations_init: Tuple[NNInitFunc] = (init.constant(0.0),init.constant(0.0))
    """Initializers for the angles."""

    kernel_init: NNInitFunc = init.normal(stddev=0.01)
    """Initializer for the jastrow parameters."""

    param_dtype: Any = jnp.float64
    """The dtype of the weights."""


    @nn.compact
    def __call__(self, x_in):
        if self.param_dtype == jnp.complex128:
            raise ValueError("Complex dtype not supported for JR")
        
        J = 0
        for f in self.features:
            if f==1:
                J += Rotation(
                    angles_init=self.rotations_init,
                    param_dtype=self.param_dtype,
                    name=f"R",
                    )(x_in)
            else:
                mask = self.param(
                    f"mask_{f}", init.zeros, (1,), jnp.float64
                )
                Jre = JastrowNBody(
                    n=f,
                    kernel_init=self.kernel_init,
                    param_dtype=self.param_dtype,
                    name=f"W{f}_re",
                )
                Jim = JastrowNBody(
                    n=f,
                    kernel_init=self.kernel_init,
                    param_dtype=self.param_dtype,
                    name=f"W{f}_im",
                )
                J += mask*(Jre(x_in) + 1j*Jim(x_in))
                
        return J