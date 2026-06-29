import jax
import jax.numpy as jnp
from functools import partial
from .base import MyEfficientOperator
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class IdOperator(MyEfficientOperator):
    def __init__(self, hilbert, sites=None, coeffs=1.0):
        super().__init__(hilbert, sites=sites, coeffs=coeffs)
    
    @property
    def max_conn_size(self):
        return 1
    
    @partial(jax.jit, static_argnums=(0,))
    def get_conn_padded(self, sigma):
        r'''
        for samples sigma (...,Ns,N),
        returns x_padded (...,Nc,N) and m_padded (...,Nc) 
        '''
        N = self.hilbert.size

        x_padded = jnp.expand_dims(sigma, axis=-2)
        m_padded = jnp.ones((*sigma.shape[:-1], 1))

        return x_padded, self._coeffs*m_padded
    
