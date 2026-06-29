import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from .base import MyEfficientOperator
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class SSOperator(MyEfficientOperator):
    r'''
    Operator multiplying two spins \vec{S}_i \cdot \vec{S}_j
    '''
    _extra_fields = ['signs']

    def __init__(self, hilbert, sites, coeffs=1.0, signs=1.0):
        super().__init__(hilbert, sites=sites, coeffs=coeffs)        
        
        if isinstance(signs, (int, float, complex, np.generic)):
            self._signs = jnp.full(self._sites.shape[0] if self._sites is not None else 1, signs, dtype=float)
        else:
            self._signs = jnp.asarray(signs)

    @property
    def max_conn_size(self):
        return len(self._sites) + 1
    
    @partial(jax.jit, static_argnums=(0,))
    def get_conn_padded(self, sigma):
        r'''
        for samples sigma (...,Ns,N),
        returns x_padded (...,Nc,N) and m_padded (...,Nc) 
        '''
        N = self.hilbert.size
        
        # diagonal part :len(sites)
        # x_padded for this part is unchanged
        x_diag = jnp.expand_dims(sigma, axis=-2)
        m_diag = jnp.expand_dims( (self._coeffs[None,:] * sigma[..., self._sites[:,0]] * sigma[..., self._sites[:,1]] ).sum(-1), axis=-1)
        # jax.debug.print("x_diag shape: {}", x_diag.shape)
        # jax.debug.print("m_diag shape: {}", m_diag.shape)

        # XY part len(sites):
        same_site = (self._sites[:, 0] == self._sites[:, 1])
        x_off = jax.vmap(
            lambda x,e: x.at[...,e[0]].set(x[...,e[1]]).at[...,e[1]].set(x[...,e[0]]), in_axes=(None,0), out_axes=-2
            )(sigma, self._sites)
        m_off = self._signs[None,:] * self._coeffs[None,:] * (1-sigma[..., self._sites[:,0]] * sigma[..., self._sites[:,1]] + 2.0 * same_site[None, :])

        # jax.debug.print("x_off shape: {}", x_off.shape)
        # jax.debug.print("m_off shape: {}", m_off.shape)
        x_padded = jnp.concatenate([x_diag, x_off], axis=-2)
        m_padded = jnp.concatenate([m_diag, m_off], axis=-1)

        return x_padded, m_padded
    