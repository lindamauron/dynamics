import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from .base import MyEfficientOperator
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class SzOperator(MyEfficientOperator):
    def __init__(self, hilbert, sites=None, coeffs=1.0):
        if sites is None:
            sites = np.arange(hilbert.size)
        else:
            try:
                sites = np.array(sites)
            except TypeError:
                sites = sites
        assert sites.ndim==1, "sites should be an array of single site indices"
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
        m_padded = jnp.expand_dims( (self._coeffs[None,:] * sigma[..., self._sites]).sum(-1), axis=-1)

        return x_padded, m_padded
    

@register_pytree_node_class
class SzzOperator(MyEfficientOperator):
    def __init__(self, hilbert, sites, coeffs=1.0):
        try:
            sites = np.array(sites)
        except TypeError:
            sites = sites
        if sites.ndim==1 and len(sites)==2:
            sites = np.expand_dims(sites, axis=0)
        else:
            assert sites.ndim==2 and sites.shape[1]==2, "sites should be an array of pairs of site indices"
        super().__init__(hilbert, sites=sites, coeffs=coeffs)
    
    @property
    def max_conn_size(self):
        return 1
    
    @partial(jax.jit, static_argnums=(0,))
    def get_conn_padded(self, sigma):
        r'''
        returns x_padded (Ns,Nc,N) and m_padded (Ns,Nc) 
        '''
        N = self.hilbert.size

        x_padded = jnp.expand_dims(sigma, axis=-2)
        m_padded = jnp.expand_dims( (self._coeffs[None,:] * sigma[..., self._sites[:,0]] * sigma[..., self._sites[:,1]] ).sum(-1), axis=-1)

        return x_padded, m_padded

@register_pytree_node_class
class SzzzzOperator(MyEfficientOperator):
    def __init__(self, hilbert, sites, coeffs=1.0):
        try:
            sites = np.array(sites)
        except TypeError:
            sites = sites
        if sites.ndim==1 and len(sites)==4:
            sites = np.expand_dims(sites, axis=0)
        else:
            assert sites.ndim==2 and sites.shape[1]==4, "sites should be an array of four site indices"
        super().__init__(hilbert, sites=sites, coeffs=coeffs)
    
    @property
    def max_conn_size(self):
        return 1
    
    @partial(jax.jit, static_argnums=(0,))
    def get_conn_padded(self, sigma):
        r'''
        returns x_padded (Ns,Nc,N) and m_padded (Ns,Nc) 
        '''
        N = self.hilbert.size

        x_padded = jnp.expand_dims(sigma, axis=-2)
        m_padded = jnp.expand_dims( (
            self._coeffs[None,:] * sigma[..., self._sites[:,0]] * sigma[..., self._sites[:,1]] * sigma[..., self._sites[:,2]] * sigma[..., self._sites[:,3]] 
            ).sum(-1), axis=-1)

        return x_padded, m_padded
    
