import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from .base import MyEfficientOperator
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class SxOperator(MyEfficientOperator):
    def __init__(self, hilbert, sites=None, coeffs=1.0):
        if sites is None:
            sites = np.arange(hilbert.size)
        else:
            try:
                sites = np.array(sites)
            except TypeError:
                # JAX may pass placeholder objects during pytree shape inference
                sites = sites
        assert sites.ndim==1, f"sites should be an array of single site indices (got {sites})"
        super().__init__(hilbert, sites=sites, coeffs=coeffs)
    
    @property
    def max_conn_size(self):
        return len(self._sites)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_conn_padded(self, sigma):
        r'''
        for samples sigma (...,Ns,N),
        returns x_padded (...,Nc,N) and m_padded (...,Nc) 
        '''
        N = self.hilbert.size

        # x_padded = jnp.repeat(sigma[:,None,:], repeats=N, axis=-2)
        # print(sigma.shape, x_padded.shape)
        x_padded = jax.vmap(lambda x,i: x.at[...,i].multiply(-1), in_axes=(None,0), out_axes=-2)(sigma, self._sites)
        m_padded = jnp.repeat(self._coeffs[None,:], repeats=sigma.shape[0], axis=0) #* jnp.ones((*sigma.shape[:-1],len(self._sites)))

        return x_padded, m_padded
    
@register_pytree_node_class
class SxxOperator(MyEfficientOperator):
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
        return len(self._sites)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_conn_padded(self, sigma):
        r'''
        for samples sigma (...,Ns,N),
        returns x_padded (...,Nc,N) and m_padded (...,Nc) 
        '''
        N = self.hilbert.size

        x_padded = jnp.repeat(jnp.expand_dims(sigma, axis=-2), repeats=len(self._sites), axis=-2)
        x_padded = jax.vmap(
            lambda x,e: x.at[...,e[0]].multiply(-1).at[...,e[1]].multiply(-1), in_axes=(-2,0), out_axes=-2
            )(x_padded, self._sites)

        m_padded = jnp.repeat(self._coeffs[None,:], repeats=sigma.shape[0], axis=0) #* jnp.ones((*sigma.shape[:-1],len(self._sites)))

        return x_padded, m_padded

@register_pytree_node_class
class SxxxxOperator(MyEfficientOperator):
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
        return len(self._sites)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_conn_padded(self, sigma):
        r'''
        for samples sigma (...,Ns,N),
        returns x_padded (...,Nc,N) and m_padded (...,Nc) 
        '''
        N = self.hilbert.size

        x_padded = jnp.repeat(jnp.expand_dims(sigma, axis=-2), repeats=len(self._sites), axis=-2)
        x_padded = jax.vmap(
            lambda x,e: x.at[...,e[0]].multiply(-1).at[...,e[1]].multiply(-1).at[...,e[2]].multiply(-1).at[...,e[3]].multiply(-1), in_axes=(-2,0), out_axes=-2
            )(x_padded, self._sites)

        m_padded = jnp.repeat(self._coeffs[None,:], repeats=sigma.shape[0], axis=0) #* jnp.ones((*sigma.shape[:-1],len(self._sites)))

        return x_padded, m_padded
    
