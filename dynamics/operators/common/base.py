# This class is defined to override NetKet's Discrete Operator interface and define
# more efficient operators for specific uses. The observed gain is a factor 10 in memory and computation time
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
from functools import partial
from numbers import Number
from jax.tree_util import register_pytree_node_class
from netket.utils.array import HashableArray

def _is_scalar_like(other):
    return isinstance(
        other,
        (int, float, complex, np.generic, jnp.ndarray, jax.Array),
    )
    
@register_pytree_node_class
class MyEfficientOperator(nk.operator.DiscreteJaxOperator):
    _extra_fields: list[str] = []  # subclasses declare extra __init__ params stored as self._<field>

    def __init__(self, hilbert, sites=None, coeffs=1.0):
        super().__init__(hilbert)
        
        self._sites = np.sort(sites, axis=-1) if sites is not None else None
        if isinstance(coeffs, (int, float, complex, np.generic)):
            self._coeffs = np.full(self._sites.shape[0] if self._sites is not None else 1, coeffs, dtype=float)
        else:
            self._coeffs = np.asarray(coeffs)


        assert self._coeffs.shape[0] == self._sites.shape[0] if self._sites is not None else 1, "Coefficient array length must match number of sites"
        
        
    @property
    def dtype(self):
        if hasattr(self._coeffs, "dtype"):
            return jnp.promote_types(jnp.float64, self._coeffs.dtype)
        # Fallback for placeholder coeff during tree inspection
        return jnp.float64
    
    @property
    def is_hermitian(self):
        return True
    
    def _extra_kwargs(self):
        return {field: getattr(self, f'_{field}') for field in self._extra_fields}

    def tree_flatten(self):
        children = ()
        aux_data = {
            'hilbert': self.hilbert,
            'sites': HashableArray(self._sites) if self._sites is not None else None,
            'coeffs': HashableArray(self._coeffs),
            **self._extra_kwargs(),
        }
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        sites = aux_data['sites']
        coeffs = aux_data['coeffs']
        extra = {field: aux_data[field] for field in cls._extra_fields}
        return cls(
            hilbert=aux_data['hilbert'],
            sites=sites.wrapped if sites is not None else None,
            coeffs=coeffs.wrapped,
            **extra,
        )

    # def __eq__(self, other):
    #     if self.__class__ != other.__class__:
    #         return False
    #     if self.hilbert != other.hilbert:
    #         return False
    #     if self._sites == other._sites:
    #         return jnp.allclose(self._coeffs, other._coeffs)

    def __hash__(self):
        sites_bytes = None if self._sites is None else self._sites.tobytes()
        coeffs_bytes = np.asarray(self._coeffs).tobytes()
        return hash((self.__class__, self.hilbert, sites_bytes, coeffs_bytes))
    
    def __add__(self, other):
        # allow summing operators (e.g., sum([...], 0))
        if isinstance(other, Number) and other == 0:
            return self
        if isinstance(other, type(self)):
            if self._sites is None and other._sites is None:
                return type(self)(self.hilbert, coeffs=self._coeffs + other._coeffs)

            if self._sites is None or other._sites is None:
                return NotImplemented

            new_sites = np.array(self._sites, copy=True)
            new_coeffs = self._coeffs

            for k in range(len(other._sites)):
                site = other._sites[k]

                if new_sites.ndim == 1:
                    matches = np.where(new_sites == site)[0]
                else:
                    matches = np.where(np.all(new_sites == site, axis=1))[0]

                if len(matches) > 0:
                    new_coeffs = new_coeffs.at[matches[0]].add(other._coeffs[k])
                else:
                    site_to_add = np.expand_dims(site, axis=0) if new_sites.ndim > 1 else np.array([site])
                    new_sites = np.concatenate((new_sites, site_to_add), axis=0)
                    new_coeffs = jnp.concatenate((new_coeffs, jnp.asarray([other._coeffs[k]])), axis=0)

            return type(self)(self.hilbert, sites=new_sites, coeffs=new_coeffs)
        return NotImplemented
    
    def __radd__(self, other):
        # handles int + operator (sum starts with 0)
        if isinstance(other, Number) and other == 0:
            return self
        return self.__add__(other)

    def _copy_with_coeff(self, coeffs):
        return type(self)(self.hilbert, sites=self._sites, coeffs=coeffs, **self._extra_kwargs())

    def __mul__(self, other):
        if _is_scalar_like(other):
            return self._copy_with_coeff(self._coeffs * other)
        else:
            return NotImplemented
        
    def __rmul__(self, other):
        return self.__mul__(other)
        
    def __truediv__(self, other):
        if _is_scalar_like(other):
            return self._copy_with_coeff(self._coeffs / other)
        else:
            return NotImplemented

    def __rtruediv__(self, other):
        if _is_scalar_like(other):
            return self._copy_with_coeff(self._coeffs / other)
        else: 
            return NotImplemented
    
    def __neg__(self):
        return self._copy_with_coeff(-self._coeffs)
    
    @partial(jax.jit, static_argnums=(0,))
    def get_conn_padded(self, sigma):
        r'''
        for samples sigma (...,Ns,N),
        returns x_padded (...,Nc,N) and m_padded (...,Nc) 
        '''
        raise NotImplementedError("This method should be implemented in subclasses.")
