import netket as nk
from netket.operator import DiscreteJaxOperator, SumOperator, ProductOperator

import jax.numpy as jnp

from netket.utils.numbers import dtype as _dtype, is_scalar
from numpy import identity

class PauliOperator(DiscreteJaxOperator):
    def __init__(self, hilbert, site, coeff=1.0):
        super().__init__(hilbert)
        self._site = site
        self._coeff = coeff

    @property
    def dtype(self):
        return jnp.complex128
    

    def __matmul__(self, other):
        if is_scalar(other) or isinstance(other, jnp.ndarray):
            return self.__class__(self.hilbert, self._site, self._coeff * other)
        if isinstance(other, ProductOperator):
            return other.__rmatmul__(self)
        elif isinstance(other, SumOperator) or isinstance(other, PauliOperator):
            return ProductOperator(*[self, other])
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        # right-multiplication: other @ self
        if is_scalar(other) or isinstance(other, jnp.ndarray):
            return self.__class__(self.hilbert, self._site, self._coeff * other)

        if isinstance(other, ProductOperator):
            # Prefer delegating to ProductOperator's internal handler if available,
            # so the ProductOperator can preserve its internal ordering/optimizations.
            return other.__matmul__(self)

        elif isinstance(other, SumOperator) or isinstance(other, PauliOperator):
            # Ordering matters: other (left) should come before self (right).
            return ProductOperator(*[other, self])

        else:
            return NotImplemented


    def __radd__(self, other):
        if is_scalar(other):
            return self.__add__(other)
        return super().__radd__(other)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __isub__(self, other):
        return self.__iadd__(-other)

    def __neg__(self):
        return -1 * self

    def __add__(self, other):

        if is_scalar(other):
            if other == 0:
                return self
            else:
                return SumOperator(*[self, identity(self.hilbert, coeff=other)])
        elif isinstance(other, PauliOperator):
            return SumOperator(*[self, other])
        elif isinstance(other, SumOperator):
            return other.__radd__(self)
        return NotImplemented

    # def __iadd__(self, other):
    #     return NotImplemented

    def __truediv__(self, other):
        if not is_scalar(other):
            raise TypeError("Only division by a scalar number is supported.")

        if other == 0:
            raise ValueError("Dividing by 0")
        return self.__mul__(1.0 / other)

    # def __rmul__(self, other):
    #     return self.__mul__(other)

    # def __mul__(self, other):
    #     if isinstance(other, DiscreteOperator):
    #         # TODO: Deprecated in September 2025
    #         warnings.warn(OperatorMultiplicationDeprecationWarning())
    #         op = self.copy(dtype=jnp.promote_types(self.dtype, _dtype(other)))
    #         return op.__imatmul__(other)
    #     elif is_scalar(other):
    #         op = self.copy(dtype=jnp.promote_types(self.dtype, _dtype(other)))
    #         return op.__imul__(other)
    #     return NotImplemented

    # def __imul__(self, other):

    #     return NotImplemented

    # def __imatmul__(self, other):
    #     return self._op_imatmul_(other)

    # def _op__matmul__(self, other: "LocalOperatorBase") -> "LocalOperatorBase":
    #     if not isinstance(other, LocalOperatorBase):
    #         return NotImplemented
    #     op = self.copy(dtype=jnp.promote_types(self.dtype, _dtype(other)))
    #     return op._op_imatmul_(other)

    # def _op_imatmul_(self, other: "LocalOperatorBase") -> "LocalOperatorBase":
    #     return NotImplemented



class identity(PauliOperator):
    def __init__(self, hilbert, coeff=1.0):
        super().__init__(hilbert, site=None, coeff=coeff)
    
    @property
    def dtype(self):
        return jnp.float64
    
    def get_conn_padded(self, sigma):
        x_padded = jnp.expand_dims(sigma, axis=1)
        m_padded = jnp.expand_dims(jnp.ones(*sigma.shape[:-1]), axis=1)
   
        return x_padded, self._coeff*m_padded


class sigmaz(PauliOperator):
    @property
    def dtype(self):
        return jnp.float64
    
    def get_conn_padded(self, sigma):
        x_padded = jnp.expand_dims(sigma, axis=1)
        m_padded = jnp.expand_dims(sigma[...,self._site], axis=1)
   
        return x_padded, self._coeff*m_padded

class sigmax(PauliOperator):
    @property
    def dtype(self):
        return jnp.float64
    
    def get_conn_padded(self, sigma):
        x_padded = jnp.expand_dims(sigma.at[...,self._site].multiply(-1), axis=1)
        m_padded = jnp.expand_dims(jnp.ones(*sigma.shape[:-1]), axis=1)
   
        return x_padded, self._coeff*m_padded

class sigmay(PauliOperator):
    @property
    def dtype(self):
        return jnp.complex128
    
    def get_conn_padded(self, sigma):
        x_padded = jnp.expand_dims(sigma.at[...,self._site].multiply(-1), axis=1)
        m_padded = jnp.expand_dims(1j * sigma[...,self._site], axis=1)
   
        return x_padded, self._coeff*m_padded