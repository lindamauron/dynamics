import jax
from jax import numpy as jnp

from netket import jax as nkjax
from netket.hilbert import AbstractHilbert
from netket.utils.types import Array, DType, Union

from netket.vqs.base import VariationalState, expect
from netket.operator import DiscreteOperator
from netket.stats import Stats
from netket.vqs.full_summ.expect import _check_hilbert


class ExactDenseState(VariationalState):
    r"""
    Allows to define a dense state in a small Hilbert space, using all functionalities from NetKet.
    The calculations are then exact and faster than a FullSumState with a LogState machine.
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        vector: Union[Array, VariationalState],
        *,
        dtype: DType = None,
        normalize: bool = None,
    ):
        r"""
        Args:
            hilbert: Hilbert space of the state.
            vector: Dense vector representing the state.
            dtype: Data type of the state.
            normalize: Whether to normalize the state.
        """
        super().__init__(hilbert)

        # Only instantiate f the Hilbert space is small enough
        if not self.hilbert.is_indexable:
            raise ValueError("Cannot create state if hilbert space is not indexable.")

        # COnvert to array if needed
        if isinstance(vector, VariationalState):
            vector = vector.to_array(normalize=True)
        self._vector = jnp.asarray(vector).ravel()

        if dtype is not None:
            self._vector = self._vector.astype(dtype)
        if normalize:
            self.normalize()

        if self.hilbert.n_states != self._vector.size:
            raise Exception(
                "Size of vector does not correspond to number of states in the hilbert space."
            )

    def __repr__(self):
        return (
            f"ExactDenseState(hilbert={self.hilbert}, #parameters={len(self.vector)})"
        )

    def to_complex(self):
        if not nkjax.is_complex_dtype(self._vector.dtype):
            dtype = nkjax.dtype_complex(self._vector.dtype)
            self._vector = self._vector.astype(dtype)

    def normalize(self):
        self._vector = _normalize(self._vector)

    @property
    def vector(self):
        return self._vector

    @vector.setter
    def vector(self, other):
        self._normalized = False
        other = jnp.asarray(other).ravel()
        if other.shape != self._vector.shape:
            raise ValueError("Provided vector does not match internal vector in shape.")
        # make sure we never down cast
        _dtype = jnp.promote_types(self._vector.dtype, other.dtype)
        self._vector = other.astype(_dtype).copy()

    @property
    def normalized_vector(self):
        return _normalize(self._vector)

    @property
    def parameters(self):
        return {"vector": self.vector}

    @parameters.setter
    def parameters(self, other):
        self.vector = other["vector"]

    def norm(self):
        return _compute_norm(self.vector)


@jax.jit
def _normalize(v):
    return v / jnp.linalg.norm(v)


@jax.jit
def _compute_norm(v):
    return (v.conj() * v).sum()


@expect.dispatch
def expect(
    vstate: ExactDenseState, Op: Union[DiscreteOperator, Array]
) -> Stats:  # noqa: F811
    r"""
    Computes the expectation value of an operator with respect to the state.
    Args:
        vstate: The state.
        Op: The operator.
    """
    _check_hilbert(vstate, Op)

    if isinstance(Op, DiscreteOperator):
        Op = Op.to_sparse()

    vstate = _normalize(vstate)
    Opsi = Op @ vstate
    expval_O = (vstate.conj().dot(Opsi)).sum()
    variance = jnp.sum(jnp.abs(Opsi - expval_O * vstate) ** 2)
    return Stats(mean=expval_O, error_of_mean=0.0, variance=variance)
