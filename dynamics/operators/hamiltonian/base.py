import numpy as np

from typing import Tuple, List
from netket.utils.types import Array
from scipy.sparse import csr_matrix as _csr_matrix

from ...frequencies import Frequency as _Frequency
from netket.operator import AbstractOperator as _AbstractOperator
from netket.operator import LocalOperatorJax as _JaxOperator

from netket.hilbert import Spin as _SpinHilbert


class TimeDependentHamiltonian:
    """
    Instantiates a callable time-dependent Hamiltonian.
    It is defined using a list of operators [h] and frequencies [f] such that 
    ..math::
        H(t) = \sum_i f_i(t) h_i.
    The utilities `to_sparse`, `to_dense` and `to_jax_operator` act on the separated operators.
    """
    def __init__(self, operators:List[_AbstractOperator], frequencies:List[_Frequency]):
        """
        operators : list of inidividual operators.
            All operators must act on the same Hilbert space.
            It is recommended to group (add) all operators that depend on the same frequency schedule for efficiency.
        frequencies : list of inidividual frequencies.
            All frequencies must act over the same annealing time.
        """
        if len(operators) != len(frequencies):
            raise AttributeError(
                f"There should be the same number of different schedules ({len(frequencies)}) as operators ({len(operators)})."
            )
        n = len(operators)
        self._hilbert = operators[0].hilbert
        T = frequencies[0].T
        H = []
        F = []
        for h, f in zip(operators, frequencies):
            if not issubclass(type(h), _AbstractOperator):
                raise ValueError("The operators should all be Netket's operators.")
            if h.hilbert != self.hilbert:
                raise ValueError("All operators must act on the same Hilbert space.")
            H.append(h)

            if not issubclass(type(f), _Frequency):
                raise ValueError(
                    "The schedules should all be instances of `Frequency`."
                )
            if f.T != T:
                raise ValueError("All frequencies must occure for the same time.")
            F.append(f)

        self._H = H
        self._F = F
        self._T = T

        self._str = f"Hamiltonian({self.hilbert}, \n \t F={self.frequencies}, \n \t O={self.operators}\n)"

    @property
    def hilbert(self) -> _SpinHilbert:
        """
        The hilbert space.
        """
        return self._hilbert

    @property
    def operators(self) -> List[_AbstractOperator]:
        """
        The list of operators acting in the Hamiltonian.
        """
        return self._H

    @property
    def frequencies(self) -> List[_Frequency]:
        """
        The list of frequencies acting on each operators of the Hamiltonian.
        """
        return self._F

    @property
    def annealing_time(self) -> float:
        """
        The total annealing time of the Hamiltonian.
        """
        return self._T

    def __repr__(self) -> str:
        """
        Representation of the class.
        """
        return self._str

    def __call__(self, t) -> _AbstractOperator:
        """
        The Hamiltonian at time t
        """
        return sum([f(t) * h for f, h in zip(self.frequencies, self.operators)])

    def plot(self, ax=None):
        """
        Plots each frequency schedule.
        """
        for f in self.frequencies:
            ax = f.plot(ax)

        return ax

    def to_sparse(self) -> List[_csr_matrix]:
        """
        Returns each individual operator, independent of time, as a sparse matrix

        returns : operators SparseMatrices
        """
        return [h.to_sparse() for h in self.operators]

    def to_dense(self) -> List[np.ndarray]:
        """
        Returns each individual operator, independent of time, as a sparse matrix

        returns : operators SparseMatrices
        """
        return [h.to_dense() for h in self.operators]

    def to_jax_operator(self) -> "TimeDependentHamiltonian":
        """
        Creates a time independent Hamiltonian with jax_operators instead of the actual ones.

        returns : TimeDependentHamiltonian with jax operators
        """
        new_operators = [h.to_jax_operator() for h in self.operators]

        return TimeDependentHamiltonian(new_operators, self.frequencies)
