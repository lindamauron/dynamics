import numpy as np
import jax.numpy as jnp

from typing import List
from netket.utils.types import Array

from .base import TimeDependentHamiltonian

from ...frequencies import Frequency as _Frequency
from netket.hilbert import Spin as _SpinHilbert


from netket.operator.spin import sigmax, sigmaz


class SpinGlassHamiltonian(TimeDependentHamiltonian):
    """
    Definition of the timme-dependent Hamiltonian H(t) = -Γ * sum_i σ_i^x + J \sum_{i<j} σ_i^z σ_j^z J_{ij}, J_{ij} ~ N(0,1)
    """

    def __init__(
        self, hi: _SpinHilbert, lattice, frequencies: List[_Frequency], seed: int = 0
    ):
        """
        hi : Hilbert space
        lattice : lattice on which the spins live.
            It must possess the attributes edges and generate_couplings(seed, precision)
        frequencies : list of callables [Γ, J]
        seed : random seed used for the random couplings
        """
        self._hilbert = hi
        N = hi.size

        Hx = -sum([sigmax(hi, i) for i in range(N)])
        szs = [sigmaz(hi, i) for i in range(N)]

        # interactions
        edges = lattice.edges
        self._edges = edges
        Js = lattice.generate_couplings(seed, 256)
        idcs = np.array(edges, dtype=int)
        idcs = (idcs[:, 0], idcs[:, 1])  # so we can do things easily
        self._J = jnp.zeros((N, N)).at[idcs].set(Js)
        Hp = 0
        for J, (i, j) in zip(Js, edges):
            Hp += szs[i] * szs[j] * J
        Hzz = Hp

        super().__init__([Hx, Hzz], frequencies)

        # representation
        self._str = f"SpinGlass({self.hilbert}, \n \t #edges={len(edges)}, \n \t F={self.frequencies}, \n \t seed={seed}\n)"

    @property
    def edges(self) -> Array:
        """
        Edges of the lattice.
        """
        return self._edges

    @property
    def J(self) -> Array:
        """
        Interaction matrix of the Hamiltonian.
        """
        return self._J
