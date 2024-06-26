import numpy as np
import netket as nk

from typing import Tuple, List
from netket.utils.types import Array
from netket.operator import DiscreteOperator as _DiscreteOperator

from .base import TimeDependentHamiltonian

from ...frequencies import Frequency as _Frequency
from netket.hilbert import Spin as _SpinHilbert


#######################################################################################################################
#################################################### Hamiltonians #####################################################
#######################################################################################################################
def vdW_potential(
    hi: _SpinHilbert,
    lattice,
    Rb: float = 2.4,
    Rcut: float = np.sqrt(7),
    r_occ: _DiscreteOperator = nk.operator.spin.sigmax,
) -> Tuple[_DiscreteOperator, Array]:
    """
    Returns the potential operator of the lattice i.e. V/Ω = sum (Rb/rij)^6 ni nj so that there is no dependence on time
    with a rydberg blockade at Rb and interactions up to Rcut
    hi : Hilbert space of the system
    lattice : lattice on which the operator should act
    Rb : Rydberg blockade radius in units of a (lattice vector)
    Rcut : range of the potential
            if r_ij > Rcut, V_ij = 0
    r_occ : operator |r><r|

    returns : LocalOperator corresponding to the full potential term V/Ω
    """
    N = lattice.N

    # express the real length
    Rb *= lattice.a

    # range of the interactions
    if Rcut is None:
        Rcut = 100 * np.max(
            lattice.distances
        )  # put it to something bigger than the lattice => all interactions are taken
    else:
        Rcut *= lattice.a

    # Construct the interaction matrix by precomputing the distances ratio up to a cut-off
    # we define our matrix R_ij = (Rb/rij)^6 for rij < Rcut but with zeros on the diagonal
    # it will be used for every update, so we do not compute it every time
    R = lattice.distances.copy()
    np.fill_diagonal(R, 1)
    R = (Rb / R) ** 6
    R[lattice.distances > Rcut] = 0
    np.fill_diagonal(R, 0)

    # this does the same thing but uses the R matrix (with vector multiplication) instead of a double sum
    r_occs = np.array([r_occ(hi, i) for i in range(N)])
    V = r_occs.T @ R @ r_occs / 2  # factor of two because each pair is counted twice

    return V, R/2


class RydbergHamiltionian(TimeDependentHamiltonian):
    """
    Constructs the support fot the hamiltonian's operator. The Hamiltonian is of the form :
    H = -Ω(t)/2 \sum_i X_i - Δ(t) \sum_i n_i + Ω_0/2 \sum_ij (Rb/r_ij)^6 n_i n_j
    where :
        - the schedule of Ω(t), Δ(t) is entirely defined by the sweep_time
        - Rb is a parameter usually set to 3rd neighbor
        - the potential can be cut up to a certain distance (Rcut)

    H(t) can be called.
    """

    def __init__(
        self,
        hi: _SpinHilbert,
        lattice,
        frequencies: List[_Frequency],
        Rb: float,
        Rcut: float,
        sigmax: _DiscreteOperator = None,
        r_occ: _DiscreteOperator = None,
    ):
        """
        Constructs the hamiltonian instance, with a rydberg blockade at Rb and interactions up to Rcut
        hi : hilbert space of the system
        lattice : lattice on which the operator should act
        frequencies : list of callables [Ω, Δ, Ω_0]
        Rcut : range of the potential
            if r_ij > Rcut, V_ij = 0
        Rb : Rydberg blockade radius in units of a (lattice vector)
        r_occ : operator |r><r|
            default : (1-σ^z)/2
        sigmax : operator |r><g| + |g><r|
            default : σ^x
        """
        N = lattice.N
        if r_occ is None:
            r_occ = lambda h, i: (1 - nk.operator.spin.sigmaz(h, i)) / 2
        if sigmax is None:
            sigmax = nk.operator.spin.sigmax

        # The total number of Rydberg excitations on the lattice
        N_op = sum([r_occ(hi, i) for i in range(N)])

        # Sum of X operators on the whole lattice (so it is not computed in each loop)
        Xtot_op = sum([sigmax(hi, i) for i in range(N)])

        # Potential
        V_op, self._R = vdW_potential(hi, lattice, Rb, Rcut, r_occ)

        super().__init__([Xtot_op, N_op, V_op], frequencies)

        # Infos of the operator
        self._str = f"Rydberg({self.hilbert}, \n \t {lattice},\n \t F={frequencies},\n \t Rb={Rb}, Rcut={Rcut}\n)"

    @property
    def R(self):
        """
        The interaction matrix of the Hamiltonian. 
        """
        return self._R
