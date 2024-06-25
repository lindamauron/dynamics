import numpy as np

import jax.numpy as jnp

import netket as nk
from netket import jax as nkjax

from typing import Union
from netket.hilbert import Spin as _SpinHilbert


class Renyi2EntanglementEntropy:
    r"""
    Rényi2 bipartite entanglement entropy of a state :math:`| \Psi \rangle`
    between partitions A and B.
    """

    def __init__(
        self,
        hilbert: _SpinHilbert,
        partition: jnp.array,
        n_boots: Union[int, None] = None,
        seed: Union[int, None] = None,
        chunk_post: Union[int, None] = None,
    ):
        r"""
        Constructs the operator computing the Rényi2 entanglement entropy of
        a state :math:`| \Psi \rangle` for a partition with partition A:

        .. math::

            S_2 = -\log \text{Tr}_A [\rho^2]

        where :math:`\rho = | \Psi \rangle \langle \Psi |` is the density
        matrix of the system and :math:`\text{Tr}_A` indicates the partial
        trace over the partition A.

        The Monte Carlo estimator of S_2 [Hastings et al., PRL 104, 157201 (2010)] is:

        .. math::

            S_2 = - \log \langle \frac{\Psi(\sigma,\eta^{\prime}) \Psi(\sigma^{\prime},\eta)}{\Psi(\sigma,\eta) \Psi(\sigma^{\prime},\eta^{\prime})} \rangle

        where the mean is taken over the distribution
        :math:`\Pi(σ,η) \Pi(σ',η')`, :math:`\sigma \in A`,
        :math:`\eta \in \bar{A}` and
        :math:`\Pi(\sigma, \eta) = |\Psi(\sigma,\eta)|^2 / \langle \Psi | \Psi \rangle`.

        This implementation enables to estimate the variance through bootstrappping.
        In this case, the R_hat returned corresponds to the maximal R_hat over all bootstraps.

        Args:
            hilbert: hilbert space of the system.
            partition: list of the indices identifying the degrees of
                freedom in one partition of the full system. All
                indices should be integers between 0 and hilbert.size
            n_boots: number of bootstraps to do
                if None, considered as 1
            seed: random seed to use if we do bootstrapping
            chunk_post: chunk_size for the post_processing in case of bootstrapping

        Returns:
            Rényi2 operator for which computing the expected value.
        """
        self._hilbert = hilbert

        self._partition = np.array(list(set(partition)))

        if (
            np.where(self._partition < 0)[0].size > 0
            or np.where(self._partition > hilbert.size - 1)[0].size > 0
        ):
            raise ValueError(
                "Invalid partition: possible negative indices or indices outside the system size."
            )
        if n_boots == 1 or n_boots == 0:
            n_boots = None

        self.n_boots = n_boots
        key = nkjax.PRNGKey(seed)
        key = nkjax.mpi_split(key)
        self.rng = key

        self.chunk_post = chunk_post

    @property
    def hilbert(self):
        r"""The hilbert space associated to this observable."""
        return self._hilbert

    @property
    def partition(self):
        r"""
        list of indices for the degrees of freedom in the partition
        """
        return self._partition

    def draw(self):
        state = np.zeros(self.hilbert.size, dtype=int)
        state[self.partition] = 1

        colors = ["k", "r"]
        return state, colors

    def __repr__(self):
        return f"Renyi2EntanglementEntropy(hilbert={self.hilbert}, partition={self.partition})"

    def _reset(self, seed: Union[int, None] = None):
        key = nkjax.PRNGKey(seed)
        key = nkjax.mpi_split(key)
        self.rng = key
        return
