import numpy as np
import jax
from jax import numpy as jnp
from typing import Optional

from netket.graph import AbstractGraph
from netket.sampler.rules.base import MetropolisRule


class ClusterFlipRule(MetropolisRule):
    r"""
    A rule flipping all spins in a randomly chosen cluster.

    This rule acts on multiple local degrees of freedom :math:`s_i,\dots s_{i+s}`, where :math:`s` is the dimension of a cluster,
    and proposes a new state: :math:`s_1 \dots s^\prime_i \dots s^\prime_{i+s} \dots s_N`,
    where :math:`s^\prime_k = -s_k`.

    The transition probability associated to this sampler can
    be decomposed into two steps:

    1. A cluster :math:`\mathcal{C}` is chosen with uniform probability.
    2. The sites are flipped, i.e. :math:`s^\prime_k = -s_k` for all :math:`k\in\mathcal{C}`.

    """

    clusters: jax.Array
    r"""2-Dimensional tensor :math:`T_{i,j}` of shape
    :math:`N_\text{clusters}\times s` where the first dimension
    runs over the list of s-site clusters and the second dimension
    runs over the s sites of those clusters.
    """

    def __init__(
        self,
        *,
        clusters: Optional[list[tuple[int, int]]] = None,
        graph: Optional[AbstractGraph] = None,
        d_max: int = 1,
        d_min: int = 0,
    ):
        r"""
        Constructs the ClusterFlip Rule.

        You can pass either a list of clusters or a netket graph object to
        determine the clusters to exchange.

        Args:
            clusters: The list of clusters that can be exchanged. This should be
                a list of 2-tuples containing two integers. Every tuple is an edge,
                or cluster of sites to be exchanged.
            graph: A graph, from which the edges determine the clusters
                that can be exchanged.
        """
        if clusters is None and graph is not None:
            assert d_max >= d_min, "d_max must be greater or equal to d_min."
            clusters = compute_clusters(graph, d_max=d_max, d_min=d_min)
        elif not (clusters is not None and graph is None):
            raise ValueError(
                """You must either provide the list of flip-clusters or a netket graph, from
                              which clusters will be computed. """
            )
        self.clusters = jnp.array(clusters)

    def transition(rule, sampler, machine, parameters, state, key, σ):
        n_chains = σ.shape[0]

        # pick a random cluster
        cluster_id = jax.random.randint(
            key, shape=(n_chains,), minval=0, maxval=rule.clusters.shape[0]
        )

        # flip the cluster
        def scalar_update_fun(σ, cluster):
            sites = rule.clusters[cluster]
            return σ.at[sites].multiply(-1.0)

        return (
            jax.vmap(scalar_update_fun, in_axes=(0, 0), out_axes=0)(σ, cluster_id),
            None,
        )

    def __repr__(self):
        return f"ClusterFlipRule(# of clusters: {len(self.clusters)})"


def compute_clusters(graph: AbstractGraph, d_max: int, d_min: int):
    """
    Given a netket graph and a maximum distance, computes all clusters.
    If `d_max = 1` this is equivalent to taking the edges of the graph.
    Then adds next-nearest neighbors and so on.
    """
    clusters = []
    distances = np.asarray(graph.distances())
    size = distances.shape[0]
    for i in range(size):
        for j in range(i + 1, size):
            if distances[i][j] <= d_max and distances[i][j] >= d_min:
                clusters.append((i, j))

    res_clusters = np.empty((len(clusters), 2), dtype=np.int64)

    for i, cluster in enumerate(clusters):
        res_clusters[i] = np.asarray(cluster)

    return res_clusters
