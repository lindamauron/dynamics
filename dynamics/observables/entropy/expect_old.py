import netket as nk

import jax
import jax.numpy as jnp


from netket.vqs import MCState, expect
from netket.stats import statistics as mpi_statistics
from netket import jax as nkjax
from functools import partial

from typing import Union

from .operator import Renyi2EntanglementEntropy


@expect.dispatch
def Renyi2(
    vstate: MCState, op: Renyi2EntanglementEntropy, chunk_size: Union[int, None]
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    samples = vstate.samples
    N = samples.shape[-1]
    n_chains = samples.shape[0]
    n_samples_per_chain = samples.shape[1]
    n_samples = n_chains * n_samples_per_chain

    if n_chains % 2 != 0 and not vstate.sampler.is_exact:
        raise ValueError("Use an even number of chains.")

    # for exact samplers, separate in two chains since they are independent anyway
    if n_chains == 1:
        if n_samples % 2 != 0:
            samples = samples[:, :-1]

        samples = samples.reshape(2, n_samples_per_chain // 2, N)
        n_chains = 2

    # if we don't do bootstrapping, just return the statistics of the samples
    if not op.n_boots:
        σ_η = samples[: (n_chains // 2)]
        σp_ηp = samples[(n_chains // 2) :]

        return Renyi2_sampling_MCState(
            vstate._apply_fun,
            vstate.parameters,
            vstate.model_state,
            σ_η,
            σp_ηp,
            op.partition,
            chunk_size=chunk_size,
        )

    # if we do bootstrapping, basically vmap the previous one over all independent (non-correlated) bootstraps
    else:
        # print('total sample shape : ', samples.shape)  #(n_chains, n_samples_per_chain,N)

        ## for each bootstrap, we randomly choose a different batch of chains
        key, op.rng = jax.random.split(op.rng, 2)
        chains = jax.vmap(jax.random.permutation, (0, None))(
            jax.random.split(key, op.n_boots), jnp.arange(n_chains)
        )
        part1, part2 = (
            chains[:, n_chains // 2 :],
            chains[:, : n_chains // 2],
        )  # (n_boots,n_chains/2)
        # print('partitions : ', part1.shape, part2.shape)

        ## the samples for each bootstrap : now they can be flattened
        @partial(jax.vmap, in_axes=(None, 0), out_axes=0)
        def f(samples, part):
            return samples[part]

        σ_η = f(samples, part1).reshape(
            op.n_boots, -1, N
        )  # (n,n_chains/2*n_samples_per_chain,N)
        σp_ηp = f(samples, part2).reshape(op.n_boots, -1, N)
        Ns = σ_η.shape[1]  # n_chains/2*n_samples_per_chain
        # print('partitioned and flattened smaples :', σ_η.shape, σp_ηp.shape) #(n,Ns,N)

        ## inside each chain 1,2 and each bootstrap, mix the samples to not have the same order between the two parts
        key1, key2 = jax.random.split(key, 2)
        keys1 = jax.random.split(key1, op.n_boots)
        bidcs1 = jax.vmap(jax.random.permutation)(
            keys1, jnp.tile(jnp.arange(Ns), (op.n_boots, 1))
        )

        keys2 = jax.random.split(key2, op.n_boots)
        bidcs2 = jax.vmap(jax.random.permutation)(
            keys2, jnp.tile(jnp.arange(Ns), (op.n_boots, 1))
        )
        # print('boots indices', bidcs1.shape, bidcs2.shape) #(n,Ns)

        ## apply to each parts
        @partial(jax.vmap, in_axes=(0, 0))
        def g(samples, b):
            return samples[b]

        σ_η = g(σ_η, bidcs1)
        σp_ηp = g(σp_ηp, bidcs2)
        # print('booted shape:', σ_η.shape, σp_ηp.shape)

        # print('total sample shape : ', samples.shape)  #(n_chains, n_samples_per_chain,N)
        # σ_η = samples[: (n_chains // 2)].reshape(-1,N) #(n_chains/2*n_samples_per_chain,N)
        # σp_ηp = samples[(n_chains // 2) :].reshape(-1,N)
        # Ns = σ_η.shape[0] #n_chains/2*n_samples_per_chain
        # print('partitioned and flattened smaples :', σ_η.shape)

        # key1, key2 = jax.random.split(key, 2)
        # keys1 = jax.random.split(key1, op.n_boots)
        # bidcs1 = jax.vmap(jax.random.permutation)(keys1, jnp.tile(jnp.arange(Ns), (op.n_boots,1)))

        # keys2 = jax.random.split(key2, op.n_boots)
        # bidcs2 = jax.vmap(jax.random.permutation)(keys2, jnp.tile(jnp.arange(Ns), (op.n_boots,1)))

        # σ_η = σ_η[bidcs1] #(n_boots,Ns,N)
        # σp_ηp = σp_ηp[bidcs2]
        # print('booted shape:', σ_η.shape, σp_ηp.shape)

        entropies = jax.vmap(
            partial(Renyi2_sampling_MCState, chunk_size=chunk_size),
            (None, None, None, 0, 0, None),
        )(
            vstate._apply_fun,
            vstate.parameters,
            vstate.model_state,
            σ_η,
            σp_ηp,
            op.partition,
        ).mean

        # print(entropies.shape, op.n_boots)

        sigma_S = entropies.var()
        return nk.stats.Stats(
            mean=entropies.mean(),
            variance=sigma_S,
            error_of_mean=jnp.sqrt(sigma_S / op.n_boots),
        )  # , mpi_statistics(entropies.reshape(1,-1))


@partial(jax.jit, static_argnames=("afun", "chunk_size"))
def Renyi2_sampling_MCState(
    afun, params, model_state, σ_η, σp_ηp, partition, *, chunk_size
):
    n_chains = σ_η.shape[0]
    N = σ_η.shape[-1]

    σ_η = σ_η.reshape(-1, N)
    σp_ηp = σp_ηp.reshape(-1, N)

    n_samples = σ_η.shape[0]

    mask_A = jnp.zeros((N,), dtype=bool)
    mask_A = mask_A.at[partition].set(True)  # true if s is in sigma batch
    mask_B = jnp.logical_not(mask_A)  # true if eta is in sigma batch

    σ_ηp = σ_η * mask_A + σp_ηp * mask_B
    σp_η = σp_ηp * mask_A + σ_η * mask_B

    @partial(
        nkjax.apply_chunked, in_axes=(None, None, 0, 0, 0, 0), chunk_size=chunk_size
    )
    def kernel_fun(params, model_state, σ_ηp, σp_η, σ_η, σp_ηp):
        W = {"params": params, **model_state}

        return jnp.exp(afun(W, σ_ηp) + afun(W, σp_η) - afun(W, σ_η) - afun(W, σp_ηp))

    # S = -log(E[ exp(logψ + logψ -logψ -logψ) ])
    s = kernel_fun(params, model_state, σ_ηp, σp_η, σ_η, σp_ηp)
    # print('individual entropies shape:', s.shape)
    # S = -jnp.log( mpi_satistics(s).mean )

    Renyi2_stats = mpi_statistics(s.reshape((n_chains, -1)).T)

    # Propagation of errors from S_2 to -log(S_2)
    Renyi2_stats = Renyi2_stats.replace(
        variance=Renyi2_stats.variance / (Renyi2_stats.mean.real) ** 2
    )

    Renyi2_stats = Renyi2_stats.replace(
        error_of_mean=jnp.sqrt(
            Renyi2_stats.variance / (n_samples * nk.utils.mpi.n_nodes)
        )
    )

    Renyi2_stats = Renyi2_stats.replace(mean=-jnp.log(Renyi2_stats.mean).real)

    return Renyi2_stats
