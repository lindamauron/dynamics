import netket as nk

import jax
import jax.numpy as jnp


from netket.stats.mc_stats import _statistics as mpi_statistics
from netket import jax as nkjax
from functools import partial


@partial(jax.jit, static_argnames=("afun", "chunk"))
def swapped_log_values(
    x1, x2, afun, params, model_state, idcs1, idcs2, mask, not_mask, chunk
):
    r"""
    Computes the log values of the swaped samples of x1, x2 : log ψ(x1,x2), log ψ(x2,x1)
    x1,x2 : samples from two independent chains
    afun: lov_value function
    idcs1,idcs2 : bootstrap indices for each partition
    mask: partition A (true if σ is in A)
    not_mask: partition B (true if σ is in A)

    return: tuple of log_values
    """
    x12 = x1[idcs1] * mask + x2[idcs2] * not_mask
    x21 = x2[idcs2] * mask + x1[idcs1] * not_mask

    @partial(nkjax.apply_chunked, in_axes=(None, None, 0), chunk_size=chunk)
    def kernel_fun(params, model_state, samples):
        W = {"params": params, **model_state}
        return afun(W, samples)

    return kernel_fun(params, model_state, x12), kernel_fun(params, model_state, x21)


@jax.jit
def postprocessing(idcs1, idcs2, lv1, lv2, lv12, lv21):
    r"""
    Computes the mean value of the observable e^(-S_2) = << ψ(x1,x2) ψ(x2,x1) / ψ(x1,x1) ψ(x2,x2) >> for ones bootstrap
    idcs1,idcs2 : bootstrap indices
    lv1, lv2: log_values of the normal chains log ψ(x1,x1), log ψ(x2,x2)
    lv12, lv21: log_values of the swaped chains log ψ(x1,x2), log ψ(x2,x1)

    return: << ψ(x1,x2) ψ(x2,x1) / ψ(x1,x1) ψ(x2,x2) >> for one bootstrap
    """

    s = mpi_statistics(jnp.exp(lv12 + lv21 - lv1[idcs1] - lv2[idcs2]))

    return -jnp.log(s.mean)  # , s.tau_corr, s.tau_corr_max, s.R_hat


@partial(jax.jit, static_argnames=("afun", "chunk_post", "chunk", "n_boots"))
def _renyi2_bootstrap(
    key, afun, params, model_state, σ_η, σp_ηp, partition, n_boots, chunk_post, chunk
):
    r"""
    Evaluates the Renyi2 entropy of a bunch of samples, randomly mixed together for n_bootstrap
    afun : apply function of the state, i.e. log ψ
    samples: samples over which to average
    op: entropy operator containg alll info (partition, n_boots, random key, chunk_post)
    chunk : chunk to use for the evaluation of the (n_boots,n_samples/2) log_values of the swapped samples

    return: << ψ(x1,x2) ψ(x2,x1) / ψ(x1,x1) ψ(x2,x2) >> for all bootstraps (n_boostrap,)
    """
    # samples is of shape (n_chains, n_samples_per_chain, N) i.e Ns = n_chains*n_samples_per_chain
    N = σ_η.shape[-1]
    Ns = σ_η.shape[0]
    # n_chains = samples.shape[0]
    # n_samples_per_chain = samples.shape[1]

    # masks
    mask = jnp.zeros((N,), dtype=bool)
    mask = mask.at[partition].set(True)  # true if s is in A batch
    not_mask = jnp.logical_not(mask)  # true if s is in B batch

    # random keys
    key, _ = jax.random.split(key, 2)

    ## for each bootstrap, we randomly choose a different batch of chains
    # σ_η = samples[:n_chains//2].reshape(-1,N) #(Ns/2,N)
    # σp_ηp = samples[n_chains//2:].reshape(-1,N) #(Ns/2,N)
    @partial(nkjax.apply_chunked, in_axes=(None, None, 0), chunk_size=chunk)
    def kernel_fun(params, model_state, samples):
        W = {"params": params, **model_state}
        return afun(W, samples)

    lv1 = kernel_fun(params, model_state, σ_η)  # (Ns/2,)
    lv2 = kernel_fun(params, model_state, σp_ηp)  # (Ns/2,)

    ## inside each chain 1,2 and each bootstrap, mix the samples to not have the same order between the two parts
    key1, key2 = jax.random.split(key, 2)
    # keys1 = jax.random.split(key1, n_boots*n_chains//2).reshape(n_boots,n_chains//2,2)
    # bidcs1 = jax.vmap(jax.vmap(jax.random.permutation,(0,None)), (0,None))( keys1, jnp.arange(n_samples_per_chain) ).reshape(n_boots,-1)
    # keys2 = jax.random.split(key2, n_boots*n_chains//2).reshape(n_boots,n_chains//2,2)
    # bidcs2 = jax.vmap(jax.vmap(jax.random.permutation,(0,None)), (0,None))( keys2, jnp.arange(n_samples_per_chain) ).reshape(n_boots,-1)

    keys1 = jax.random.split(key1, n_boots)
    bidcs1 = jax.vmap(jax.random.permutation, (0, None))(keys1, jnp.arange(Ns))
    keys2 = jax.random.split(key2, n_boots)
    bidcs2 = jax.vmap(jax.random.permutation, (0, None))(keys2, jnp.arange(Ns))
    # print('boots indices', bidcs1.shape, bidcs2.shape) #(n,Ns)

    lv12, lv21 = nk.jax.vmap_chunked(
        swapped_log_values,
        in_axes=(None, None, None, None, None, 0, 0, None, None, None),
        chunk_size=chunk_post,
    )(σ_η, σp_ηp, afun, params, model_state, bidcs1, bidcs2, mask, not_mask, chunk)
    # print(lv12.shape, lv21.shape) #(n,Ns/2)

    vmap2 = partial(nk.jax.vmap_chunked, chunk_size=chunk_post)

    return vmap2(postprocessing, (0, 0, None, None, 0, 0))(
        bidcs1, bidcs2, lv1, lv2, lv12, lv21
    )  # (n,)


@partial(jax.jit, static_argnames=("afun", "chunk"))
def _renyi2(afun, params, model_state, σ_η, σp_ηp, partition, *, chunk):
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

    @partial(nkjax.apply_chunked, in_axes=(None, None, 0, 0, 0, 0), chunk_size=chunk)
    def kernel_fun(params, model_state, σ_ηp, σp_η, σ_η, σp_ηp):
        W = {"params": params, **model_state}

        return jnp.exp(afun(W, σ_ηp) + afun(W, σp_η) - afun(W, σ_η) - afun(W, σp_ηp))

    # S = -log(E[ exp(logψ + logψ -logψ -logψ) ])
    s = kernel_fun(params, model_state, σ_ηp, σp_η, σ_η, σp_ηp)

    return mpi_statistics(s.reshape((n_chains, -1)).T)
