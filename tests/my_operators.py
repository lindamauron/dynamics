import jax 
print(f"p{jax.process_index()} | local devices :", jax.local_devices())
print(f"p{jax.process_index()} | global devices:", jax.devices())
print('started')
print('platform:', jax.config.read("jax_platform_name"))
import jax.numpy as jnp
from jax import random as rnd

import numpy as np
import netket as nk
from netket.operator.spin import sigmax, sigmay, sigmaz
import timeit
import tracemalloc
from functools import partial

import dynamics as nkt

def run_with_memory(operator, samples):
    fn = lambda : operator.get_conn_padded(samples)
    fn()

    t = timeit.timeit(lambda: jax.block_until_ready(fn()), number=10)
    print(f'{operator.__class__.__name__} Time taken (10 runs): {t:.2e}s')

    tracemalloc.start()
    out = jax.block_until_ready(fn())
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f'{operator.__class__.__name__} Memory usage | current: {current/1e3:.2f} kB, peak: {peak/1e3:.2f} kB')

    return out


L = 3
lattice = nk.graph.Square(L, pbc=True)
N = lattice.n_nodes
edges = lattice.edges()
vertices = [np.random.choice(N, 4, replace=False) for _ in range(2*N)]  # each vertex is a random set of 4 sites
plaquettes = np.random.choice(N, (N,4), replace=True) # each vertex is a random set of 4 sites

hi = nk.hilbert.Spin(1/2, N)
print(hi)
samples = hi.random_state(rnd.PRNGKey(1), 20) #.reshape(10,-1,N)
# print(plaquettes, vertices)

# Sx_nk = sum(-(i+1)*sigmax(hi, i) for i in range(N))
# Sz_nk = sum((i+1)*sigmaz(hi, i) for i in range(N))
# Sxx_nk = sum(-(i+j) * sigmax(hi, i) @ sigmax(hi, j) for i, j in edges)
# Szz_nk = sum((i-j) * sigmaz(hi, i) @ sigmaz(hi, j) for i, j in edges)
# Av_nk = sum(sum(v) * sigmax(hi,v[0]) @ sigmax(hi,v[1]) @ sigmax(hi,v[2]) @ sigmax(hi,v[3]) for v in vertices)
# Bp_nk = sum(sum(p) * sigmaz(hi,p[0]) @ sigmaz(hi,p[1]) @ sigmaz(hi,p[2]) @ sigmaz(hi,p[3]) for p in plaquettes)
# Id_nk = nk.operator.spin.identity(hi)
SS_nk = sum((-1)**(i+j) * (sigmaz(hi,i)@sigmaz(hi,j) + (-1)**i * sigmay(hi,i)@sigmay(hi,j) + (-1)**i * sigmax(hi,i)@sigmax(hi,j)) for (i,j) in edges)


from dynamics.operators import SzOperator, SxOperator, SzzOperator, SxxOperator, IdOperator, SzzzzOperator, SxxxxOperator, SSOperator
# Sx_me = SxOperator(hi, coeffs=-(1+np.arange(N)))
# Sz_me = SzOperator(hi, coeffs=1+np.arange(N))
# Sxx_me = SxxOperator(hi, edges, coeffs=-np.array([i+j for i,j in edges]))
# Szz_me = SzzOperator(hi, edges, coeffs=np.array([i-j for i,j in edges]))
# Av_me = sum(sum(v) * SxxxxOperator(hi, v) for v in vertices)
# Bp_me = sum(sum(p) * SzzzzOperator(hi, p) for p in plaquettes)
# Id_me = IdOperator(hi)
SS_me = SSOperator(hi, sites=edges, coeffs=(-1.0)**(np.array(edges).sum(axis=1)), sign=[(-1)**i for i,j in edges])

# print(vertices)

for op_me, op_nk in zip(
    [SS_me/2], #, 2*Id_me/N, -Sz_me, 2*Szz_me, 1/2 * Sx_me, Sxx_me/N, 2*Bp_me, -Av_me, ],
    [SS_nk/2], #, 2*Id_nk/N, -Sz_nk, 2*Szz_nk, 1/2 * Sx_nk, Sxx_nk/N, 2*Bp_nk, -Av_nk, ]
    ):
    print('\n', '#'*50)

    xp_me, m_me = run_with_memory(op_me, samples)
    xp_nk, m_nk = run_with_memory(op_nk, samples)
    assert xp_nk.shape == xp_me.shape, f"Shape mismatch: {xp_nk.shape} != {xp_me.shape}"


    # xp has shape (..., max_conn_size, N), m has shape (..., max_conn_size)
    # Flatten all leading sample dims so we iterate over individual samples.
    N = samples.shape[-1]
    Ns = samples.size // N
    flat_samples = np.asarray(samples).reshape(Ns, N)
    flat_xp_nk = np.asarray(xp_nk).reshape(Ns, -1, N)
    flat_xp_me = np.asarray(xp_me).reshape(Ns, -1, N)
    flat_m_nk  = np.asarray(m_nk).reshape(Ns, -1)
    flat_m_me  = np.asarray(m_me).reshape(Ns, -1)

    def build_mel_map(xp, m):
        mel_map = {}
        for j in range(xp.shape[0]):
            key = tuple(xp[j].tolist())
            mel_map[key] = mel_map.get(key, 0) + m[j]
        return mel_map

    # Order-independent check: accumulate matrix elements per connected state and compare.
    all_match = True
    for i in range(Ns):
        map_nk = build_mel_map(flat_xp_nk[i], flat_m_nk[i])
        map_me = build_mel_map(flat_xp_me[i], flat_m_me[i])

        for state in set(map_nk) | set(map_me):
            v_nk = map_nk.get(state, 0)
            v_me = map_me.get(state, 0)
            if not np.isclose(v_nk, v_me):
                print(f"Sample {flat_samples[i]}: state={np.array(state)} m_nk={v_nk}, m_me={v_me}")
                all_match = False

    print('Connected states and matrix elements match (order-independent):',
          '\033[92m' if all_match else '\033[91m', all_match, '\033[0m')
