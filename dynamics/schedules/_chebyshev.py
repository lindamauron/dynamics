import numpy as np
from scipy.interpolate import CubicSpline
import jax
import jax.numpy as jnp

from .base import Schedule


import jax
import jax.numpy as jnp
from flax import linen as nn
import optax

# ── Model ────────────────────────────────────────────────────────────────────
def chebyshev_basis(t: jnp.ndarray, n: int) -> jnp.ndarray:
    """Returns array of shape (N, n) — one row per time point."""
    t_norm = 2.0 * t - 1.0  # map to [-1, 1]

    Ts = [jnp.ones_like(t_norm), t_norm]
    for k in range(2, n):
        Ts.append(2.0 * t_norm * Ts[-1] - Ts[-2])

    return jnp.stack(Ts, axis=-1)  # (N, n)

@jax.jit
def log_cheby_apply(coeffs: jnp.ndarray, t: jnp.ndarray) -> jnp.ndarray:
    basis = chebyshev_basis(t, coeffs.shape[0])  # (N, n)
    log_f = basis @ coeffs                        # (N,)
    return jnp.exp(log_f)

def fit_log_cheby(
    t_data: jnp.ndarray,
    f_data: jnp.ndarray,
    n_coeffs: int = 100,
    show_error=False
):
    basis = chebyshev_basis(t_data, n_coeffs)            # (N, n)
    log_f_data = jnp.log(f_data)  # (N,)

    coeffs, _, rank, sv = jnp.linalg.lstsq(basis, log_f_data, rcond=None)
    if show_error:
        print(f"lstsq: rank={rank}, max_sv={sv[0]:.3f}, min_sv={sv[-1]:.6f}")
        print(f"log-space MSE: {jnp.mean((basis @ coeffs - log_f_data)**2):.6e}")

    return coeffs

class Chebyshev(Schedule):
    def __init__(self, T, xvalues, yvalues, show_error=False):
        xvalues = np.sort(xvalues)
        xvalues = (xvalues-xvalues.min()) / (xvalues.max() - xvalues.min())
        self.parameters = fit_log_cheby(xvalues, yvalues, n_coeffs=100, show_error=show_error)

        schedule = jax.jit(lambda t: log_cheby_apply(self.parameters, t))

        super().__init__(T, schedule, integral=None, name="Chebyshev")
