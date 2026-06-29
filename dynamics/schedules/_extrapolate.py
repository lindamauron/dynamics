import numpy as np
from scipy.interpolate import CubicSpline
import jax
import jax.numpy as jnp

from .base import Schedule


def fromarray(xvalues, yvalues):
    return CubicSpline(xvalues, yvalues)


class Extrapolated(Schedule):
    def __init__(self, xvalues, yvalues, T=None):
        if T is None:
            T = np.max(xvalues) - np.min(xvalues)

        schedule = CubicSpline(xvalues, yvalues)

        super().__init__(T, schedule, integral=None, name="Extrapolated")

def jax_schedule(t, x, c):
    t = jnp.asarray(t)

    # Interval index in [0, n_intervals-1]
    idx = jnp.searchsorted(x[1:-1], t, side="right")
    idx = jnp.clip(idx, 0, x.shape[0] - 2)


    c0 = c[0, idx]
    c1 = c[1, idx]
    c2 = c[2, idx]
    c3 = c[3, idx]

    # Horner form
    dt = t - x[idx]
    return ((c0 * dt + c1) * dt + c2) * dt + c3


class JaxExtrapolated(Schedule):
    def __init__(self, T, tvalues, yvalues):
        spline = CubicSpline(tvalues, yvalues)
        x = jnp.asarray(spline.x)
        c = jnp.asarray(spline.c)

        super().__init__(T, jax.jit(lambda t: jax_schedule(t, x, c)), integral=None, name="JaxExtrapolated")
        