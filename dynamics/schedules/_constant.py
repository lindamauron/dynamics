import numpy as np
import jax.numpy as jnp

from .base import Schedule


class Constant(Schedule):
    """
    Defines the constant frequency schedule.
    """

    def __init__(self, T=1.0, value=1.0):
        """
        T : total annealing time (s).
        value : value of f(t) (in Hz)
        """
        schedule = lambda t: 2 * np.pi * value * jnp.ones_like(t)
        integral = lambda t1, t2: 2 * np.pi * value
        super().__init__(T, schedule, integral, name="Constant")
