import numpy as np

from .base import Frequency


class Constant(Frequency):
    """
    Defines the constant frequency schedule.
    """

    def __init__(self, T=1.0, value=1.0):
        """
        T : total annealing time (s).
        value : value of f(t) (in Hz)
        """
        schedule = lambda t: 2 * np.pi * value * np.ones_like(t)
        integral = lambda t1, t2: 2 * np.pi * value
        super().__init__(T, schedule, integral, name="Constant")
