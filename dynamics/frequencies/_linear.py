import numpy as np

from .base import Frequency


def linear_sweep(fi, ff):
    slope = ff - fi
    offset = fi
    f = np.array([slope, offset])

    return lambda t: (f[0] * t + f[1]), lambda t1, t2: f[0] / 2 * (t2 + t1) + f[1]


class Linear(Frequency):
    """
    Defines the linear frequency schedule.
    It is defined as starting with at t=0 with f(0) = f_initial and reaching the final value f(T) = f_final.
    """

    def __init__(self, T, f_initial, f_final):
        """
        T : total annealing time (s).
        f_initial : value of f(0) (in Hz)
        f_final : value of f(1) (in Hz)
        """

        schedule, integral = linear_sweep(2 * np.pi * f_initial, 2 * np.pi * f_final)
        super().__init__(T, schedule, integral, name = "Linear")
