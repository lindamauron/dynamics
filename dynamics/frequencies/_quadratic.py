import numpy as np

from .base import Frequency


def quadratic_sweep(
    fi,
    ff,
):
    a = ff - fi
    c = fi

    return (
        lambda t: a * t**2 + c,
        lambda t1, t2: 1 / 3 * a * (t2**2 + t1 * t2 + t1**2) + c,
    )


class Quadratic(Frequency):
    """
    Defines the linear frequency schedule.
    It is defined as starting with an extrema at t=0 with f(0) = f_initial and reaching the final value f(T) = f_final.
    """

    def __init__(self, T, f_initial, f_final):
        """
        T : total annealing time (s).
        f_initial : value of f(0) (in Hz)
        f_final : value of f(1) (in Hz)
        """

        schedule, integral = quadratic_sweep(2 * np.pi * f_initial, 2 * np.pi * f_final)
        super().__init__(T, schedule, integral, name="Quadratic")
