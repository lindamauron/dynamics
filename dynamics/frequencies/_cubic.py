import numpy as np

from scipy.interpolate import interp1d

from .base import Frequency


def cubic_sweep(fi, ff, tInf, fInf, slope=0.0):
    midT = 1e-2

    midF = slope * midT
    tPoints = np.array([0.0, tInf - midT, tInf, tInf + midT, 1.0])
    fPoints = np.array([fi, fInf - midF, fInf, fInf + midF, ff])
    fInterp = interp1d(
        tPoints, fPoints, kind="cubic", bounds_error=False, fill_value=(fi, ff)
    )

    times = np.linspace(0.0, 1.0, 10000)
    f = np.polyfit(times, fInterp(times), 3)

    return (
        lambda t: f[0] * t**3 + f[1] * t**2 + f[2] * t + f[3],
        lambda t1, t2: f[0] / 4 * (t2 + t1) * (t2**2 + t1**2)
        + f[1] / 3 * (t2**2 + t1 * t2 + t1**2)
        + f[2] / 2 * (t2 + t1)
        + f[3],
    )


class Cubic(Frequency):
    """
    Defines the cubic frequency schedule.
    It is defined as starting t=0 with f(0) = f_initial and reaching the final value f(T) = f_final.
    Furthermore, it crosses a saddle point in between with chosen position and value.
    """

    def __init__(self, T, f_initial, f_final, saddle_point=None, saddle_value=None):
        """
        T : total annealing time (s).
        f_initial : value of f(0) (in Hz)
        f_final : value of f(1) (in Hz)
        saddle_point : point in time raching a saddle point, i.e. t where f'(t) = 0
            default : T/2
        saddle_value : value at the saddle point, i.e. f(saddle_point)
            default : (f_initial+f_final)/2
        """
        if saddle_point is None:
            saddle_point = 1 / 2
        if saddle_value is None:
            saddle_value = (f_initial + f_final) / 2

        schedule, integral = cubic_sweep(
            2 * np.pi * f_initial,
            2 * np.pi * f_final,
            saddle_point,
            2 * np.pi * saddle_value,
        )
        super().__init__(T, schedule, integral, name = "Cubic")
