import numpy as np
from scipy.interpolate import CubicSpline

from .base import Frequency


def fromarray(xvalues, yvalues):
    return CubicSpline(xvalues, yvalues)


class Extrapolated(Frequency):
    def __init__(self, xvalues, yvalues, T=None):
        if T is None:
            T = np.max(xvalues) - np.min(xvalues)

        schedule = CubicSpline(xvalues, yvalues)

        super().__init__(T, schedule, integral=None, name="Extrapolated")
