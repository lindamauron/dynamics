import abc
import numpy as np
from typing import Callable, Optional
import jax

twopi = 2 * np.pi


class Schedule:
    """
    General class for a frequency schedule.
    It is also possible to integrate and derivate the frequencies over any time interval.

    This class comes with multiple algebraical tools, allowing to multiply the whole frequency schedule, shift the time range and so on.
    Multiple frequencies can even be appended after each other, keeping all previous capacities.
    """

    def __init__(
        self, T, schedule: Callable, integral: Optional[Callable] = None, name=None
    ):
        """
        T : total annealing time (s).
            It has to be a postivie number.
        schedule : callable giving the frequency at each time in the annealing process
            The function must be a callable defined on the range [0,1].
        integral : callable giving the value of the mean integral $\int_t1^t2 f(τ) dτ$.
            If nothing is provided, the mean is computed using the trapezoidal rule (if called).
        """
        if T <= 0 or not np.issubdtype(type(T), np.number):
            raise ValueError(f"T must be a positive number, instead got {T}.")
        self._T = float(T)

        self._schedule = check_schedule(schedule, "schedule")
        if integral is not None:
            integral = check_integral(integral, "integral")
        self._integral = integral

        self.name = "Schedule" if name is None else name

    def __call__(self, t):
        """
        Computes the frequency at time t
        t : time of the evolution (in s)

        return : corresponding frequency (in rad/s)
        """
        return self._schedule(t / self.T)

    @property
    def T(self) -> float:
        """
        The total time of annealing
        """
        return self._T

    def herz(self, t):
        """
        Computes the frequency at time t in Hz
        t : time of the evolution (in s)

        return : corresponding f/2π (Hz)
        """
        return self(t) / twopi

    @abc.abstractmethod
    def mean(self, t1, t2):
        r"""
        Does the operation $\int_t1^t2 f(τ) dτ$
        t1 : smallest time (s)
        t2 : highest time (s)

        returns : mean of f (rad)
        """
        if self._integral is None:
            n = np.max([int((t2 - t1) / 1e-3) + 1, 11])
            xs = np.linspace(t1, t2, endpoint=True, num=n)
            ys = self(xs)
            return np.trapz(ys, xs) / (t2 - t1)

        else:
            return self._integral(t1 / self.T, t2 / self.T)

    def __repr__(self) -> str:
        return (
            self.name
            + f"(T={self.T}, in 2π[{self(0.0)/twopi:.2f},{self(self.T)/twopi:.2f}])"
        )

    def plot(self, ax=None):
        """
        Draws the frequency schedule for the whole annealing time.
        ax : the ax on which to draw
            if no argument is provided, a new ax is created and returned
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(3, 2))
        times = np.linspace(0, self.T, int(self.T * 100) + 1)
        ax.plot(times, self(times) / twopi)
        ax.set_xlabel(r"$t$ (s)")
        ax.set_ylabel(r"$F(t)/2\pi$ (Hz)")
        return ax

    ################ Algebra ################
    def __neg__(self) -> "Schedule":
        return (-1.0) * self
    
    def __add__(self, other) -> "Schedule":
        if np.issubdtype(type(other), np.number):
            new_schedule = lambda t: self._schedule(t) + other * 2 * np.pi
            return Schedule(self.T, new_schedule, self._integral, self.name)
        elif issubclass(type(other), Schedule):
            if other.T != self.T:
                raise ValueError("All frequencies must occure for the same time.")
            new_schedule = lambda t: self._schedule(t) + other._schedule(t)
            if self._integral is not None and other._integral is not None:
                new_integral = lambda t1, t2: self._integral(t1, t2) + other._integral(t1, t2)
            else:
                new_integral = None
            return Schedule(self.T, new_schedule, new_integral, self.name + "+" + other.name)
        else:
            raise NotImplementedError
        
    def __radd__(self, other) -> "Schedule":
        return self.__add__(other)
        
    def __sub__(self, other) -> "Schedule":
        return self.__add__(-other)

    def __rsub__(self, other) -> "Schedule":
        return -self + other

    def __mul__(self, other) -> "Schedule":
        if np.issubdtype(type(other), np.number):
            new_schedule = lambda t: self._schedule(t) * other

            if self._integral is not None:
                new_integral = lambda t1, t2: self._integral(t1, t2) * other
            else:
                new_integral = None

            return Schedule(self.T, new_schedule, new_integral, self.name)
        
        elif issubclass(type(other), Schedule):
            if other.T != self.T:
                raise ValueError("All frequencies must occure for the same time.")
            new_schedule = lambda t: self._schedule(t) * other._schedule(t) / 2/np.pi
            if self._integral is not None and other._integral is not None:
                new_integral = lambda t1, t2: self._integral(t1, t2) * other._integral(t1, t2) / 2/np.pi
            else:
                new_integral = None
            return Schedule(self.T, new_schedule, new_integral, self.name + "*" + other.name)
        else:
            raise NotImplementedError

    def __rmul__(self, other) -> "Schedule":
        return self.__mul__(other)
    
    def __truediv__(self, other) -> "Schedule":
        if np.issubdtype(type(other), np.number):
            return self.__mul__(1.0 / other)
        elif issubclass(type(other), Schedule):
            return self.__mul__(other.__opposite__())
        else:
            raise NotImplementedError
    
    def __rtrudiv__(self, other) -> "Schedule":
        return other * self.__opposite__()
    
    def __opposite__(self) -> "Schedule":
        return Schedule(self.T, lambda t: (2*np.pi)**2 / self._schedule(t), self._integral, "1/" + self.name)

    ################ Appending ################
    def time_shift(self, dt) -> "Schedule":
        """
        Shifts the time origin by dt. This maco is usefull in order to append multiple frequencies after the other.
        """
        new_schedule = lambda t: self._schedule(t - dt / self.T)
        if self._integral is not None:
            new_integral = lambda t1, t2: self._integral(t1 + dt / self.T, t2 / self.T)
        else:
            new_integral = None
        return Schedule(self.T, new_schedule, new_integral, self.name)

    def append(self, other) -> "Schedule":
        """
        This functionality allows to append a frequency schedule to follow the current one. In particular, for `F = F1.append(F2)` one has
            F(t) = F1(t) if t < F1.T else F2(t-F1.T)
        but everything can be used smoothly from the new object's interface.
        """
        if not issubclass(type(other), Schedule):
            raise NotImplementedError

        new_name = self.name + other.name
        new_T = self.T + other.T

        def new_schedule(t):
            flag = t <= self.T / new_T
            return self._schedule(t * new_T / self.T) * flag + other._schedule(
                (t * new_T - self.T) / other.T
            ) * (1 - flag)

        if self._integral is None or other._integral is None:
            new_integral = None
        else:

            def new_integral(t1, t2):
                t1, t2 = t1 * new_T, t2 * new_T

                int1 = self._integral
                int2 = other._integral
                T = self.T
                flag1 = t1 < T
                flag2 = t2 > T
                y1 = t2 * (1 - flag2) + T * flag2
                x2 = t1 * (1 - flag1) + T * flag1

                return (
                    flag1 * int1(t1, y1) * (y1 - t1) + flag2 * int2(x2, t2) * (t2 - x2)
                ) / (t2 - t1)

        return Schedule(new_T, new_schedule, new_integral, new_name)


def check_schedule(fct, name="function"):
    if not isinstance(fct, Callable):
        raise AttributeError(f"The {name} must be a callable.")

    if not (np.isfinite(fct(0.0)) and np.isfinite(fct(1.0))):
        raise ValueError(f"The {name} must be finite in the range [0,T].")

    # verify vectorization
    try:
        _ = fct(np.array([0.0, 1.0]))
    except TypeError:
        fct = jax.vmap(fct)

    return fct


def check_integral(fct, name="function"):
    if not isinstance(fct, Callable):
        raise AttributeError(f"The {name} must be a callable.")

    if not np.isfinite(fct(0.0, 1.0)):
        raise ValueError(f"The {name} must be finite in the range [0,T].")

    # verify vectorization
    try:
        _ = fct(np.array([0.0, 0.5]), np.array([0.5, 1.0]))
    except TypeError:
        fct = jax.vmap(fct)

    return fct
