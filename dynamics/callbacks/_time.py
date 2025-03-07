import jax.numpy as jnp
from typing import Callable
from netket.utils.types import JaxArray
from netket.experimental.driver.tdvp_common import TDVPBaseDriver

def monitor_dt(step: JaxArray, log_data: dict, driver: TDVPBaseDriver):
    """
    Monitors the value of `dt` during time evolution.
    """
    log_data["dt"] = driver.integrator.dt
    return True

def well_dt(T, init_value, extremal_value, extrema=0.5, var=1 / 40):
    """
    Defines a schedule forming a well. The value outside the well is init_value and deep in the well is extremal_value.
    T : total time of evolution
    init_value : value of dt outside the well (most of the time)
    extremal_value : value of dt at the peak of the well
    extrema : position (in time) of the well
    var : variance (width) of the well

    returns : callable schedule of dt
    """

    def schedule(step: JaxArray, log_data: dict, driver: TDVPBaseDriver):
        new_dt = init_value + (extremal_value - init_value) * jnp.exp(
            -((step / T - extrema) ** 2) / var
        )

        return new_dt

    return schedule


def linear_dt(T, init_value, final_value):
    """
    Defines a schedule forming a linear rampe from init_value to final_value.
    T : total time of evolution
    init_value : initial value
    final_value : final value

    returns : callable schedule of dt
    """

    def schedule(step: JaxArray, log_data: dict, driver: TDVPBaseDriver):
        new_dt = init_value + (final_value - init_value) / T * step

        return new_dt

    return schedule


def unchanged_dt():
    """
    Defines a schedule of constant value fixed by the driver.

    returns : callable schedule of dt
    """
    def schedule(step: JaxArray, log_data: dict, driver: TDVPBaseDriver):
        return driver.integrator.dt

    return schedule

class DynamicalTimeStep:
    """
    Modifies the value of dt according to a callable schedule passed as an initializing argument.
    In general, the schedule can use whatever quantity is present in log_data or driver to calculate the next time step.
    When called, it reports the previous value of dt and then modifies it for the next step accodring to the schedule.
    """

    def __init__(self, schedule: Callable = unchanged_dt()):
        """
        schedule : callable returning the time step whenenver called inside a driver.
            It must take the arguments step (float), log_data (dict) and driver (TDVPCommon).
            default : constant time step of 10^(-2)
        """
        if not isinstance(schedule, Callable):
            raise AttributeError(f"The dt schedule must be a callable.")

        self._schedule = schedule

    def __call__(self, step: JaxArray, log_data: dict, driver: TDVPBaseDriver):
        monitor_dt(step, log_data, driver)

        new_dt = self._schedule(step, log_data, driver)

        driver.integrator._state = driver.integrator._state.replace(dt=new_dt)

        return True
