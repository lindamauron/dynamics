import numpy as np
from netket.sampler import ParallelTemperingSampler
from netket.utils.types import JaxArray
from netket.driver.abstract_variational_driver import AbstractVariationalDriver


class CallbackSampler:
    """
    Reports all values of interest for the sampler. In particular, this same class can be used for any sampler.
    Exact : nothing is reported
    Metropolis : - acceptance
    ParrallelTempering : - acceptance
        - beta_position
        - beta_diffusion
    """

    def __init__(self, sampler):
        if sampler.is_exact:
            self._call = lambda x, y, z: True
        elif type(sampler) == ParallelTemperingSampler:
            self._call = callback_tempering
        else:
            self._call = callback_acc

    def __call__(
        self, step: JaxArray, log_data: dict, driver: AbstractVariationalDriver
    ):
        return self._call(step, log_data, driver)


def callback_acc(step: JaxArray, log_data: dict, driver: AbstractVariationalDriver):
    """
    Acceptance of the sampler during the evolution
    """
    log_data["acc"] = driver.state.sampler_state.acceptance

    return True


def callback_tempering(
    step: JaxArray, log_data: dict, driver: AbstractVariationalDriver
):
    """
    Acceptance and temperatur statistics of the sampler during the evolution
    """
    log_data["acc"] = driver.state.sampler_state.acceptance
    log_data["beta_pos"] = driver.state.sampler_state.normalized_position()
    log_data["beta_diff"] = driver.state.sampler_state.normalized_diffusion()

    return True
