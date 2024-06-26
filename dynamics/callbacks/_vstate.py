from netket.jax import tree_ravel
from netket.utils.types import JaxArray
from netket.driver.abstract_variational_driver import AbstractVariationalDriver


def callback_parameters(
    step: JaxArray, log_data: dict, driver: AbstractVariationalDriver
):
    """
    Reports the parameters of the variational state as a flatten PyTree.
    """
    pars = driver.state.parameters
    log_data["pars"], _ = tree_ravel(pars)

    return True
