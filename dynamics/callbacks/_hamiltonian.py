from netket.jax import tree_ravel
from netket.utils.types import JaxArray
from netket.experimental.driver.tdvp_common import TDVPBaseDriver
from netket.experimental.driver.tdvp import TDVP


def callback_frequencies(step: JaxArray, log_data: dict, driver: TDVPBaseDriver):
    """
    Reports the values of each frequency during the evolution
    """
    for k,f in enumerate(driver.generator.frequencies):
        log_data[f"F{k}"] = f(step)
    log_data["tau"] = step / driver.generator.annealing_time

    return True


def callback_R2(step: JaxArray, log_data: dict, te: TDVP):
    """
    Reports the TDVP error during time evolution.
    As it only uses quantities stored in the driver, this callback is almost free to use.
    """
    if te._dw is not None:
        dw, _ = tree_ravel(te._dw)
        F, _ = tree_ravel(te._loss_forces)
        G = te._loss_grad_factor * F

        log_data["r2"] = (
            1
            + (dw.conj().T @ (te._last_qgt @ dw - G) - G.conj().T @ dw)
            / te._loss_stats.variance
        )

    return True
