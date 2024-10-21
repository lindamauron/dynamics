from netket.experimental.driver import TDVP
from netket.driver import VMC
from netket.vqs import MCState


def describe(driver: TDVP):
    return (
        "TDVP("
        + f"\n  step_count={driver.step_count}, t={driver.t},"
        + f"\n  vs : {describe(driver.state)},"
        + f"\n  H : {driver._generator_repr},"
        + f"\n  solver = {repr(driver.linear_solver)},"
        + f"\n  integrator = {repr(driver._integrator)},"
        + f"\n  QGT = {repr(driver.qgt)},"
        + f"\n)"
    )


def describe(driver: VMC):
    return (
        "VMC("
        + f"\n  step_count={driver.step_count},"
        + f"\n  vs : {describe(driver.state)},"
        + f"\n  H : {repr(driver._ham)},"
        + f"\n  preconditioner = {repr(driver.preconditioner)},"
        + f"\n  optimizer = {repr(driver.optimizer)},"
        + f"\n)"
    )


def describe(vs: MCState):
    infos = repr(vs)[:-1] + ","
    infos += f"\n  model = {vs.model},"
    infos += f"\n)"

    return infos
