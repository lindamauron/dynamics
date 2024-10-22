from netket.operator import AbstractOperator
from netket.utils.types import Union, Array, Callable
from netket.vqs import VariationalState
from ..operators.hamiltonian import TimeDependentHamiltonian
from ..schedules import Constant as ConstantSchedule

from netket.experimental.dynamics import AbstractSolver
from netket.experimental.driver.tdvp_common import TDVPBaseDriver, odefun

from .state import ExactDenseState
from netket.stats import Stats


class ExactEvolution(TDVPBaseDriver):
    def __init__(
        self,
        operator: Union[AbstractOperator, TimeDependentHamiltonian],
        vector_state: Union[Array, VariationalState],
        ode_solver: AbstractSolver,
        *,
        t0: float = 0.0,
        propagation_type: str = "real",
        error_norm: Union[str, Callable] = "euclidean",
    ):
        r"""
        Args:
            operator : The operator governing the dynamics, either an abstract operator or a time-dependent Hamiltonian.
            vector_state : The initial state of the system, either as an array or a variational state.
            ode_solver : The ODE solver to be used for time evolution.
            t0 : The initial time, default is 0.0.
            propagation_type : The type of propagation, default is "real". Currently, only "real" propagation is implemented.
            error_norm : The error norm to be used, default is "euclidean". Can be a string or a callable.
        """
        if isinstance(vector_state, ExactDenseState):
            state = vector_state
        else:
            state = ExactDenseState(operator.hilbert, vector_state)

        if not propagation_type == "real":
            raise ValueError("Only 'real' propagaation_type is implemented")
        self.propagation_type = propagation_type
        self._dynamics_factor = 1j
        state.to_complex()

        if not issubclass(type(operator), TimeDependentHamiltonian):
            operator = TimeDependentHamiltonian(
                [operator], [ConstantSchedule(1.0, 1.0)]
            )
        self.sparse_generator = operator.to_sparse()
        self.frequencies = operator.frequencies
        self._generator_repr = repr(operator)

        self._time_independent_op = isinstance(operator, AbstractOperator)

        super().__init__(operator, state, ode_solver, t0=t0, error_norm=error_norm)

        self._integrator = None
        self.ode_solver = ode_solver

    def __repr__(self):
        return "{}(t0={}, state={}, generator={}, solver={})".format(
            "ExactEvolution",
            self.t0,
            self.state,
            self._generator_repr,
            self.ode_solver,
        )


def _Hpsi_and_expH(psi, t, driver):
    """
    returns H(t) @ psi = \sum_k f_k(t)* (h_k@psi) and psi.conj().T @ H(t) @ psi = \sum_k f_k(t) * (psi.conj().T@(h_k@psi) )
    for H(t) = \sum_k f_k(t)* h_k
    """
    hpsis = [h @ psi for h in driver.sparse_generator]
    fs = [f(t) for f in driver.frequencies]

    Es = [f * (psi.conj().dot(opsi)) for f, opsi in zip(fs, hpsis)]

    return sum([f * hv for f, hv in zip(fs, hpsis)]), sum(Es)


@odefun.dispatch
def odefun_tdvp(  # noqa: F811
    state: ExactDenseState, driver: ExactEvolution, t, w, *, stage=0
):
    state.parameters = w
    state.reset()

    HPsi, E = _Hpsi_and_expH(state.vector, t, driver)
    norm = state.norm()
    driver._loss_stats = Stats(mean=E/norm, error_of_mean=0.0)
    dPsi_dt = -1j * HPsi

    driver._dw = {"vector": dPsi_dt}

    if stage == 0:
        # save the info at the initial point
        driver._loss_stats = {
            "loss_stats": driver._loss_stats,
        }

    return driver._dw
