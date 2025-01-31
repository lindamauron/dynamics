# Copyright 2020, 2021  The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable, Sequence

import numpy as np
import warnings

import netket as nk
from netket.jax import tree_cast
from netket.utils.dispatch import dispatch
from netket.utils.types import Any, Union
from netket.optimizer import LinearOperator
from netket.vqs import (
    MCState,
    FullSumState,
)

PureState = Union[MCState, FullSumState]
from netket.experimental.dynamics import AbstractSolver

from netket.experimental.driver.tdvp_common import odefun
from netket.experimental.driver.tdvp import _map_parameters, TDVP

from dynamics.operators import TimeDependentHamiltonian
from dynamics.schedules import Constant as ConstantSchedule


class SemiExactTDVP(TDVP):
    r"""
    Variational time evolution driver which applies the diagonal part of the Hamiltonian exactly.
    For an Hamiltonian `H(t) = A(t) Hx + B(t) Hz`, the driver applies at each time step:
        1. `Uz = exp(-i B(t) Hz dt/2)`
        2. TDVP with `A(t) Hx`
        3. `Uz = exp(-i B(t+dt/2) Hz dt/2)`

    The resulting evolution is a Trotterized version of the original dynamics.

    This driver then only works for models which possess an external layer resembling the diagonal Hamiltonian, i.e.
    `log_psi(z) ~ \sum_ij W_ij z_i z_j + \sum_i V_i z_i` for `Hz = \sum_ij J_ij z_i z_j + \sum_i h_i z_i`.
    Thus, each model, hamiltonian pair has to define its own `Uz` and `Hx` functions to be applied for evolution.
        defaults to `Uz=1` and `Hx=H(t)`, i.e. standard (slowed down) TDVP.
    """

    def __init__(
        self,
        operator: TimeDependentHamiltonian,
        variational_state: PureState,
        ode_solver: AbstractSolver = None,
        *,
        t0: float = 0.0,
        propagation_type: str = "real",
        qgt: LinearOperator = None,
        linear_solver=nk.optimizer.solver.pinv_smooth,
        linear_solver_restart: bool = False,
        error_norm: str | Callable = "euclidean",
    ):
        r"""
        Initializes the time evolution driver.

        Args:
            operator: The generator of the dynamics.
            variational_state: The variational state.
            ode_solver: Solving algorithm used the ODE.
            t0: Initial time at the start of the time evolution.
            propagation_type: Determines the equation of motion: "real"  for the
                real-time Schrödinger equation (SE), "imag" for the imaginary-time SE.
            qgt: The QGT specification.
            linear_solver: The solver for solving the linear system determining the time evolution.
                This must be a jax-jittable function :code:`f(A,b) -> x` that accepts a Matrix-like, Linear Operator
                PyTree object :math:`A` and a vector-like PyTree :math:`b` and returns the PyTree :math:`x` solving
                the system :math:`Ax=b`.
                Defaults to :func:`nk.optimizer.solver.pinv_smooth` with the default svd threshold of 1e-10.
                To change the svd threshold you can use :func:`functools.partial` as follows:
                :code:`functools.partial(nk.optimizer.solver.pinv_smooth, rcond_smooth=1e-5)`.
            linear_solver_restart: If False (default), the last solution of the linear system
                is used as initial value in subsequent steps.
            error_norm: Norm function used to calculate the error with adaptive integrators.
                Can be either "euclidean" for the standard L2 vector norm :math:`w^\dagger w`,
                "maximum" for the maximum norm :math:`\max_i |w_i|`
                or "qgt", in which case the scalar product induced by the QGT :math:`S` is used
                to compute the norm :math:`\Vert w \Vert^2_S = w^\dagger S w` as suggested
                in PRL 125, 100503 (2020).
                Additionally, it possible to pass a custom function with signature
                :code:`norm(x: PyTree) -> float`
                which maps a PyTree of parameters :code:`x` to the corresponding norm.
                Note that norm is used in jax.jit-compiled code.
        """
        # Convert the Hamiltonian
        if not issubclass(type(operator), TimeDependentHamiltonian) and not callable(operator):
            operator = TimeDependentHamiltonian(
                [operator], [ConstantSchedule(1.0, 1.0)]
            )

        super().__init__(
            operator,
            variational_state,
            ode_solver,
            t0=t0,
            propagation_type=propagation_type,
            qgt=qgt,
            linear_solver=linear_solver,
            linear_solver_restart=linear_solver_restart,
            error_norm=error_norm,
        )
        self._uz = lambda t1, t2, **kwargs: Uz(
            self.state, self, self.state.model, t1, t2, **kwargs
        )

    def _iter(
        self,
        T: float,
        tstops: Sequence[float] | None = None,
        callback: Callable | None = None,
    ):
        """
        Implementation of :code:`iter`. This method accepts and additional `callback` object, which
        is called after every accepted step.
        """
        t_end = self.t + T
        if tstops is not None and (
            np.any(np.less(tstops, self.t)) or np.any(np.greater(tstops, t_end))
        ):
            raise ValueError(
                f"All tstops must be in range [t, t + T]=[{self.t}, {t_end}]"
            )

        if tstops is not None and len(tstops) > 0:
            tstops = np.sort(tstops)
            always_stop = False
        else:
            tstops = []
            always_stop = True

        while self.t < t_end:
            if always_stop or (
                len(tstops) > 0
                and (np.isclose(self.t, tstops[0]) or self.t > tstops[0])
            ):
                self._stop_count += 1
                yield self.t
                tstops = tstops[1:]

            # store the intermediate time since it will be automatically updated by the integrator
            # in the case of adaptive time-stepping, the mid_t is not really mid, but what matters
            # is only that we get to the same final time
            mid_t = self.t + self.dt

            # apply Uz
            self._integrator._state = self._integrator._state.replace(
                y=self._uz(self.t, mid_t)
            )

            # apply the rest of the Hamiltonian
            step_accepted = False
            while not step_accepted:
                if not always_stop and len(tstops) > 0:
                    max_dt = tstops[0] - self.t
                else:
                    max_dt = None
                step_accepted = self._integrator.step(max_dt=max_dt)
                if self._integrator.errors:
                    raise RuntimeError(
                        f"ODE integrator: {self._integrator.errors.message()}"
                    )
                elif self._integrator.warnings:
                    warnings.warn(
                        f"ODE integrator: {self._integrator.warnings.message()}",
                        UserWarning,
                        stacklevel=3,
                    )

            # apply Uz
            self._integrator._state = self._integrator._state.replace(
                y=self._uz(mid_t, self.t)
            )

            self._step_count += 1
            # optionally call callback
            if callback:
                callback()

        # Yield one last time if the remaining tstop is at t_end
        if (always_stop and np.isclose(self.t, t_end)) or (
            len(tstops) > 0 and np.isclose(tstops[0], t_end)
        ):
            yield self.t

    def __repr__(self):
        return "{}(t0={}, state={}, generator={}, solver={})".format(
            "SemiExactTDVP",
            self.t0,
            self.state,
            self._generator_repr,
            self.ode_solver,
        )


@odefun.dispatch
def Ux(state: MCState | FullSumState, driver: SemiExactTDVP, t, w, *, stage=0):
    # pylint: disable=protected-access

    state.parameters = w
    state.reset()

    # since
    driver._loss_stats = state.expect(driver.generator(t))

    ## CHANGE HERE : only apply Hx
    op_t = Hx(driver.generator, t)
    _, driver._loss_forces = state.expect_and_forces(
        op_t,
    )
    driver._loss_grad = _map_parameters(
        driver._loss_forces,
        state.parameters,
        driver._loss_grad_factor,
        driver.propagation_type,
        type(state),
    )

    qgt = driver.qgt(driver.state)
    if stage == 0:  # TODO: This does not work with FSAL.
        driver._last_qgt = qgt

    initial_dw = None if driver.linear_solver_restart else driver._dw
    driver._dw, _ = qgt.solve(driver.linear_solver, driver._loss_grad, x0=initial_dw)

    # If parameters are real, then take only real part of the gradient (if it's complex)
    driver._dw = tree_cast(driver._dw, state.parameters)

    return driver._dw


@dispatch
def Hx(hamiltonian, t, **kwargs):
    r"""
    Generates the off-diagonal hamiltonain at time t.
    Must be `dispatch`ed for each specific Hamiltonian.

    Args:
        hamiltonian: The generator of the dynamics
        t: The time parameter.

    Returns:
        The resulting off-diagonal operator at time `t`.
            default: The complete hamiltonian at time `t`.
    """

    return hamiltonian(t)


@dispatch
def Uz(state, driver, machine, t1, t2, **kwargs):
    r"""
    Updates the parameters of the wave function by exactly applying the diagonal part of the Hamiltonian,
    i.e. `Uz = exp(-i B(t) Hz dt/2)`. Other variants (in particular concerning the way to deal with t1 and t2) can be implemented.
    Must be `dispatch`ed for each specific Hamiltonian and model.

    Args:
        state: The state to be evolved.
        driver: The driver.
        machine: The variational-wave-function model.
        t1: The initial time.
        t2: The final time.

    Returns:
        The updated parameters of the state.
            default: The same parameters as the input state.
    """

    return state.parameters
