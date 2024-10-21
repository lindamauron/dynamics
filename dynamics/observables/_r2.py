import jax.numpy as jnp

import netket as nk
import jax

from netket.optimizer.qgt import QGTAuto
from netket.stats import Stats

from netket.experimental.driver.tdvp import _map_parameters
from functools import partial
import warnings


class TDVPError:
    r"""
    TDVP error estimate r^2 mostly inspired from [1] but also [2].
    This operator estimates the error produced when solving the TDVP equation of motion
    .. math::
        S \dot{θ} = c_f F,
    with the QGT :math:`S_{ij} = \langle \partial_i \psi | \partial_j \psi \rangle - \langle \partial_i \psi | \psi \rangle \langl \psi | \partial_j \psi \rangle`
    and the vector of forces :math:`F_i = \langle \partial_i \psi | H | \psi \rangle - \langle \partial_i \psi | \psi \rangle \langle \psi | H | \psi \rangle`
    is multiplied with some coefficient :math:`c_f` according to the problem wished (real-time evolution :math:`c_f = -1j`, imaginary :math:`c_f = -1`).

    The second order estimate of the error is then expressed as
    .. math::
        r2 = 1 + \frac{\dot{\theta}^{\dagger}(S \dot{\theta} - c_f F) + (c_f F)^{\dagger} \dot{\theta}}{(\delta E)^2},
    where :math:`\delta E` is the variance of the energy estimate.

    This implementation can either estimate the "simple" error by solving the TDVP and directly using the same quantities to compute r2 ("tr"),
    or it can swap the :math:`\dot{\theta}` of indepentent samples to estimate it, to avoid overfitting ("val").
    Both of these quantities can be computed once or averaged multiple times to estimate errors bars.

    [1] Hoffman D., Fabiani G., Role of stochastic noise and generalization error in the time propagation of neural-network quantum states, https://scipost.org/10.21468/SciPostPhys.12.5.165 (2022)
    [2] Schmitt M., Heyl M., Quantum Many-Body Dynamics in Two Dimensions with Artificial Neural Networks, https://link.aps.org/doi/10.1103/PhysRevLett.125.100503 (2020)
    """

    def __init__(
        self,
        propagation_type="real",
        qgt=None,
        linear_solver=nk.optimizer.solver.pinv_smooth,
        mode="tr",
        boots=1,
    ):
        r"""
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
        mode: The error estimate to choose: "tr" for training, wihch compares the solution witht the same
            samples for all quantities and "val" for validation, which uses independent samples for the solution
            and the error estimate.
        boots: The number of 'bootstraps' to use to average the error.
            default: 1 (i.e. no averaging) for "tr", 2 for "val".
        """

        if qgt is None:
            qgt = QGTAuto(solver=linear_solver)
        self.qgt = qgt

        self.propagation_type = propagation_type

        self.linear_solver = linear_solver

        if propagation_type == "real":
            loss_grad_factor = -1.0j
        elif propagation_type == "imag":
            loss_grad_factor = -1.0
        else:
            raise ValueError("propagation_type must be one of 'real', 'imag'")

        self._map_parameters = partial(
            jax.jit(
                _map_parameters,
                static_argnames=("loss_grad_factor", "propagation_type", "state_T"),
            ),
            loss_grad_factor=loss_grad_factor,
            propagation_type=propagation_type,
        )
        self._last_dw = None

        if mode == "tr":
            self._shift = 0
        elif mode == "val":
            self._shift = 1
        else:
            raise ValueError(
                f'The mode can be either "tr" or "val", instad got "{mode}".'
            )

        if not (isinstance(boots, int) and boots > 0):
            raise ValueError(
                f"The number of bootstraps must be a positive integer, instead got {boots}."
            )

        if mode == "val" and boots == 1:
            warnings.warn(
                'With "val" mode, the number of boots must be min. 2, increased it automatically.'
            )
            boots = 2
        self.n_boots = boots

    def __call__(self, H, vs):
        r"""
        Finds the TDVP error of a state given an Hamiltonian.
        H : Hamiltonian generating the dynamics.
        vs: variational state

        returns: Stats(tdvpError).
        """

        def compute_eom_elements(i):
            vs.sample()
            _loss_stats, _loss_forces = vs.expect_and_forces(H)

            _loss_grad = self._map_parameters(
                forces=_loss_forces,
                parameters=vs.parameters,
                state_T=type(vs),
            )
            S = self.qgt(vs)

            dw, _ = S.solve(self.linear_solver, _loss_grad, x0=self._last_dw)

            dw, _ = nk.jax.tree_ravel(dw)
            G, _ = nk.jax.tree_ravel(_loss_grad)

            return _loss_stats.variance, G, S, dw

        dE, Gs, Ss, dws = jax.vmap(compute_eom_elements, out_axes=(0, 0, 0, 0))(
            jnp.arange(self.n_boots)
        )

        r2s = jax.vmap(R2, in_axes=(0, 0, 0, 0))(
            Ss, Gs, dE, jnp.roll(dws, self._shift, axis=0)
        )

        self._last_dw = jnp.mean(dws, axis=0)

        return Stats(
            mean=r2s.mean(),
            variance=r2s.var(),
            error_of_mean=jnp.sqrt(r2s.var() / self.n_boots),
        )


@jax.jit
def R2(S, G, dE, dw):
    return 1 + (dw.conj().T @ (S @ dw - G) - G.conj().T @ dw) / dE
