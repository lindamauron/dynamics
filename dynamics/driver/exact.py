from netket.utils.types import PyTree
import numpy as np

from typing import Callable, Sequence
import warnings

from typing import Callable, Optional
from collections.abc import Sequence

from tqdm import tqdm

from netket.driver import AbstractVariationalDriver
from netket.driver.abstract_variational_driver import _to_iterable
from netket.logging.json_log import JsonLog
from netket.vqs import (
    VariationalState,
)
from netket.operator import AbstractOperator

from ..operators.hamiltonian import TimeDependentHamiltonian
from netket.experimental.dynamics import RungeKuttaIntegrator


_Nan = float("NaN")

def expect(psi, Op, norm=1):
    return psi.conj().T@(Op@psi)/norm


class EactEvolution:
    """
    Variational time evolution based on the time-dependent variational principle solved analytically for a mean-field ansatz.
    The solution being analytical, it can only be used for Rydberg hamiltonians. 

    .. note::
        This TDVP Driver only works for ansatzes with an external mean-field part. It will put all other parameters to zero. 
        The integrator in use is RK4. 

    """

    def __init__(
        self,
        operator: TimeDependentHamiltonian,
        initial_state: VariationalState,
        integrator: RungeKuttaIntegrator,
        *,
        t0: float = 0.0,
    ):
        r"""
        Initializes the time evolution driver.

        Args:
            operator: The generator of the dynamics (RydbergHamiltonian).
            variational_state: The variational state.
            integrator: the integrator of the ode for the parameters
            t0: Initial time at the start of the time evolution.
        """

        # First, prepare the operator as a list of sparse matrices
        self.sparse_generator = operator.to_sparse()
        self.frequencies = operator.frequencies
        self._generator_repr = repr(operator)
        self._step_count = 0
        

        def psi_dot(t, psi, **_):
            hpsis = [h@psi for h in self.sparse_generator]
            fs = [f(t) for f in self.frequencies]

            return sum([(-1j*f)*h for f,h in zip(fs,hpsis)])
        
        self._integrator = integrator(psi_dot, t0, initial_state)
        self._integrator._rkstate = self._integrator._rkstate.replace(y=self.state/np.linalg.norm(self.state))

        self._loss_stats = self._forward(self.t)
        self._loss_name="Generator"

    @property
    def integrator(self):
        return self._integrator
    @property
    def state(self):
        return self._integrator.y
    @property
    def t(self):
        return self._integrator.t
    @property
    def step_count(self):
        return self._step_count
 
    def __repr__(self):
        return (
            "TDVP_MF("
            + f"\n  generator = {self._generator_repr},"
            + f"\n  integrator = {self._integrator},"
            + f"\n  time = {self.t},"
            + f"\n  state = {self.state}"
            +"\n)"
        )


    def _forward(self, t):

        hs = [expect(self.state, h) for h in self.sparse_generator]
        fs = [f(t) for f in self.frequencies]

        forward = sum([f*h for f,h in zip(fs,hs)])
    

        return forward

    
    def iter(self, T: float, *, tstops: Optional[Sequence[float]] = None):
        """
        Returns a generator which advances the time evolution for an interval
        of length :code:`T`, stopping at :code:`tstops`.

        Args:
            T: Length of the integration interval.
            tstops: A sequence of stopping times, each within the interval :code:`[self.t0, self.t0 + T]`,
                at which this method will stop and yield. By default, a stop is performed
                after each time step (at potentially varying step size if an adaptive
                integrator is used).
        Yields:
            The current step count.
        """
        yield from self._iter(T, tstops)
    
    def _iter(
        self,
        T: float,
        tstops: Optional[Sequence[float]] = None,
        callback: Callable = None,
    ):
        """
        Implementation of :code:`iter`. This method accepts and additional `callback` object, which
        is called after every accepted step.
        """
        t_end = self.t + T
        if tstops is not None and (
            np.any(np.less(tstops, self.t)) or np.any(np.greater(tstops, t_end))
        ):
            raise ValueError(f"All tstops must be in range [t, t + T]=[{self.t}, {T}]")

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
                yield self.t
                tstops = tstops[1:]

            step_accepted = False
            while not step_accepted:
                if not always_stop and len(tstops) > 0:
                    max_dt = tstops[0] - self.t
                else:
                    max_dt = None

                    
                step_accepted = self.integrator.step()
                if self.integrator.errors:
                    raise RuntimeError(
                        f"RK solver: {self.integrator.errors.message()}"
                    )
                elif self.integrator.warnings:
                    warnings.warn(
                        f"RK solver: {self.integrator.warnings.message()}",
                        UserWarning,
                        stacklevel=3,
                    )
            self._loss_stats = self._forward(self.t)

            self._step_count += 1
            # optionally call callback
            if callback:
                callback()

        # Yield one last time if the remaining tstop is at t_end
        if (always_stop and np.isclose(self.t, t_end)) or (
            len(tstops) > 0 and np.isclose(tstops[0], t_end)
        ):
            yield self.t

    
    def run(
        self,
        T,
        out=(),
        obs=None,
        *,
        tstops=None,
        show_progress=True,
        callback=None,
    ):
        """
        Runs the time evolution.

        By default uses :class:`netket.logging.JsonLog`. To know about the output format
        check it's documentation. The logger object is also returned at the end of this function
        so that you can inspect the results without reading the json output.

        Args:
            T: The integration time period.
            out: A logger object, or an iterable of loggers, to be used to store simulation log and data.
                If this argument is a string, it will be used as output prefix for the standard JSON logger.
            obs: An iterable containing the observables that should be computed.
            tstops: A sequence of stopping times, each within the interval :code:`[self.t0, self.t0 + T]`,
                at which the driver will stop and perform estimation of observables, logging, and execute
                the callback function. By default, a stop is performed after each time step (at potentially
                varying step size if an adaptive integrator is used).
            show_progress: If true displays a progress bar (default=True)
            callback: Callable or list of callable callback functions to be executed at each
                stopping time.
        """
        if obs is None:
            obs = {}
        # sprsify the observables
        for o in obs:
            if issubclass(type(o), AbstractOperator):
                obs[o] = obs[o].to_sparse()

        if callback is None:
            callback = lambda *_args, **_kwargs: True

        if out is None:
            loggers = ()
        # if out is a path, create an overwriting Json Log for output
        elif isinstance(out, str):
            loggers = (JsonLog(out+"exact", "w", save_params=False),)
        else:
            loggers = _to_iterable(out)

        callbacks = _to_iterable(callback)
        if type(callbacks)==tuple:
            if isinstance(out, str):
                callbacks = callbacks + (callback_save(out), )
            else:
                callbacks = callbacks + (callback_save(''), )
        elif type(callbacks)==list:
            if isinstance(out, str):
                callbacks.append( callback_save(out) )
            else:
                callbacks.append( callback_save('') )
                pass

        callback_stop = False

        t_end = np.asarray(self.t + T)
        with tqdm(
            total=t_end,
            disable=not show_progress,
            unit_scale=True,
            dynamic_ncols=True,
        ) as pbar:
            first_step = True

            # We need a closure to pass to self._iter in order to update the progress bar even if
            # there are no tstops
            def update_progress_bar():
                # Reset the timing of tqdm after the first step to ignore compilation time
                nonlocal first_step
                if first_step:
                    first_step = False
                    pbar.unpause()

                pbar.n = min(np.asarray(self.t), t_end)
                self._postfix["n"] = self.step_count
                self._postfix.update(
                    {
                        self._loss_name: f"{self._loss_stats:.4f}",
                    }
                )

                pbar.set_postfix(self._postfix)
                pbar.refresh()

            for step in self._iter(T, tstops=tstops, callback=update_progress_bar):
                log_data = {}
                for o in obs:
                    log_data[o] = expect(self.state,obs[o])

                self._log_additional_data(log_data, self.t)

                self._postfix = {"n": self.step_count}
                # if the cost-function is defined then report it in the progress bar
                if self._loss_stats is not None:
                    self._postfix.update(
                        {
                            self._loss_name: str(self._loss_stats),
                        }
                    )
                    log_data[self._loss_name] = self._loss_stats
                pbar.set_postfix(self._postfix)

                # Execute callbacks before loggers because they can append to log_data
                for callback in callbacks:
                    if not callback(step, log_data, self):
                        callback_stop = True

                for logger in loggers:
                    logger(float(self.t), log_data, self.state)

                if len(callbacks) > 0:
                    if callback_stop:
                        break
                update_progress_bar()

            # Final update so that it shows up filled.
            update_progress_bar()

        # flush at the end of the evolution so that final values are saved to
        # file
        for logger in loggers:
            logger.flush(self.state)

        return loggers
    
    
    def _log_additional_data(self, log_dict, step):

        log_dict["t"] = self.t

        return 



class callback_save:
    def __init__(self,folder=''):
        import os

        self.folder = folder+'states/'
        self.pars = []
        if not os.path.exists(self.folder):
            os.makedirs(os.path.dirname(self.folder), exist_ok=True)

    def __call__(self,step, log_data, driver):

        np.save(self.folder+f'{step:.4f}.npy', driver.state )

        return True
