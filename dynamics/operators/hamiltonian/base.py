import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import netket.jax as nkjax

from typing import Tuple, List
from netket.utils.types import Array, PyTree, Callable
from scipy.sparse import csr_matrix as _csr_matrix

from ...schedules import Schedule as _Schedule
from netket.operator import AbstractOperator as _AbstractOperator
from netket.operator import LocalOperatorJax as _JaxOperator

from netket.vqs import MCState
from netket.hilbert import Spin as _SpinHilbert
from netket.stats import statistics, Stats

class TimeDependentHamiltonian:
    """
    Instantiates a callable time-dependent Hamiltonian.
    It is defined using a list of operators [h] and frequencies [f] such that
    ..math::
        H(t) = \sum_i f_i(t) h_i.
    The utilities `to_sparse`, `to_dense` and `to_jax_operator` act on the separated operators.
    """

    def __init__(
        self, operators: List[_AbstractOperator], frequencies: List[_Schedule]
    ):
        """
        operators : list of inidividual operators.
            All operators must act on the same Hilbert space.
            It is recommended to group (add) all operators that depend on the same frequency schedule for efficiency.
        frequencies : list of inidividual frequencies.
            All frequencies must act over the same annealing time.
        """
        if len(operators) != len(frequencies):
            raise AttributeError(
                f"There should be the same number of different schedules ({len(frequencies)}) as operators ({len(operators)})."
            )
        n = len(operators)
        self._hilbert = operators[0].hilbert
        T = frequencies[0].T
        H = []
        F = []
        for h, f in zip(operators, frequencies):
            if not issubclass(type(h), _AbstractOperator):
                raise ValueError("The operators should all be Netket's operators.")
            if h.hilbert != self.hilbert:
                raise ValueError("All operators must act on the same Hilbert space.")

            H.append(h)

            if not issubclass(type(f), _Schedule):
                raise ValueError(
                    "The schedules should all be instances of `Frequency`."
                )
            if f.T != T:
                raise ValueError("All frequencies must occure for the same time.")
            F.append(f)

        self._H = H
        self._F = F
        self._T = T
        self.n_operators = n

        self._str = self.__class__.__name__ + f"({self.hilbert}, \n \t F={self.frequencies}, \n \t O={self.operators}\n)"

    @property
    def hilbert(self) -> _SpinHilbert:
        """
        The hilbert space.
        """
        return self._hilbert

    @property
    def operators(self) -> List[_AbstractOperator]:
        """
        The list of operators acting in the Hamiltonian.
        """
        return self._H

    @property
    def frequencies(self) -> List[_Schedule]:
        """
        The list of frequencies acting on each operators of the Hamiltonian.
        """
        return self._F

    @property
    def annealing_time(self) -> float:
        """
        The total annealing time of the Hamiltonian.
        """
        return self._T

    def __repr__(self) -> str:
        """
        Representation of the class.
        """
        return self._str

    def __call__(self, t) -> _AbstractOperator:
        """
        The Hamiltonian at time t
        """
        return sum([f(t) * h for f, h in zip(self.frequencies, self.operators)])

    def plot(self, ax=None):
        """
        Plots each frequency schedule.
        """
        for f in self.frequencies:
            ax = f.plot(ax)

        return ax

    def to_sparse(self, jax_=False) -> List[_csr_matrix]:
        """
        Returns each individual operator, independent of time, as a sparse matrix

        returns : operators SparseMatrices
        """
        return [h.to_sparse(jax_=jax_) for h in self.operators]

    def to_dense(self) -> List[np.ndarray]:
        """
        Returns each individual operator, independent of time, as a sparse matrix

        returns : operators SparseMatrices
        """
        return [h.to_dense() for h in self.operators]

    def to_jax_operator(self) -> "TimeDependentHamiltonian":
        """
        Creates a time independent Hamiltonian with jax_operators instead of the actual ones.

        returns : TimeDependentHamiltonian with jax operators
        """
        new_operators = [h.to_jax_operator() for h in self.operators]

        return TimeDependentHamiltonian(new_operators, self.frequencies)

    @property
    def max_conn_size(self) -> int:
        """
        Total number of connections of all operators.
        """
        return sum(h.max_conn_size for h in self.operators)

    @property
    def dtype(self):
        return jnp.float64
        dtypes = [h.dtype for h in self.operators if hasattr(h, "dtype")]
        if not dtypes:
            return jnp.complex128
        dtype = dtypes[0]
        for dt in dtypes[1:]:
            dtype = jnp.promote_types(dtype, dt)
        return jnp.promote_types(dtype, jnp.complex128)
    
    def get_conn_padded(self, σ: Array, t: float) -> tuple[Array, Array]:
        freqs_t = tuple(jnp.asarray(frequency(t), dtype=self.dtype) for frequency in self.frequencies)

        eta_parts = []
        mel_parts = []
        for operator, freq_t in zip(self.operators, freqs_t):
            eta, m = operator.get_conn_padded(σ)
            eta_parts.append(eta)
            mel_parts.append(jnp.asarray(m, dtype=self.dtype) * freq_t)

        return jnp.concatenate(eta_parts, axis=1), jnp.concatenate(mel_parts, axis=1)
    
    def get_olocs(self, t: float, vs) -> Array:
        sigma = vs.samples.reshape(-1, self.hilbert.size)
        logpsi = vs._apply_fun
        pars = vs.variables
        # check that sigma has been reshaped to 2D, eta is 3D
        # sigma is (Nsamples, Nsites)
        assert sigma.ndim == 2

        if vs.chunk_size is None:
            olocs = self._get_olocs_jitted(logpsi, pars, sigma, t)
        else:
            olocs = self._get_olocs_chunked_jitted(
                logpsi,
                pars,
                sigma,
                t,
                vs.chunk_size,
            )

        return olocs.reshape(vs.samples.shape[:-1])
   
    def expect(self, t: float, vs: MCState) -> Stats:
        '''
        Memory-saving version of expect.
        Uses vs.chunk_size if available and falls back to the fast unchunked path otherwise.
        '''
        local_energies = self.get_olocs(t, vs)

        return statistics(local_energies)

    def expect_and_forces(self, t: float, vs: MCState, *, mutable=False):
        """
        Estimates the expectation value and forces for this time-dependent Hamiltonian.
        Uses chunked forward/VJP passes when vs.chunk_size is set.
        """
        sigma = vs.samples
        sigma_shape = sigma.shape
        if sigma.ndim != 2:
            sigma = sigma.reshape((-1, sigma_shape[-1]))

        model_apply_fun = vs._apply_fun
        parameters = vs.parameters
        model_state = vs.model_state

        if mutable is False:
            if vs.chunk_size is None:
                return self._expect_and_forces_jitted(
                    model_apply_fun,
                    parameters,
                    model_state,
                    sigma,
                    sigma_shape,
                    t,
                )
            else:
                return self._expect_and_forces_chunked_jitted(
                    model_apply_fun,
                    parameters,
                    model_state,
                    sigma,
                    sigma_shape,
                    t,
                    vs.chunk_size,
                )

        is_mutable = True
        if vs.chunk_size is not None:
            raise NotImplementedError(
                "Chunked expect_and_forces with mutable model state is not implemented."
            )

        O_loc = self._get_olocs_jitted(model_apply_fun, {"params": parameters, **model_state}, sigma, t)
        O_bar = statistics(O_loc.reshape(sigma_shape[:-1]))
        centered = O_loc - O_bar.mean

        _, vjp_fun, new_variables = nkjax.vjp(
            lambda w: model_apply_fun(
                {"params": w, **model_state}, sigma, mutable=mutable
            ),
            parameters,
            conjugate=True,
            has_aux=is_mutable,
        )
        forces = vjp_fun(jnp.conjugate(centered) / sigma.shape[0])[0]
        vs.model_state = new_variables[0]
        return O_bar, forces

    @partial(jax.jit, static_argnums=(0, 1, 5))
    def _expect_and_forces_jitted(
        self,
        model_apply_fun: Callable,
        parameters: PyTree,
        model_state: PyTree,
        sigma: Array,
        sigma_shape: tuple[int, ...],
        t: float,
    ):
        n_samples = sigma.shape[0]
        variables = {"params": parameters, **model_state}
        O_loc = self._get_olocs_jitted(model_apply_fun, variables, sigma, t)
        O_bar = statistics(O_loc.reshape(sigma_shape[:-1]))
        centered = O_loc - O_bar.mean

        _, vjp_fun = nkjax.vjp(
            lambda w: model_apply_fun({"params": w, **model_state}, sigma),
            parameters,
            conjugate=True,
        )
        forces = vjp_fun(jnp.conjugate(centered) / n_samples)[0]
        return O_bar, forces

    @partial(jax.jit, static_argnums=(0, 1, 5, 7))
    def _expect_and_forces_chunked_jitted(
        self,
        model_apply_fun: Callable,
        parameters: PyTree,
        model_state: PyTree,
        sigma: Array,
        sigma_shape: tuple[int, ...],
        t: float,
        chunk_size: int,
    ):
        n_samples = sigma.shape[0]
        variables = {"params": parameters, **model_state}
        O_loc = self._get_olocs_chunked_jitted(
            model_apply_fun,
            variables,
            sigma,
            t,
            chunk_size,
        )
        O_bar = statistics(O_loc.reshape(sigma_shape[:-1]))
        centered = O_loc - O_bar.mean

        vjp_fun = nkjax.vjp_chunked(
            lambda w, ms, s: model_apply_fun({"params": w, **ms}, s),
            parameters,
            model_state,
            sigma,
            conjugate=True,
            chunk_size=chunk_size,
            chunk_argnums=2,
            nondiff_argnums=(1, 2),
        )
        forces = vjp_fun(jnp.conjugate(centered) / n_samples)[0]
        return O_bar, forces

    @partial(jax.jit, static_argnums=(0, 1))
    def _get_olocs_jitted(
        self,
        logpsi: Callable,
        pars: PyTree,
        sigma: Array,
        t: float,
    ) -> Array:
        logpsi_fn = lambda s: logpsi(pars, s)

        logpsi_sigma = logpsi_fn(sigma)
        olocs = None
        for operator, freq in zip(self.operators, self.frequencies):
            eta, mels = operator.get_conn_padded(sigma)
            loc_vals = local_value_kernel(logpsi_fn, logpsi_sigma, (eta, mels))
            term = jnp.asarray(freq(t)) * loc_vals
            olocs = term if olocs is None else olocs + term

        return olocs

    @partial(jax.jit, static_argnums=(0, 1, 5))
    def _get_olocs_chunked_jitted(
        self,
        logpsi: Callable,
        pars: PyTree,
        sigma: Array,
        t: float,
        chunk_size: int,
    ) -> Array:
        logpsi_fn = nkjax.apply_chunked(
            lambda s: logpsi(pars, s),
            in_axes=0,
            chunk_size=chunk_size,
        )
        logpsi_sigma = logpsi_fn(sigma)
        olocs = None
        for operator, freq in zip(self.operators, self.frequencies):
            eta, mels = operator.get_conn_padded(sigma)
            loc_vals = local_value_kernel_chunked(
                logpsi_fn,
                logpsi_sigma,
                (eta, mels),
            )
            term = jnp.asarray(freq(t)) * loc_vals
            olocs = term if olocs is None else olocs + term

        return olocs
    
@partial(jax.jit, static_argnums=(0,))
def local_value_kernel(
    logpsi: Callable,
    logpsi_σ: Array,
    extra_args: Tuple[Array, Array],
):
    """
    local_value kernel for MCState for jax-compatible operators
    """
    σp, mel = extra_args
    logpsi_σp = logpsi(σp.reshape(-1, σp.shape[-1])).reshape(σp.shape[:-1])
    return local_value(logpsi_σ, logpsi_σp, mel) #jnp.sum(mel * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1)

@partial(jax.jit, static_argnums=(0,))
def local_value_kernel_chunked(
    logpsi: Callable,
    logpsi_σ: Array,
    extra_args: Tuple[Array, Array],
):
    """
    Chunked local-value kernel following NetKet's discrete-jax logic.
    If chunks are large compared to the connection count, chunk over samples.
    Otherwise, chunk the expensive logpsi evaluation over connected states.
    """
    σp, mel = extra_args
    logpsi_σp = logpsi(σp.reshape(-1, σ.shape[-1])).reshape(σp.shape[:-1])
    return local_value(logpsi_σ, logpsi_σp, mel)

@jax.jit
def local_value(logpsi_σ, logpsi_σp, mel):
    return jnp.sum(mel * jnp.exp(logpsi_σp - jnp.expand_dims(logpsi_σ, -1)), axis=-1)
