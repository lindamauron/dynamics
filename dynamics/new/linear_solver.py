import jax.numpy as jnp
from netket.jax import tree_ravel
import jax

class LinearSolver:
    def __init__(self, rtol=1e-8, rtol_smooth=1e-7, error_norm="L2", ):
        self.rtol = rtol
        self.rtol_smooth = rtol_smooth

        if error_norm == "L2":
            self.norm_fn = jnp.linalg.norm
        elif error_norm == "L1":
            self.norm_fn = lambda x : jnp.sum(jnp.abs(x))
        else:
            raise ValueError(f"Unknown error norm {error_norm}")

        self._last_A = None
        self._last_b = None
        self._last_x = None

    def __call__(self, A, b, *, x0=None, stage=0):
        if stage==0 and self._last_A is not None:
            residual = self.norm_fn( self._last_A @ x0 - self._last_b )
            print(residual)

        solution = self._pinv(A, b, rtol=self.rtol, rtol_smooth=self.rtol_smooth, x0=x0, stage=stage)
    
    def _pinv(self, A, b, *, rtol, rtol_smooth, x0=None, stage=0):
        del x0
        if not isinstance(A, jax.Array):
            A = A.to_dense()
        b, unravel = tree_ravel(b)

        Σ, U = jnp.linalg.eigh(A)

        # Discard eigenvalues below numerical precision
        Σ_inv = jnp.where(jnp.abs(Σ / Σ[-1]) > rtol, jnp.reciprocal(Σ), 0.0)

        # Set regularizer for singular value cutoff
        regularizer = 1.0 / (1.0 + (rtol_smooth / jnp.abs(Σ / Σ[-1])) ** 6)

        Σ_inv = Σ_inv * regularizer

        x = U @ (Σ_inv * (U.conj().T @ b))

        # save to analyze for next step
        if stage==0:
            self._last_A = A
            self._last_b = b
            self._last_x = x

        return unravel(x), None
