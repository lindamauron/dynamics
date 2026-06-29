import numpy as np
import netket as nk


class VscoreConvergence(nk.callbacks.ConvergenceStopping):
    def __init__(self, target: float, smoothing_window: int = 10, patience: int = 10,):
        super().__init__(target, 'V-score', smoothing_window=smoothing_window, patience=patience)

    def __call__(self, step, log_data, driver):
        N = driver.state.hilbert.size
        vscore = N * log_data[driver._loss_name].variance / (log_data[driver._loss_name].mean**2)
        log_data['V-score'] = vscore
                    
        loss = np.asarray(np.real(vscore))

        self._loss_window.append(loss)
        loss_smooth = np.mean(self._loss_window)

        if loss_smooth <= self.target:
            self._patience_counter += 1
        else:
            self._patience_counter = 0

        if self._patience_counter > self.patience:
            print(f"Convergence reached at step {step} with V-score {loss_smooth:.2e} (target was {self.target:.2e}).")
            return False

        return True
