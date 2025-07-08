import time

from flax import serialization

from netket.jax.sharding import extract_replicated

from netket.logging import JsonLog as _JsonLog
from netket.logging import StateLog as _StateLog
from netket.logging.state_log import save_binary_to_tar

### MAYBE get rid of extract_replicated ???
class JsonLog(_JsonLog):
    r"""
    This logger works as the netket logger, except that it saves the whole variational state (including samples)
    """
    def _flush_params(self, variational_state):
        if not self._save_params:
            return
        if variational_state is None:
            return

        self._last_flush_pars_time = time.time()
        binary_data = serialization.to_bytes(
            extract_replicated(variational_state)
        )
        if self._is_master_process:
            with open(self._prefix + ".mpack", "wb") as outfile:
                outfile.write(binary_data)
        self._last_flush_pars_runtime = time.time() - self._last_flush_pars_time

        self._flush_pars_time += self._last_flush_pars_runtime
        self._steps_notflushed_pars = 0


class StateLog(_StateLog):
    r"""
    This logger works as the netket logger, except that it saves the whole variational state (including samples)
    """

    def _save_variables(self, variational_state):
        if self._init is False:
            self._init_output()

        _time = time.time()
        binary_data = serialization.to_bytes(
            extract_replicated(variational_state)
        )

        if self._is_master_process:
            if self._tar:
                save_binary_to_tar(
                    self._tar_file, binary_data, str(self._file_step) + ".mpack"
                )
            else:
                with open(self._prefix + str(self._file_step) + ".mpack", "wb") as f:
                    f.write(binary_data)

        self._file_step += 1
        self._runtime_taken += time.time() - _time
