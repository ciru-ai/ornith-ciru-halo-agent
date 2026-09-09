"""Project-owned worker lifecycle, retained from Ciru's tilebank adapter.

Only inherited methods used by the G256 worker are included. Model admission
and native binding belong to OrnithG256Worker.
"""
# Copyright 2026 Ciru.
from vllm.v1.worker.gpu_worker import Worker as VllmGPUWorker


class OrnithWorkerBase(VllmGPUWorker):
    def _require_lifecycle(self):
        if self._ornith_lifecycle is None:
            raise RuntimeError('Ornith worker model must load and bind before execution')
        self._ornith_lifecycle.raise_if_failed()
        return self._ornith_lifecycle

    def _startup_region(self, name, operation):
        lifecycle = self._require_lifecycle()
        if self._ornith_pending is not None:
            raise RuntimeError('Startup/profile cannot overlap an unfinished worker step')
        ticket = lifecycle.begin(name)
        try:
            result = operation()
        except Exception as error:
            lifecycle.abort(ticket, error)
        lifecycle.queue(ticket)
        lifecycle.complete(ticket)
        return result

    def determine_available_memory(self):
        return self._startup_region('profile-memory', super().determine_available_memory)

    def compile_or_warm_up_model(self):
        return self._startup_region('warmup-and-capture', super().compile_or_warm_up_model)

    def execute_model(self, scheduler_output):
        lifecycle = self._require_lifecycle()
        if self._ornith_pending is not None:
            raise RuntimeError('sample_tokens must complete the preceding Ornith execute_model step')
        ticket = lifecycle.begin('execute-model')
        try:
            output = super().execute_model(scheduler_output)
        except Exception as error:
            lifecycle.abort(ticket, error)
        # This D2H precedes the ordinary sampler's CPU completion event. No
        # device/error data is read here, and flags are not reset by sampling.
        lifecycle.queue(ticket)
        if output is None:
            self._ornith_pending = ticket
            return None
        return lifecycle.checked_output(output, ticket)

    def sample_tokens(self, grammar_output):
        lifecycle = self._require_lifecycle()
        ticket = self._ornith_pending
        if ticket is None:
            return super().sample_tokens(grammar_output)
        self._ornith_pending = None
        try:
            output = super().sample_tokens(grammar_output)
        except Exception as error:
            lifecycle.abort(ticket, error)
        return lifecycle.checked_output(output, ticket)

    def sleep(self, level=1):
        raise RuntimeError('Ornith worker sleep/reload is unsupported while native graphs own the arena')

    def wake_up(self, tags=None):
        raise RuntimeError('Ornith worker wake-up requires a fresh worker/model load')

    def reload_weights(self, *args, **kwargs):
        raise RuntimeError('Reloading sealed Ornith weights requires a fresh worker')

    def update_config(self, overrides):
        raise RuntimeError('Changing the sealed Ornith runtime requires a fresh worker')

    def update_max_model_len(self, max_model_len):
        raise RuntimeError('Changing Ornith capacity requires a fresh worker')

    def start_weight_update(self):
        raise RuntimeError('Ornith native weight updates require a fresh worker')

    def start_draft_weight_update(self):
        raise RuntimeError('Ornith draft weight updates are not admitted')

    def update_weights(self, update_info):
        raise RuntimeError('Ornith native weight updates require a fresh worker')

    def finish_weight_update(self):
        raise RuntimeError('Ornith native weight updates require a fresh worker')

    def shutdown(self):
        error = None
        try:
            if self._ornith_lifecycle is not None:
                self._ornith_lifecycle.drain()
        except Exception as caught:
            error = caught
        try:
            # The pinned parent first synchronizes and destroys captured model
            # graphs. The worker binding still strongly owns arena/library here.
            super().shutdown()
        except BaseException:
            # If parent graph destruction fails, retain native storage/library
            # until process teardown; releasing it here could dangle graph args.
            raise
        self._ornith_pending = self._ornith_lifecycle = self._ornith_binding = None
        if hasattr(self.vllm_config, '_ornith_grouped_runtime'):
            delattr(self.vllm_config, '_ornith_grouped_runtime')
        if error is not None:
            raise error
