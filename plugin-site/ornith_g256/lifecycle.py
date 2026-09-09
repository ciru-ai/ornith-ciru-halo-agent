"""Bounded worker-step error snapshots, separate from captured model work."""
# Copyright 2026 Ciru.
from collections import deque
from dataclasses import dataclass
import threading

import torch
from vllm.v1.outputs import AsyncModelRunnerOutput

def _outside_capture(device):
    if device.type == 'cuda':
        with torch.cuda.device(device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Worker flag reset/check is forbidden during graph capture')


class OrnithWorkerFault(RuntimeError):
    """A native error prevents this worker from delivering further outputs."""


@dataclass
class _Slot:
    host: object
    event: object


@dataclass
class StepTicket:
    sequence: int
    name: str
    slot: _Slot
    queued: bool = False
    completed: bool = False
    flags: tuple[int, ...] | None = None


class CudaFlagTransport:
    """One 160-byte D2H snapshot per 40-layer step, using preallocated storage."""
    def __init__(self, binding):
        self.binding = binding
        self.device = binding.backend.device
        if self.device.type != 'cuda':
            raise ValueError('Production worker flag transport requires ROCm GPU storage')
        self.fallback_synchronizations = 0

    def allocate_slot(self):
        _outside_capture(self.device)
        return _Slot(torch.empty(len(self.binding.layer_prefixes), dtype=torch.int32,
                                 device='cpu', pin_memory=True),
                     torch.cuda.Event(blocking=True))

    def queue(self, slot):
        _outside_capture(self.device)
        with torch.cuda.device(self.device):
            slot.host.copy_(self.binding.error_flags, non_blocking=True)
            slot.event.record(torch.cuda.current_stream(self.device))

    def finish(self, slot):
        _outside_capture(self.device)
        with torch.cuda.device(self.device):
            # Normal sampling already waited for its later D2H event. A direct
            # return/startup/error path may need this one step-level wait.
            if not slot.event.query():
                self.fallback_synchronizations += 1
                slot.event.synchronize()
        return tuple(int(value) for value in slot.host.tolist())


class WorkerStepLifecycle:
    """Own snapshots across execute→sample and asynchronous output completion.

    Model calls stay serialized on one worker stream. Tickets permit bounded
    overlap with CPU output handling, not concurrent use of the native arena.
    Earlier tickets are checked before a later output can be delivered.
    """
    def __init__(self, binding, max_inflight, *, _transport=None):
        if type(max_inflight) is not int or not 1 <= max_inflight <= 64:
            raise ValueError('Invalid bounded worker output capacity')
        self.binding = binding
        self.transport = _transport if _transport is not None else CudaFlagTransport(binding)
        self._free = deque(self.transport.allocate_slot() for _ in range(max_inflight))
        self._pending = {}
        self._lock = threading.RLock()
        self._sequence = 0
        self._fault = None
        self.resets = self.copies = self.completions = 0

    def begin(self, name):
        with self._lock:
            self.raise_if_failed()
            if not self._free:
                raise RuntimeError('Ornith output snapshot pool exhausted before a new worker step')
            slot = self._free.popleft()
            ticket = StepTicket(self._sequence, str(name), slot)
            self._sequence += 1
            try:
                self.binding.reset_error_flags()
            except BaseException as error:
                self._free.appendleft(slot)
                self.poison(error)
                raise
            self.resets += 1
            self._pending[ticket.sequence] = ticket
            return ticket

    def raise_if_failed(self):
        if self._fault is not None:
            raise self._fault

    def poison(self, error):
        """A failed GPU/worker boundary requires process replacement, not reuse."""
        with self._lock:
            if self._fault is None:
                self._fault = OrnithWorkerFault(
                    f'Ornith worker failed and cannot deliver more outputs: {type(error).__name__}: {error}')
                self._fault.__cause__ = error

    def queue(self, ticket):
        with self._lock:
            if ticket.completed or ticket.sequence not in self._pending:
                raise RuntimeError('Cannot snapshot a completed/unknown worker step')
            if ticket.queued:
                return
            try:
                self.transport.queue(ticket.slot)
            except BaseException as error:
                self.poison(error)
                raise
            ticket.queued = True
            self.copies += 1

    def complete(self, ticket):
        with self._lock:
            # Same-stream GPU ordering means completing a later output also
            # admits all earlier snapshots. Never let a healthy later step
            # conceal an earlier native error, even if consumers reorder calls.
            for sequence in sorted(self._pending):
                if sequence > ticket.sequence:
                    break
                earlier = self._pending[sequence]
                if not earlier.queued:
                    raise RuntimeError('Worker completion reached an unsnapshotted earlier step')
                try:
                    flags = self.transport.finish(earlier.slot)
                    if len(flags) != len(self.binding.layer_prefixes):
                        raise RuntimeError('Worker snapshot width changed')
                except BaseException as error:
                    self.poison(error)
                    raise
                earlier.flags, earlier.completed = flags, True
                if any(flags) and self._fault is None:
                    failed = {prefix: value for prefix, value in zip(self.binding.layer_prefixes, flags) if value}
                    self._fault = OrnithWorkerFault(
                        f'Ornith native worker step {earlier.sequence} ({earlier.name}) failed: {failed}')
                self._free.append(earlier.slot)
                del self._pending[sequence]
                self.completions += 1
            self.raise_if_failed()
            if not ticket.completed:
                raise RuntimeError('Unknown or incomplete Ornith worker ticket')
            return ticket.flags

    def abort(self, ticket, error):
        """Capture flags on an exceptional worker boundary and preserve cause."""
        try:
            if not ticket.queued:
                self.queue(ticket)
            self.complete(ticket)
        except BaseException as flag_error:
            if isinstance(flag_error, OrnithWorkerFault):
                if flag_error is error:
                    raise
                raise flag_error from error
            error.add_note(f'Ornith flag capture also failed: {type(flag_error).__name__}: {flag_error}')
        self.poison(error)
        raise error

    def checked_output(self, output, ticket):
        if isinstance(output, AsyncModelRunnerOutput):
            return CheckedAsyncOutput(output, self, ticket)
        self.complete(ticket)
        return output

    def drain(self):
        with self._lock:
            tickets = list(self._pending.values())
        for ticket in tickets:
            if not ticket.queued:
                self.queue(ticket)
        if tickets:
            self.complete(tickets[-1])
        self.raise_if_failed()


class CheckedAsyncOutput(AsyncModelRunnerOutput):
    """The existing executor calls get_output before delivering any tokens."""
    def __init__(self, delegate, lifecycle, ticket):
        self._delegate, self._lifecycle, self._ticket = delegate, lifecycle, ticket
        self._used = False

    def get_output(self):
        if self._used:
            raise RuntimeError('Async Ornith worker output may be consumed only once')
        self._used = True
        try:
            output = self._delegate.get_output()
        except BaseException as error:
            self._lifecycle.abort(self._ticket, error)
        self._lifecycle.complete(self._ticket)
        return output
