"""Model-independent execution lifecycle over the backend contract.

Schedulers may supply a captured observation; the session never performs an
extra device read in that mode. Device setup/reset and policy queues belong to
adapters, not this lifecycle.
"""
from dataclasses import asdict
from typing import Protocol

from .execution_contract import ActionRequest


class ActionProvider(Protocol):
    """Adapt model inference or a scheduled chunk to one backend action."""
    def __call__(self, observation) -> ActionRequest: ...


class ExecutionSession:
    def __init__(self, backend, record=None):
        self.backend = backend
        self.capabilities = backend.capabilities.validate()
        self.record = record or (lambda event: None)
        self.closed = False
        self.stop_receipt = None
        self.stop_error = None
        self.steps = 0

    def stop(self):
        """Attempt backend stop once, including when recording/dispatch fails."""
        if self.closed:
            if self.stop_error is not None:
                raise self.stop_error
            return self.stop_receipt
        self.closed = True
        try:
            self.stop_receipt = self.backend.stop().validate(self.capabilities)
        except BaseException as exc:
            self.stop_error = exc
            raise
        self.record({'event': 'execution_stop', 'receipt': asdict(self.stop_receipt)})
        return self.stop_receipt

    def step(self, action_provider: ActionProvider, *, observation=None):
        if self.closed:
            raise RuntimeError('execution session is closed')
        try:
            frame = self.backend.observe() if observation is None else observation
            frame.validate(self.capabilities)
            self.record({'event': 'execution_observation', 'step': self.steps,
                         'observation_id': frame.observation_id,
                         'clock_domain': frame.clock_domain,
                         'state_capture_ns': frame.state_capture_ns,
                         'camera_frame_ids': frame.camera_frame_ids,
                         'camera_capture_ns': frame.camera_capture_ns})
            request = action_provider(frame)
            if not isinstance(request, ActionRequest):
                raise ValueError('provider must return ActionRequest')
            request.validate(frame, self.capabilities)
            self.record({'event': 'execution_request', 'step': self.steps,
                         'request': asdict(request)})
            receipt = self.backend.dispatch(request).validate(request, self.capabilities)
            self.record({'event': 'execution_receipt', 'step': self.steps,
                         'observation_id': frame.observation_id, 'receipt': asdict(receipt)})
            self.steps += 1
            stop = None if receipt.accepted else self.stop()
            return frame, request, receipt, stop
        except BaseException as exc:
            try:
                self.stop()
            except BaseException as stop_exc:
                exc.add_note(f'Backend stop also failed: {stop_exc!r}')
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.stop()
        except BaseException as stop_exc:
            if exc is None:
                raise
            exc.add_note(f'Backend stop also failed: {stop_exc!r}')
        return False
