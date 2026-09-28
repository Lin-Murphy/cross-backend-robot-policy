"""Copyable two-joint example. No models, devices or simulation engine required."""
from cross_backend.execution_contract import (
    BackendCapabilities, BackendObservation, ActionReceipt, StopReceipt,
    action_request_from_policy,
)


class MemoryBackend:
    """Replace observation/dispatch I/O when implementing a real backend."""
    capabilities = BackendCapabilities('memory', ('slide', 'hinge'), ('m', 'rad'), (),
        'example_ticks', 'automatic', 'none', 'fake_only', False)

    def __init__(self):
        self.position = (0.0, 0.0)
        self.index = 0
        self.stopped = False

    def observe(self):
        return BackendObservation(f'memory:{self.index}', self.capabilities.joint_names,
            self.capabilities.action_units, self.position, {}, {}, {}, None, 'example_ticks')

    def dispatch(self, request):
        if self.stopped:
            return ActionReceipt(False, request.target, 'fake_only', 0, None, 'backend_stopped')
        self.position = request.target
        self.index += 1
        return ActionReceipt(True, request.target, 'fake_only', 0, None)

    def stop(self):
        self.stopped = True
        return StopReceipt(True, 'memory_latched', 0, 'fake_only')


class ExamplePolicy:
    """Replace the calculation with inference; preserve explicit names and units."""
    def __init__(self, capabilities):
        self.capabilities = capabilities

    def __call__(self, observation):
        slide, hinge = observation.joint_position
        # Model order intentionally differs from backend order.
        return action_request_from_policy(observation, self.capabilities,
            policy_joint_names=('model_hinge', 'model_slide'),
            policy_action_units=('rad', 'm'), target=(hinge + 0.02, slide + 0.01),
            backend_to_policy={'slide': 'model_slide', 'hinge': 'model_hinge'})
