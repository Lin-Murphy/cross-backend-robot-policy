"""Shared execution boundary; it never supplies a physical safety profile."""
from dataclasses import dataclass
from typing import Protocol
import math
import numpy as np

@dataclass(frozen=True)
class BackendCapabilities:
    backend: str
    joint_names: tuple[str, ...]
    action_units: tuple[str, ...]
    cameras: tuple[str, ...]
    clock_domain: str
    reset_mode: str
    feedback_mode: str
    dispatch_ack_scope: str
    physical: bool

    def validate(self):
        if not self.backend or not self.joint_names or len(set(self.joint_names)) != len(self.joint_names) or \
           any(not name for name in self.joint_names) or len(self.action_units) != len(self.joint_names) or \
           any(not unit for unit in self.action_units) or not self.clock_domain:
            raise ValueError('invalid backend action/clock capabilities')
        if len(set(self.cameras)) != len(self.cameras) or any(not name for name in self.cameras):
            raise ValueError('invalid camera capabilities')
        if self.reset_mode not in ('automatic', 'manual', 'unavailable') or \
           self.feedback_mode not in ('sim_truth', 'operator', 'none') or \
           self.dispatch_ack_scope not in ('simulation_step', 'sync_transport_return', 'fake_only') or \
           type(self.physical) is not bool:
            raise ValueError('invalid backend behavior capabilities')
        if self.physical and self.dispatch_ack_scope == 'fake_only':
            raise ValueError('physical backend cannot report fake dispatch')
        return self


@dataclass(frozen=True)
class BackendObservation:
    observation_id: str
    joint_names: tuple[str, ...]
    joint_units: tuple[str, ...]
    joint_position: tuple[float, ...]
    images_rgb: dict[str, np.ndarray]
    camera_frame_ids: dict[str, str]
    camera_capture_ns: dict[str, int | None]
    state_capture_ns: int | None
    clock_domain: str

    def validate(self, capabilities):
        c = capabilities.validate()
        if not self.observation_id or self.joint_names != c.joint_names or \
           self.joint_units != c.action_units or self.clock_domain != c.clock_domain or \
           len(self.joint_position) != len(c.joint_names) or any(not math.isfinite(x) for x in self.joint_position):
            raise ValueError('observation identity, state, units or clock mismatch')
        if set(self.images_rgb) != set(c.cameras) or set(self.camera_frame_ids) != set(c.cameras) or \
           set(self.camera_capture_ns) != set(c.cameras):
            raise ValueError('camera capability mismatch')
        if any(not value for value in self.camera_frame_ids.values()):
            raise ValueError('missing camera frame identity')
        if any(not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3 or
               image.dtype != np.uint8 for image in self.images_rgb.values()):
            raise ValueError('invalid RGB image')
        for stamp in (*self.camera_capture_ns.values(), self.state_capture_ns):
            if stamp is not None and (type(stamp) is not int or stamp < 0):
                raise ValueError('invalid capture timestamp')
        return self


@dataclass(frozen=True)
class ActionRequest:
    source_observation_id: str
    joint_names: tuple[str, ...]
    action_units: tuple[str, ...]
    target: tuple[float, ...]

    def validate(self, observation, capabilities):
        if self.source_observation_id != observation.observation_id or \
           self.joint_names != capabilities.joint_names or self.action_units != capabilities.action_units or \
           len(self.target) != len(capabilities.joint_names) or any(not math.isfinite(x) for x in self.target):
            raise ValueError('action source, joint order, units or values mismatch')
        return self


def action_request_from_policy(observation, capabilities, *, policy_joint_names,
                               policy_action_units, target, backend_to_policy):
    """Map named policy actions into backend order with an explicit bijection.

    The mapping never changes units; adapters must declare matching units per joint.
    """
    observation.validate(capabilities)
    names, units, values = tuple(policy_joint_names), tuple(policy_action_units), tuple(target)
    mapping = dict(backend_to_policy)
    if not names or len(names) != len(set(names)) or len(units) != len(names) or len(values) != len(names):
        raise ValueError('invalid policy joint action dimensions')
    if set(mapping) != set(capabilities.joint_names) or set(mapping.values()) != set(names) or \
       len(mapping) != len(names) or len(set(mapping.values())) != len(names):
        raise ValueError('policy/backend joint mapping is not bijective')
    by_name = {name: (unit, value) for name, unit, value in zip(names, units, values)}
    ordered = []
    for backend_name, backend_unit in zip(capabilities.joint_names, capabilities.action_units):
        policy_unit, value = by_name[mapping[backend_name]]
        if policy_unit != backend_unit:
            raise ValueError('policy/backend action unit mismatch')
        ordered.append(value)
    request = ActionRequest(observation.observation_id, capabilities.joint_names,
                            capabilities.action_units, tuple(ordered))
    return request.validate(observation, capabilities)


@dataclass(frozen=True)
class ActionReceipt:
    accepted: bool
    target: tuple[float, ...]
    ack_scope: str
    physical_dispatches: int
    backend_clock_ns: int | None
    reason: str | None = None

    def validate(self, request, capabilities):
        if type(self.accepted) is not bool or tuple(self.target) != request.target or \
           self.ack_scope != capabilities.dispatch_ack_scope or \
           type(self.physical_dispatches) is not int or self.physical_dispatches < 0 or \
           self.physical_dispatches > 1:
            raise ValueError('invalid backend action receipt')
        if not self.accepted and self.physical_dispatches:
            raise ValueError('rejected action cannot claim physical dispatch')
        if not capabilities.physical and self.physical_dispatches:
            raise ValueError('nonphysical backend cannot claim physical dispatch')
        if self.backend_clock_ns is not None and (type(self.backend_clock_ns) is not int or self.backend_clock_ns < 0):
            raise ValueError('invalid receipt clock')
        if self.accepted and self.reason is not None:
            raise ValueError('accepted receipt cannot have rejection reason')
        if not self.accepted and not self.reason:
            raise ValueError('rejected receipt requires reason')
        return self


@dataclass(frozen=True)
class StopReceipt:
    completed: bool
    mechanism: str
    physical_hold_dispatches: int
    ack_scope: str

    def validate(self, capabilities):
        if type(self.completed) is not bool or not self.mechanism or \
           type(self.physical_hold_dispatches) is not int or self.physical_hold_dispatches not in (0, 1) or \
           self.ack_scope != capabilities.dispatch_ack_scope:
            raise ValueError('invalid stop receipt')
        if not capabilities.physical and self.physical_hold_dispatches:
            raise ValueError('nonphysical backend cannot claim hold dispatch')
        return self


class ExecutionBackend(Protocol):
    capabilities: BackendCapabilities
    def observe(self) -> BackendObservation: ...
    def dispatch(self, request: ActionRequest) -> ActionReceipt: ...
    def stop(self) -> StopReceipt: ...


def execute_from_observation(backend: ExecutionBackend, observation: BackendObservation,
                             request: ActionRequest):
    """Validate and dispatch an already captured frame without another device read."""
    capabilities = backend.capabilities.validate()
    try:
        observation.validate(capabilities)
        request.validate(observation, capabilities)
        receipt = backend.dispatch(request).validate(request, capabilities)
    except BaseException:
        backend.stop()
        raise
    stop_receipt = None
    if not receipt.accepted:
        stop_receipt = backend.stop().validate(capabilities)
    return request, receipt, stop_receipt


def execute_one(backend: ExecutionBackend, action_provider):
    """One common capture/dispatch/stop boundary; provider returns an action request."""
    try:
        observation = backend.observe()
        request = action_provider(observation)
        if not isinstance(request, ActionRequest):
            raise ValueError('provider must return ActionRequest')
    except BaseException:
        backend.stop()
        raise
    request, receipt, stop_receipt = execute_from_observation(backend, observation, request)
    return observation, request, receipt, stop_receipt
