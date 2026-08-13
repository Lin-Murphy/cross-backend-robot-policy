"""SO101 adapter for an already approved, connected and audited LeRobot session.

Construction never connects or enables motors. A caller must supply the trial
specific guard/audit and a current-position stop callback before dispatch.
"""
import time
import numpy as np
from .execution_contract import (BackendCapabilities, BackendObservation,
                                 ActionReceipt, StopReceipt)
from .legacy_so101_sync_guard import NAMES

CAMERAS = ('follower', 'camera2')
UNITS = ('deg',) * 5 + ('calibrated_0_100',)


class SO101LeRobotBackendAdapter:
    def __init__(self, robot, guard, stop_current, *, approved_trial_id, send_action=None):
        if not approved_trial_id or guard is None or not callable(stop_current):
            raise ValueError('approved audited trial and current-position stop required')
        self.robot = robot
        self.guard = guard
        self.stop_current = stop_current
        self.send_action = send_action or robot.send_action
        self.trial_id = approved_trial_id
        self.capabilities = BackendCapabilities('so101', NAMES, UNITS, CAMERAS,
            'host_monotonic', 'manual', 'operator', 'sync_transport_return', True)
        self.stopped = False
        self.observation_index = 0

    def observe(self):
        started = time.perf_counter_ns()
        return self.capture(self.robot.get_observation(), started_ns=started)

    def capture(self, observation, *, started_ns):
        if self.stopped:
            raise RuntimeError('SO101 adapter stopped')
        now = time.perf_counter_ns()
        if self.guard.source_observation_ns is None or self.guard.source_observation_ns < started_ns or self.guard.latched:
            raise RuntimeError('fresh guarded observation required')
        images = {name: np.asarray(observation[name]) for name in CAMERAS}
        state = tuple(float(observation[name + '.pos']) for name in NAMES)
        identity = f'{self.trial_id}:obs:{self.observation_index}:{now}'
        self.observation_index += 1
        # LeRobot provides neither camera exposure nor state acquisition stamp.
        return BackendObservation(identity, NAMES, UNITS, state, images,
            {name: identity + ':' + name for name in CAMERAS},
            {name: None for name in CAMERAS}, None, 'host_monotonic')

    def dispatch(self, request):
        if self.stopped:
            return ActionReceipt(False, request.target, 'sync_transport_return', 0,
                                 time.perf_counter_ns(), 'backend_stopped')
        if self.guard.feedback is None or self.guard.source_observation_ns is None:
            return ActionReceipt(False, request.target, 'sync_transport_return', 0,
                                 time.perf_counter_ns(), 'guard_missing_fresh_observation')
        before = self.guard.goal_packets
        action = {name + '.pos': value for name, value in zip(NAMES, request.target)}
        self.send_action(action)
        if self.guard.goal_packets != before + 1:
            raise RuntimeError('one audited six-motor sync packet required')
        return ActionReceipt(True, request.target, 'sync_transport_return', 1,
                             time.perf_counter_ns())

    def stop(self):
        self.stopped = True
        receipt = self.stop_current()
        if not isinstance(receipt, StopReceipt):
            raise RuntimeError('stop callback must return audited StopReceipt')
        receipt.validate(self.capabilities)
        if not receipt.completed or receipt.physical_hold_dispatches != 1:
            raise RuntimeError('one audited current-position hold transport return required')
        return receipt
