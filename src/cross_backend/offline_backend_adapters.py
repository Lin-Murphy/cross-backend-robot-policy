"""MuJoCo and saved-SO101 fake adapters for the same execution boundary."""
from pathlib import Path
import hashlib
import numpy as np
from .legacy_so101_sync_guard import NAMES as JOINTS
from .execution_contract import (BackendCapabilities, BackendObservation,
                                 ActionReceipt, StopReceipt)

CAMERAS = ('follower', 'camera2')


class SimJointBackendAdapter:
    def __init__(self, simulator):
        self.simulator = simulator
        self.capabilities = BackendCapabilities('mujoco', JOINTS, ('rad',) * 6, CAMERAS,
            'simulation', 'automatic', 'sim_truth', 'simulation_step', False)
        self.stopped = False

    def observe(self):
        state = self.simulator.state()
        frames = self.simulator.render()
        stamp = state['sim_ns']
        self.frames = frames
        identity = f'sim:{self.simulator.index}:{stamp}'
        return BackendObservation(identity, JOINTS, self.capabilities.action_units,
            tuple(state['joint_position']), {name: frames['observation.images.' + name] for name in CAMERAS},
            {name: identity + ':' + name for name in CAMERAS},
            {name: stamp for name in CAMERAS}, stamp, 'simulation')

    def dispatch(self, request):
        if self.stopped:
            return ActionReceipt(False, request.target, 'simulation_step', 0, None, 'backend_stopped')
        state = self.simulator.step_sim_targets(np.asarray(request.target, dtype=float))
        return ActionReceipt(True, request.target, 'simulation_step', 0, state['sim_ns'])

    def stop(self):
        self.stopped = True
        return StopReceipt(True, 'simulation_loop_latched', 0, 'simulation_step')


class SavedSO101FakeBackendAdapter:
    """Saved observation and fake sink; no bus/device import or physical send path."""
    def __init__(self, sample: Path, *, reject=False):
        self.sample = Path(sample)
        self.reject = reject
        self.sample_sha256 = hashlib.sha256(self.sample.read_bytes()).hexdigest()
        self.capabilities = BackendCapabilities('so101_saved_fake', JOINTS,
            ('deg',) * 5 + ('calibrated_0_100',), CAMERAS,
            'host_monotonic', 'manual', 'operator', 'fake_only', False)
        self.stopped = False
        self.fake_actions = []

    def observe(self):
        with np.load(self.sample) as data:
            state = tuple(float(x) for x in data['state'])
            images = {camera: data['observation.images.' + camera].copy() for camera in CAMERAS}
            for camera in CAMERAS:
                image = images[camera]
                if image.shape != (480, 640, 3) or image.dtype != np.uint8:
                    raise ValueError('invalid saved SO101 camera frame')
        identity = 'saved:' + self.sample_sha256
        # This NPZ has host read intervals but no verified exposure timestamp.
        return BackendObservation(identity, JOINTS, self.capabilities.action_units,
            state, images, {name: identity + ':' + name for name in CAMERAS},
            {name: None for name in CAMERAS}, None, 'host_monotonic')

    def dispatch(self, request):
        if self.stopped or self.reject:
            return ActionReceipt(False, request.target, 'fake_only', 0, None,
                                 'backend_stopped' if self.stopped else 'injected_rejection')
        self.fake_actions.append(request.target)
        return ActionReceipt(True, request.target, 'fake_only', 0, None)

    def stop(self):
        self.stopped = True
        return StopReceipt(True, 'fake_sink_latched', 0, 'fake_only')
