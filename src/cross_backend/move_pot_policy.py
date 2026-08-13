"""Named dual-camera ACT adapter and offline action provenance. No hardware imports.

Joint bounds must be supplied by a separately validated execution backend. The
old normalized-unit SO101CommandGate is not applied to degree-valued joints.
"""
from collections import deque
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Mapping, Protocol

import numpy as np

from .chunk_policy import ACTChunkAdapter

CAMERAS = ('observation.images.follower', 'observation.images.camera2')
JOINTS = ('shoulder_pan.pos', 'shoulder_lift.pos', 'elbow_flex.pos',
          'wrist_flex.pos', 'wrist_roll.pos', 'gripper.pos')
UNITS = ('deg', 'deg', 'deg', 'deg', 'deg', 'calibrated_0_100')


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class MovePotObservation:
    images_rgb: Mapping[str, np.ndarray]
    state: np.ndarray
    observation_id: str
    frame_ids: Mapping[str, str]
    camera_capture_ns: Mapping[str, int | None]
    state_capture_ns: int | None
    joint_names: tuple[str, ...] = JOINTS
    joint_units: tuple[str, ...] = UNITS
    dataset_timestamp_s: float | None = None
    task: str = 'move pot'

    def validate(self):
        if not self.observation_id or self.joint_names != JOINTS or self.joint_units != UNITS:
            raise ValueError('Missing observation identity or incompatible joint names/units')
        if set(self.images_rgb) != set(CAMERAS) or set(self.frame_ids) != set(CAMERAS):
            raise ValueError('Both named RGB cameras and frame identities are required')
        if set(self.camera_capture_ns) != set(CAMERAS):
            raise ValueError('Declare both camera capture times (None for unknown replay times)')
        for key in CAMERAS:
            image = self.images_rgb[key]
            if image.shape != (480, 640, 3) or image.dtype != np.uint8 or not self.frame_ids[key]:
                raise ValueError('Expected named 480x640 uint8 RGB with frame identity')
        if self.state.shape != (6,) or not np.isfinite(self.state).all():
            raise ValueError('Expected six finite state positions')
        for stamp in [*self.camera_capture_ns.values(), self.state_capture_ns]:
            if stamp is not None and (type(stamp) is not int or stamp < 0):
                raise ValueError('Capture times must be monotonic integer nanoseconds or None')
        if self.dataset_timestamp_s is not None and (not np.isfinite(self.dataset_timestamp_s) or self.dataset_timestamp_s < 0):
            raise ValueError('Invalid dataset timestamp')


@dataclass(frozen=True)
class MovePotChunk:
    actions: np.ndarray
    source: dict
    prediction_start_ns: int
    prediction_end_ns: int
    forward_ms: float
    processing_and_forward_ms: float


class MovePotPolicyAdapter(Protocol):
    """Implemented by ACTMovePotAdapter and SmolVLAMovePotAdapter; queue is model-independent."""
    action_names: tuple[str, ...]
    action_units: tuple[str, ...]
    predicted_steps: int
    execute_steps: int

    def reset(self) -> None: ...
    def predict_chunk(self, observation: MovePotObservation) -> MovePotChunk: ...


class ACTMovePotAdapter(ACTChunkAdapter):
    """Reuses the existing ACT reset, horizon and native-queue verification methods."""
    action_names = JOINTS
    action_units = UNITS

    def __init__(self, checkpoint: Path, device: str = 'cpu'):
        import torch
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies import make_pre_post_processors
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_dir():
            raise ValueError('A complete local checkpoint directory is required')
        self.torch, self.device = torch, device
        cfg = ACTConfig.from_pretrained(str(checkpoint), local_files_only=True)
        expected = {k: (3, 480, 640) for k in CAMERAS}
        expected['observation.state'] = (6,)
        if {k: tuple(v.shape) for k, v in cfg.input_features.items()} != expected:
            raise ValueError('Checkpoint must have exactly the move pot dual-camera/state features')
        if tuple(cfg.action_feature.shape) != (6,) or cfg.temporal_ensemble_coeff is not None:
            raise ValueError('Expected six-dimensional actions without temporal ensembling')
        if cfg.chunk_size != 50 or not 1 <= cfg.n_action_steps <= cfg.chunk_size:
            raise ValueError('R1 requires a 50-action predicted horizon and valid execution prefix')
        cfg.device, cfg.use_amp, cfg.pretrained_backbone_weights = device, False, None
        self.policy = ACTPolicy.from_pretrained(str(checkpoint), config=cfg, local_files_only=True, strict=True)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=str(checkpoint),
            preprocessor_overrides={'device_processor': {'device': device}},
            postprocessor_overrides={'device_processor': {'device': 'cpu'}})
        self.checkpoint_manifest = {f.name: file_sha256(f) for f in sorted(checkpoint.iterdir()) if f.is_file()}
        self.execute_steps, self.predicted_steps = cfg.n_action_steps, cfg.chunk_size
        self.reset()

    def _prepare(self, observation):
        observation.validate()
        t = self.torch
        batch = {k: t.from_numpy(observation.images_rgb[k].copy()).permute(2, 0, 1).float()/255 for k in CAMERAS}
        batch['observation.state'] = t.from_numpy(observation.state.copy()).float()
        return self.pre(batch)

    def predict_chunk(self, observation):
        self._sync()
        start = time.perf_counter_ns()
        with self.torch.inference_mode():
            batch = self._prepare(observation)
            self._sync(); forward_start = time.perf_counter_ns()
            normalized = self.policy.predict_action_chunk(batch)
            self._sync(); forward_end = time.perf_counter_ns()
            actions = self.post(normalized).detach().cpu().numpy()
        end = time.perf_counter_ns()
        if actions.shape != (1, self.predicted_steps, 6) or not np.isfinite(actions).all():
            raise ValueError('Invalid predicted action chunk')
        source = {'observation_id': observation.observation_id,
                  'frame_ids': dict(observation.frame_ids),
                  'camera_capture_ns': dict(observation.camera_capture_ns),
                  'state_capture_ns': observation.state_capture_ns,
                  'dataset_timestamp_s': observation.dataset_timestamp_s,
                  'rgb_sha256': {k: hashlib.sha256(observation.images_rgb[k].tobytes()).hexdigest() for k in CAMERAS},
                  'state': observation.state.tolist(), 'joint_names': list(JOINTS), 'joint_units': list(UNITS),
                  'model_sha256': self.checkpoint_manifest['model.safetensors']}
        return MovePotChunk(actions[0].copy(), source, start, end,
                            (forward_end-forward_start)/1e6, (end-start)/1e6)


class ReplayChunkQueue:
    """Offline candidate queue. Never represents validation as hardware dispatch.

    A caller can supply a validator to exercise gate/rejection logging. No default
    joint gate or physical send method is supplied. Rejection latches until reset.
    """
    def __init__(self, event_path: Path, trial_id: str):
        self.stream = Path(event_path).open('x')
        self.trial_id = trial_id
        self.pending = deque()
        self.failed = False
        self.next_chunk_id = 0

    def _record(self, event):
        # Preserve invalid numbers explicitly as strings in valid JSON rejection logs.
        def safe(value):
            if isinstance(value, np.ndarray):
                return safe(value.tolist())
            if isinstance(value, np.generic):
                return safe(value.item())
            if isinstance(value, float) and not np.isfinite(value):
                return {'nonfinite': repr(value)}
            if isinstance(value, dict):
                return {k: safe(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [safe(v) for v in value]
            return value
        self.stream.write(json.dumps(safe({'trial_id': self.trial_id, **event}), allow_nan=False)+'\n')
        self.stream.flush()

    def enqueue(self, chunk: MovePotChunk, execute_steps: int):
        if self.failed or self.pending:
            raise ValueError('Reset failed queue or consume pending prefix before replanning')
        if type(execute_steps) is not int or not 1 <= execute_steps <= len(chunk.actions):
            raise ValueError('Invalid executed horizon')
        if chunk.actions.ndim != 2 or chunk.actions.shape[1] != 6 or not np.isfinite(chunk.actions).all():
            self.failed = True
            self._record({'event': 'prediction_rejected', 'actions': chunk.actions,
                          'source': chunk.source, 'prediction_start_ns': chunk.prediction_start_ns,
                          'prediction_end_ns': chunk.prediction_end_ns,
                          'rejection_reasons': ['invalid_six_dimensional_finite_chunk'],
                          'hardware_dispatched': False, 'dispatch_monotonic_ns': None})
            raise ValueError('Invalid six-dimensional chunk')
        cid = self.next_chunk_id; self.next_chunk_id += 1
        # Persist full prediction and source before any candidate/gate can fail.
        event = {'event': 'prediction', 'chunk_id': cid, 'actions': chunk.actions.tolist(),
                 'execute_steps': execute_steps, 'source': chunk.source,
                 'prediction_start_ns': chunk.prediction_start_ns, 'prediction_end_ns': chunk.prediction_end_ns,
                 'forward_ms': chunk.forward_ms, 'processing_and_forward_ms': chunk.processing_and_forward_ms}
        self._record(event)
        source = json.loads(json.dumps(chunk.source))
        for offset, action in enumerate(chunk.actions[:execute_steps]):
            self.pending.append((cid, offset, action.copy(), source, chunk.prediction_start_ns, chunk.prediction_end_ns))

    def pop(self, validator: Callable | None = None, *, context_validator: Callable | None = None):
        if validator is not None and context_validator is not None:
            raise ValueError('Use one validator interface at a time')
        if self.failed:
            raise ValueError('Queue is latched after a rejected action; reset required')
        if not self.pending:
            raise IndexError('Queue empty; a new observation and prediction are required')
        cid, offset, action, source, start_ns, end_ns = self.pending.popleft()
        event = {'event': 'candidate', 'chunk_id': cid, 'chunk_offset': offset,
                 'action': action.tolist(), 'source': source,
                 'prediction_start_ns': start_ns, 'prediction_end_ns': end_ns, 'selected_monotonic_ns': time.perf_counter_ns(),
                 'dispatch_monotonic_ns': None, 'hardware_dispatched': False,
                 'gate_status': 'not_checked' if validator is None and context_validator is None else 'accepted'}
        try:
            if validator is not None:
                validator(action.copy())
            if context_validator is not None:
                context_validator(action.copy(), event)
        except Exception as exc:
            self.failed = True
            event.update(gate_status='rejected', error_type=type(exc).__name__, error=str(exc),
                         rejection_reasons=getattr(exc, 'reasons', [str(exc)]))
            self._record(event)
            raise
        self._record(event)
        return action

    def reset(self):
        self._record({'event': 'reset', 'discarded_actions': len(self.pending),
                      'discarded_candidates': [{'chunk_id': item[0], 'chunk_offset': item[1]} for item in self.pending]})
        self.pending.clear(); self.failed = False

    def close(self):
        self.stream.close()
