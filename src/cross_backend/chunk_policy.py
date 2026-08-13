"""Minimal chunk-policy boundary for the simulation pilot.

The runner owns action scheduling; adapters own model-specific processing.
Only ACT is implemented and validated in this slice.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import time
import numpy as np


@dataclass(frozen=True)
class PolicyObservation:
    image_rgb: np.ndarray
    state: np.ndarray
    step: int
    simulation_time_s: float
    captured_monotonic_ns: int


@dataclass(frozen=True)
class PredictedChunk:
    actions: np.ndarray
    source_step: int
    source_simulation_time_s: float
    source_captured_monotonic_ns: int
    prediction_start_ns: int
    prediction_end_ns: int
    forward_ms: float
    processing_and_forward_ms: float


class PolicyAdapter(Protocol):
    action_names: tuple[str, ...]
    action_units: str
    execute_steps: int

    def reset(self) -> None: ...
    def predict_chunk(self, observation: PolicyObservation) -> PredictedChunk: ...


class ACTChunkAdapter:
    action_names = ('agent_x', 'agent_y')
    action_units = 'absolute simulation coordinates, [0,512]'

    def __init__(self, checkpoint: Path, device: str):
        import torch
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.policies.act.modeling_act import ACTPolicy
        from lerobot.policies import make_pre_post_processors

        self.torch = torch
        self.device = device
        cfg = ACTConfig.from_pretrained(str(checkpoint), local_files_only=True)
        cfg.device = device
        # A complete checkpoint supplies the backbone; never fetch initialization weights.
        cfg.pretrained_backbone_weights = None
        cfg.use_amp = False
        expected = {'observation.image': (3, 96, 96), 'observation.state': (2,)}
        if {k: tuple(v.shape) for k, v in cfg.input_features.items()} != expected:
            raise ValueError('ACT input features do not match this PushT configuration')
        if tuple(cfg.action_feature.shape) != (2,) or cfg.temporal_ensemble_coeff is not None:
            raise ValueError('This pilot requires 2D actions and no temporal ensembling')
        self.policy = ACTPolicy.from_pretrained(str(checkpoint), config=cfg, local_files_only=True, strict=True)
        self.policy.eval()
        self.pre, self.post = make_pre_post_processors(
            cfg, pretrained_path=str(checkpoint),
            preprocessor_overrides={'device_processor': {'device': device}},
            postprocessor_overrides={'device_processor': {'device': 'cpu'}},
        )
        self.execute_steps = cfg.n_action_steps
        self.predicted_steps = cfg.chunk_size
        self.reset()

    def set_execution_steps(self, steps: int) -> None:
        if type(steps) is not int or not 1 <= steps <= self.predicted_steps:
            raise ValueError('Executed horizon must be an integer within the predicted chunk')
        self.execute_steps = steps
        self.policy.config.n_action_steps = steps
        self.reset()

    def reset(self):
        self.policy.reset()
        self.pre.reset()
        self.post.reset()

    def _sync(self):
        if self.device.startswith('cuda'):
            self.torch.cuda.synchronize()

    def _prepare(self, observation):
        t = self.torch
        if observation.image_rgb.shape != (96, 96, 3) or observation.image_rgb.dtype != np.uint8:
            raise ValueError('Expected unnormalized 96x96 uint8 RGB')
        if observation.state.shape != (2,) or not np.isfinite(observation.state).all():
            raise ValueError('Expected finite 2D state')
        return self.pre({
            'observation.image': t.from_numpy(observation.image_rgb.copy()).permute(2, 0, 1).float() / 255,
            'observation.state': t.from_numpy(observation.state.copy()).float(),
        })

    def predict_chunk(self, observation):
        self._sync()
        start = time.perf_counter_ns()
        with self.torch.inference_mode():
            batch = self._prepare(observation)
            self._sync()
            forward_start = time.perf_counter_ns()
            normalized = self.policy.predict_action_chunk(batch)
            self._sync()
            forward_end = time.perf_counter_ns()
            actions = self.post(normalized).detach().cpu().numpy()
        end = time.perf_counter_ns()
        if actions.shape != (1, self.predicted_steps, 2) or not np.isfinite(actions).all():
            raise ValueError(f'Invalid ACT action chunk: {actions.shape}')
        return PredictedChunk(
            actions[0].copy(), observation.step, observation.simulation_time_s,
            observation.captured_monotonic_ns, start, end,
            (forward_end-forward_start)/1e6, (end-start)/1e6,
        )

    def verify_native_queue(self, observation):
        """Same recorded input: external chunk scheduling must match official select_action.
        Also warms model kernels without stepping the environment. Reset before/after.
        """
        self.reset()
        chunk = self.predict_chunk(observation)
        self.reset()
        with self.torch.inference_mode():
            batch = self._prepare(observation)
            native = np.stack([
                self.post(self.policy.select_action(batch)).detach().cpu().numpy()[0]
                for _ in range(self.execute_steps)
            ])
        error = float(np.max(np.abs(native - chunk.actions[:self.execute_steps])))
        self.reset()
        if error > 1e-4:
            raise AssertionError(f'Native queue differs by {error}')
        return {'execute_steps': self.execute_steps, 'max_abs_error': error,
                'environment_steps': 0, 'reset_before_and_after': True,
                'warmup_full_chunk_forward_calls': 2}
