"""SmolVLA implementation of the named move-pot adapter contract; no hardware."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .move_pot_policy import CAMERAS, JOINTS, UNITS, MovePotChunk, file_sha256


class SmolVLAMovePotAdapter:
    action_names = JOINTS
    action_units = UNITS

    @staticmethod
    def validate_profile(cfg, processor):
        image_keys = set(cfg.image_features)
        expected = {'observation.images.camera1', 'observation.images.camera2',
                    'observation.images.camera3', 'observation.images.empty_camera_0',
                    'observation.images.empty_camera_1'}
        if image_keys != expected or tuple(cfg.input_features['observation.state'].shape) != (6,):
            raise ValueError('Unexpected SmolVLA dual-camera checkpoint profile')
        if tuple(cfg.action_feature.shape) != (6,) or cfg.chunk_size != 50 or cfg.n_obs_steps != 1:
            raise ValueError('Expected six-dimensional, single-observation, 50-step SmolVLA')
        if not 1 <= cfg.n_action_steps <= cfg.chunk_size:
            raise ValueError('Execution prefix exceeds predicted horizon')
        for key, feature in cfg.image_features.items():
            expected_shape = (3, 480, 640) if 'empty_camera' in key else (3, 256, 256)
            if tuple(feature.shape) != expected_shape:
                raise ValueError('Unexpected saved image feature shape')
        if cfg.empty_cameras != 2 or cfg.adapt_to_pi_aloha or cfg.use_delta_joint_actions_aloha:
            raise ValueError('Incompatible camera padding or action-unit conversion')
        if cfg.rtc_config is not None and cfg.rtc_config.enabled:
            raise ValueError('This synchronous adapter does not support RTC')
        renames = [s['config'].get('rename_map', {}) for s in processor['steps']
                   if s.get('registry_name') == 'rename_observations_processor']
        if renames != [{CAMERAS[0]: 'observation.images.camera1'}]:
            raise ValueError('Saved follower-to-camera1 mapping must match the trained checkpoint')

    def __init__(self, checkpoint: Path, device='cpu', seed=1000):
        import torch
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from lerobot.policies import make_pre_post_processors
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_dir(): raise ValueError('Local checkpoint required')
        cfg = SmolVLAConfig.from_pretrained(str(checkpoint), local_files_only=True)
        self.validate_profile(cfg, json.loads((checkpoint/'policy_preprocessor.json').read_text()))
        if type(seed) is not int or seed < 0: raise ValueError('Nonnegative integer seed required')
        self.torch, self.device, self.seed = torch, device, seed
        cfg.device, cfg.use_amp, cfg.compile_model = device, False, False
        # Preserve checkpoint construction precision (cached VLM initialization), then strictly
        # replace all model state from the task checkpoint. Caller must enforce HF offline mode.
        self.policy = SmolVLAPolicy.from_pretrained(str(checkpoint), config=cfg,
                                                   local_files_only=True, strict=True).eval()
        self.pre, self.post = make_pre_post_processors(cfg, pretrained_path=str(checkpoint),
            preprocessor_overrides={'device_processor': {'device': device}},
            postprocessor_overrides={'device_processor': {'device': 'cpu'}})
        self.checkpoint_manifest = {f.name: file_sha256(f) for f in sorted(checkpoint.iterdir()) if f.is_file()}
        self.predicted_steps, self.execute_steps = cfg.chunk_size, cfg.n_action_steps
        self.generator = torch.Generator(device=device)
        self.reset()

    def reset(self):
        self.policy.reset(); self.pre.reset(); self.post.reset()
        self.generator.manual_seed(self.seed)
        self.sampling_index = 0

    def set_execution_steps(self, steps):
        if type(steps) is not int or not 1 <= steps <= self.predicted_steps:
            raise ValueError('Executed horizon must be an integer within predicted chunk')
        self.execute_steps = steps; self.policy.config.n_action_steps = steps; self.reset()

    def _sync(self):
        if self.device.startswith('cuda'): self.torch.cuda.synchronize()

    def _prepare(self, observation):
        observation.validate()
        if not isinstance(observation.task, str) or not observation.task.strip():
            raise ValueError('SmolVLA requires a nonempty task text')
        t = self.torch
        raw = {k: t.from_numpy(observation.images_rgb[k].copy()).permute(2, 0, 1).float()/255 for k in CAMERAS}
        raw.update({'observation.state': t.from_numpy(observation.state.copy()).float(), 'task': observation.task})
        batch = self.pre(raw)
        actual = {k for k in self.policy.config.image_features if k in batch}
        if actual != {'observation.images.camera1', 'observation.images.camera2'}:
            raise ValueError('Saved preprocessing did not produce the two physical camera inputs')
        return batch

    def _noise(self):
        return self.torch.randn((1, self.predicted_steps, self.policy.config.max_action_dim),
                                generator=self.generator, device=self.device, dtype=self.torch.float32)

    def predict_chunk(self, observation):
        self._sync(); start = time.perf_counter_ns()
        with self.torch.inference_mode():
            batch = self._prepare(observation)
            noise = self._noise()
            noise_hash = hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
            self._sync(); forward_start = time.perf_counter_ns()
            normalized = self.policy.predict_action_chunk(batch, noise=noise)
            self._sync(); forward_end = time.perf_counter_ns()
            actions = self.post(normalized).detach().cpu().numpy()
        end = time.perf_counter_ns()
        if actions.shape != (1, self.predicted_steps, 6) or not np.isfinite(actions).all():
            raise ValueError('Invalid SmolVLA action chunk')
        source = {'observation_id': observation.observation_id, 'frame_ids': dict(observation.frame_ids),
                  'camera_capture_ns': dict(observation.camera_capture_ns),
                  'state_capture_ns': observation.state_capture_ns, 'dataset_timestamp_s': observation.dataset_timestamp_s,
                  'rgb_sha256': {k: hashlib.sha256(observation.images_rgb[k].tobytes()).hexdigest() for k in CAMERAS},
                  'state': observation.state.tolist(), 'joint_names': list(JOINTS), 'joint_units': list(UNITS),
                  'task': observation.task, 'model_sha256': self.checkpoint_manifest['model.safetensors'],
                  'sampling_seed': self.seed, 'sampling_index': self.sampling_index, 'noise_sha256': noise_hash,
                  'num_denoising_steps': self.policy.config.num_steps}
        self.sampling_index += 1
        return MovePotChunk(actions[0].copy(), source, start, end,
                            (forward_end-forward_start)/1e6, (end-start)/1e6)

    def verify_native_queue(self, observation):
        self.reset()
        try:
            chunk = self.predict_chunk(observation)
            self.reset()
            with self.torch.inference_mode():
                batch, noise = self._prepare(observation), self._noise()
                native = np.stack([self.post(self.policy.select_action(batch, noise=noise)).cpu().numpy()[0]
                                   for _ in range(self.execute_steps)])
            error = float(np.max(np.abs(native-chunk.actions[:self.execute_steps])))
            if error > 1e-4: raise AssertionError(f'SmolVLA native queue differs by {error}')
            if len(self.policy._queues['action']) != 0: raise AssertionError('Native queue did not drain')
            return {'execute_steps': self.execute_steps, 'max_abs_error': error,
                    'noise_control': 'same explicit seeded noise tensor', 'hardware_dispatches': 0}
        finally:
            self.reset()
