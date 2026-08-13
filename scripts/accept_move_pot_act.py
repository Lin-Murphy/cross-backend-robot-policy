"""Offline completed-run acceptance. Defaults to CPU; never trains or dispatches.

Use --check-readiness during training. Real acceptance requires a successful run
exit and exact checkpoint step, never a rounded '40K' console label.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    os.environ[key] = '1'
os.environ['WANDB_MODE'] = 'disabled'


def readiness(run, steps):
    status = run/'process-result.json'
    checkpoint = run/f'train/checkpoints/{steps:06d}/pretrained_model'
    step_file = checkpoint.parent/'training_state/training_step.json'
    reasons = []
    if not status.exists():
        reasons.append('Training process has not recorded completion')
    elif json.loads(status.read_text())['exit_code'] != 0:
        reasons.append('Training exited unsuccessfully')
    if not step_file.exists() or json.loads(step_file.read_text())['step'] != steps:
        reasons.append('Exact requested checkpoint step is missing')
    if not (checkpoint/'model.safetensors').exists():
        reasons.append('Model weights missing')
    return {'ready': not reasons, 'reasons': reasons, 'expected_steps': steps, 'checkpoint': str(checkpoint)}


def main(args):
    ready = readiness(args.run_dir, args.expected_steps)
    if args.check_readiness:
        print(json.dumps(ready, indent=2)); return
    if not ready['ready']:
        raise RuntimeError(json.dumps(ready))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    def save(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    try:
        import draccus
        import numpy as np
        import torch
        from safetensors import safe_open
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.datasets.factory import make_dataset
        from lerobot.policies.act.processor_act import make_act_pre_post_processors
        from cross_backend.move_pot_policy import (ACTMovePotAdapter, MovePotObservation, ReplayChunkQueue,
                                                   CAMERAS, JOINTS, UNITS, file_sha256)
        torch.set_num_threads(2)
        def deny(*a, **kw):
            raise RuntimeError('Acceptance forbids downloads and network access')
        torch.hub.download_url_to_file = deny
        original_connect = socket.socket.connect
        def offline_connect(sock, address):
            if sock.family in (socket.AF_INET, socket.AF_INET6): deny()
            return original_connect(sock, address)
        socket.socket.connect = offline_connect
        cp = Path(ready['checkpoint'])
        cfg = draccus.decode(TrainPipelineConfig, json.loads((cp/'train_config.json').read_text()))
        assert cfg.steps == args.expected_steps and cfg.batch_size == 8 and cfg.seed == 1000
        assert cfg.dataset.episodes == list(range(30)) and not cfg.resume
        assert cfg.dataset.repo_id == 'local/move_pot_20260923_142914'
        assert cfg.policy.chunk_size == cfg.policy.n_action_steps == 50
        run_hashes = {str(f): file_sha256(f) for f in args.run_dir.rglob('*') if f.is_file() and not f.is_symlink()}
        save('run_hashes_before.json', run_hashes)
        manifest = args.run_dir/'train-files.sha256'
        assert manifest.exists(), 'Launcher checksum manifest missing'
        for line in manifest.read_text().splitlines():
            digest, path = line.split('  ', 1)
            assert file_sha256(path) == digest, path
        checkpoint_checks = []
        for step_file in sorted((args.run_dir/'train/checkpoints').glob('*/training_state/training_step.json')):
            if step_file.parent.parent.is_symlink(): continue
            folder = step_file.parent.parent
            assert int(folder.name) == json.loads(step_file.read_text())['step']
            counts = {}
            for rel in ('pretrained_model/model.safetensors', 'training_state/optimizer_state.safetensors'):
                with safe_open(folder/rel, framework='pt', device='cpu') as data:
                    for key in data.keys():
                        assert torch.isfinite(data.get_tensor(key)).all(), (folder, key)
                    counts[rel] = len(list(data.keys()))
            for rel in ('training_state/rng_state.safetensors', 'training_state/optimizer_param_groups.json'):
                assert (folder/rel).stat().st_size > 0
            checkpoint_checks.append({'step': int(folder.name), 'finite_tensors': counts})
        expected_saved = sorted(set(range(cfg.save_freq, cfg.steps+1, cfg.save_freq)) | {cfg.steps})
        assert [v['step'] for v in checkpoint_checks] == expected_saved
        # Rounded log steps such as 20K are deliberately excluded from completion evidence.
        windows = []
        for line in (args.run_dir/'train.log').read_text().splitlines():
            if not all(token in line for token in ('loss:', 'updt_s:', 'data_s:', 'grdn:')): continue
            values = {k: float(v) for k, v in re.findall(r'(?<!\S)(loss|grdn|lr|updt_s|data_s|mem_gb|l1_loss|kld_loss):([^\s]+)', line)}
            assert all(math.isfinite(v) for v in values.values())
            windows.append(values)
        assert len(windows) == args.expected_steps//cfg.log_freq
        assert 'End of training' in (args.run_dir/'train.log').read_text()
        save('training_windows.json', windows)
        adapter = ACTMovePotAdapter(cp, args.device)
        # Strict loading plus exact equality proves that no randomly initialized tensors remain.
        with safe_open(cp/'model.safetensors', framework='pt', device='cpu') as saved:
            for key, value in adapter.policy.state_dict().items():
                torch.testing.assert_close(value.cpu(), saved.get_tensor(key), rtol=0, atol=0)
        cfg.dataset.image_transforms.enable = False
        ds = make_dataset(cfg)
        assert tuple(ds.features['action']['names']) == tuple(ds.features['observation.state']['names']) == JOINTS
        fresh_pre, fresh_post = make_act_pre_post_processors(adapter.policy.config, ds.meta.stats)
        records = []
        queue = ReplayChunkQueue(args.output/'events.jsonl', 'offline_acceptance')
        for index in (0, 5620, 10768):
            item = ds[index]
            images = {key: item[key].permute(1, 2, 0).numpy() for key in CAMERAS}
            observation = MovePotObservation(images, item['observation.state'].numpy(),
                f'dataset:{ds.repo_id}:{index}', {key: f'{index}:{key}' for key in CAMERAS},
                {key: None for key in CAMERAS}, None, dataset_timestamp_s=float(item['timestamp']))
            batch = adapter._prepare(observation)
            raw = {key: item[key].float()/255 for key in CAMERAS}
            raw['observation.state'] = item['observation.state']
            expected = fresh_pre(raw)
            for key in raw: torch.testing.assert_close(batch[key], expected[key], rtol=0, atol=0)
            chunk = adapter.predict_chunk(observation)
            with torch.inference_mode():
                direct = fresh_post(adapter.policy.predict_action_chunk(batch)).cpu().numpy()[0]
            np.testing.assert_allclose(chunk.actions, direct, rtol=0, atol=1e-5)
            native = adapter.verify_native_queue(observation)
            with torch.inference_mode():
                first = adapter.post(adapter.policy.select_action(batch)).cpu().numpy()[0]
                assert len(adapter.policy._action_queue) == 49
                adapter.reset()
                assert len(adapter.policy._action_queue) == 0
                np.testing.assert_allclose(adapter.post(adapter.policy.select_action(batch)).cpu().numpy()[0], first, atol=1e-5, rtol=0)
                adapter.reset()
            queue.enqueue(chunk, adapter.execute_steps)
            replay = np.stack([queue.pop() for _ in range(adapter.execute_steps)])
            np.testing.assert_array_equal(replay, chunk.actions)
            queue.reset()
            np.savez_compressed(args.output/f'sample-{index:05d}.npz', **{k: v for k, v in images.items()},
                                state=observation.state, actions=chunk.actions)
            bad = np.flatnonzero((chunk.actions[:, -1] < 0) | (chunk.actions[:, -1] > 100)).tolist()
            records.append({'index': index, 'native_queue': native,
                            'gripper_out_of_range_offsets': bad, 'source': chunk.source})
        queue.close()
        with torch.inference_mode():
            for _ in range(2): adapter.policy.predict_action_chunk(batch)
            adapter._sync()
            if args.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
            times = []
            for _ in range(5):
                start = time.perf_counter_ns(); adapter.policy.predict_action_chunk(batch); adapter._sync()
                times.append((time.perf_counter_ns()-start)/1e6)
        all_good = not any(r['gripper_out_of_range_offsets'] for r in records)
        assert all(file_sha256(f) == h for f, h in run_hashes.items())
        save('summary.json', {'status': 'completed' if all_good else 'completed_with_action_findings',
            'offline_pipeline_passed': True, 'task_or_hardware_approved': False,
            'actual_steps': args.expected_steps, 'saved_checkpoints': checkpoint_checks,
            'checkpoint_identity': adapter.checkpoint_manifest, 'samples': records,
            'gripper_nominal_range_check': 'passed_on_samples' if all_good else 'failed_on_samples',
            'other_joint_hardware_limits': 'not_established', 'timing_device': args.device,
            'forward_times_ms': times, 'forward_median_ms': float(np.median(times)),
            'replay_peak_allocated_mib': torch.cuda.max_memory_allocated()/2**20 if args.device.startswith('cuda') else None,
            'training_peak_allocated_gib_rounded': max(v['mem_gb'] for v in windows),
            'timing_scope': 'five batch1 forwards after two warmups; not complete control latency',
            'training_outputs_unchanged': True, 'training_updates': 0, 'hardware_dispatches': 0})
        print('Completed offline acceptance:', args.output)
    except BaseException:
        (args.output/'failure.txt').write_text(traceback.format_exc())
        save('status.json', {'status': 'failed', 'hardware_dispatches': 0})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--expected-steps', type=int, default=40000)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--check-readiness', action='store_true')
    args = parser.parse_args()
    if not args.check_readiness and args.output is None: parser.error('--output required for acceptance')
    args.run_dir = args.run_dir.resolve()
    main(args)
