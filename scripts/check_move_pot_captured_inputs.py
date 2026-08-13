"""Replay previously captured observations through both models, without hardware access."""
import os
for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    os.environ[key] = '1'
os.environ['WANDB_MODE'] = 'disabled'
import argparse
import gc
import json
from pathlib import Path
import socket
import sys
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(capture, out):
    import numpy as np
    import torch
    from cross_backend.move_pot_policy import ACTMovePotAdapter, MovePotObservation, ReplayChunkQueue, CAMERAS, file_sha256
    from cross_backend.smolvla_move_pot import SmolVLAMovePotAdapter
    original = socket.socket.connect
    def offline(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError('Network disabled')
        return original(sock, address)
    socket.socket.connect = offline
    torch.set_num_threads(2)
    meta = json.loads((capture / 'summary.json').read_text())
    assert meta['status'] == 'passed' and len(meta['samples']) == 3
    manifests = {str(p): file_sha256(p) for p in capture.iterdir() if p.is_file()}
    result = {'status': 'running', 'capture_manifest': manifests, 'models': {}, 'hardware_dispatches': 0,
              'task_success_evaluated': False, 'physical_joint_gate_validated': False,
              'capture_times': 'unknown exposure times; host read intervals retained separately'}
    specs = [('act', ACTMovePotAdapter, ROOT / 'artifacts/r1-act-formal-40000-run02/train/checkpoints/040000/pretrained_model'),
             ('smolvla', SmolVLAMovePotAdapter, Path('/home/murphy/project/lerobot/outputs/train/smolvla_move_pot_dualcam_20260923_v3/checkpoints/040000/pretrained_model'))]
    for name, factory, cp in specs:
        print('Loading ' + name, flush=True)
        adapter = factory(cp, 'cuda')
        q = ReplayChunkQueue(out / (name + '-events.jsonl'), name + '_captured_replay')
        rows = []
        try:
            for sample in meta['samples']:
                index = sample['index']; path = capture / f'sample-{index:02d}.npz'
                with np.load(path) as data:
                    obs = MovePotObservation({k: data[k].copy() for k in CAMERAS}, data['state'].copy(),
                        f'{capture.name}:{index}', {k: f'{capture.name}:{index}:{k}' for k in CAMERAS},
                        {k: None for k in CAMERAS}, None)
                adapter.reset()
                chunk = adapter.predict_chunk(obs)
                chunk.source['host_read_intervals'] = sample
                chunk.source['observation_npz_sha256'] = file_sha256(path)
                q.enqueue(chunk, 50)
                np.testing.assert_array_equal(np.stack([q.pop() for _ in range(50)]), chunk.actions)
                q.reset()
                np.savez_compressed(out / f'{name}-{index:02d}-actions.npz', actions=chunk.actions)
                rows.append({'index': index, 'forward_ms': chunk.forward_ms,
                    'processing_and_forward_ms': chunk.processing_and_forward_ms,
                    'gripper_min': float(chunk.actions[:, -1].min()), 'gripper_max': float(chunk.actions[:, -1].max()),
                    'gripper_out_of_range_offsets': np.flatnonzero((chunk.actions[:, -1] < 0) | (chunk.actions[:, -1] > 100)).tolist(),
                    'first_action': chunk.actions[0].tolist(),
                    'first_action_minus_state': (chunk.actions[0] - obs.state).tolist(),
                    'max_abs_first_action_body_delta_deg': float(np.max(np.abs(chunk.actions[0, :5] - obs.state[:5]))),
                    'action_min': chunk.actions.min(axis=0).tolist(), 'action_max': chunk.actions.max(axis=0).tolist()})
        finally:
            q.close()
        assert all(file_sha256(cp / f) == h for f, h in adapter.checkpoint_manifest.items())
        result['models'][name] = {'checkpoint_manifest': adapter.checkpoint_manifest, 'samples': rows,
                                  'checkpoint_unchanged': True, 'finite_and_queue_checks': 'passed'}
        (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(name + ': ' + json.dumps(rows), flush=True)
        del adapter, chunk
        gc.collect(); torch.cuda.empty_cache()
    assert all(file_sha256(p) == h for p, h in manifests.items())
    result['status'] = 'passed'
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'executed_script.py').write_bytes(Path(__file__).read_bytes())
    try:
        main(args.capture, args.output)
    except BaseException:
        (args.output / 'failure.txt').write_text(traceback.format_exc()); raise
