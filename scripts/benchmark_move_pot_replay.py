"""Bounded paired GPU replay; no cameras, motor connections, downloads or training."""
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
import time
import traceback
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(out):
    import numpy as np
    import torch
    from cross_backend.move_pot_policy import ACTMovePotAdapter, CAMERAS, MovePotObservation, ReplayChunkQueue, file_sha256
    from cross_backend.smolvla_move_pot import SmolVLAMovePotAdapter
    original = socket.socket.connect
    def offline(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError('Network disabled for benchmark')
        return original(sock, address)
    socket.socket.connect = offline
    torch.set_num_threads(2)
    assert torch.cuda.is_available(), 'CUDA required'
    fixture = ROOT / 'artifacts/r1-parallel-preparation-20260924/pilot-acceptance-cpu/sample-00000.npz'
    with np.load(fixture) as data:
        obs = MovePotObservation({k: data[k].copy() for k in CAMERAS}, data['state'].copy(),
            'dataset:0', {k: 'dataset:0:' + k for k in CAMERAS}, {k: None for k in CAMERAS}, None)
    result = {'input': str(fixture), 'input_sha256': file_sha256(fixture),
        'device': torch.cuda.get_device_name(), 'torch': torch.__version__, 'warmup_chunks': 2,
        'measured_chunks': 10, 'hardware_dispatches': 0, 'training_updates': 0,
        'capture_included': False, 'task_success_evaluated': False, 'models': {}}
    checkpoints = {
        'act': (ACTMovePotAdapter, ROOT / 'artifacts/r1-act-formal-40000-run02/train/checkpoints/040000/pretrained_model'),
        'smolvla': (SmolVLAMovePotAdapter, Path('/home/murphy/project/lerobot/outputs/train/smolvla_move_pot_dualcam_20260923_v3/checkpoints/040000/pretrained_model'))}
    for name, (factory, checkpoint) in checkpoints.items():
        print('Loading ' + name, flush=True)
        adapter = factory(checkpoint, 'cuda')
        manifest = adapter.checkpoint_manifest.copy()
        for _ in range(2):
            adapter.predict_chunk(obs)
        adapter.reset()
        torch.cuda.reset_peak_memory_stats()
        queue = ReplayChunkQueue(out / (name + '-events.jsonl'), name + '_benchmark')
        rows, actions = [], []
        try:
            for i in range(10):
                torch.cuda.synchronize()
                start = time.perf_counter_ns()
                chunk = adapter.predict_chunk(obs)
                queue.enqueue(chunk, 50)
                replay = np.stack([queue.pop() for _ in range(50)])
                np.testing.assert_array_equal(replay, chunk.actions)
                elapsed = (time.perf_counter_ns() - start) / 1e6
                actions.append(chunk.actions)
                rows.append({'index': i, 'forward_ms': chunk.forward_ms,
                    'processing_and_forward_ms': chunk.processing_and_forward_ms,
                    'predict_queue_logging_verify_ms': elapsed})
        finally:
            queue.close()
        assert all(file_sha256(checkpoint / f) == digest for f, digest in manifest.items())
        values = np.stack(actions)
        np.savez_compressed(out / (name + '-actions.npz'), actions=values)
        stats = {key: {'median': float(np.median([r[key] for r in rows])),
                       'p95': float(np.percentile([r[key] for r in rows], 95)),
                       'max': float(max(r[key] for r in rows))} for key in rows[0] if key != 'index'}
        result['models'][name] = {'checkpoint': str(checkpoint), 'manifest': manifest,
            'rows': rows, 'milliseconds': stats, 'peak_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
            'gripper_out_of_range_count': int(((values[..., -1] < 0) | (values[..., -1] > 100)).sum()),
            'finite': bool(np.isfinite(values).all()), 'checkpoint_unchanged': True}
        (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
        print(name + ': ' + json.dumps(stats), flush=True)
        del adapter, chunk
        gc.collect()
        torch.cuda.empty_cache()
    result['status'] = 'passed'
    (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'executed_script.py').write_bytes(Path(__file__).read_bytes())
    try:
        main(args.output)
    except BaseException:
        (args.output / 'failure.txt').write_text(traceback.format_exc())
        raise
