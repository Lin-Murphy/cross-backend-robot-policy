"""Real camera/state -> ACT -> candidate safety gate, with motor writes forbidden."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
import subprocess

import cv2
import numpy as np
import torch
from cross_backend.safety import SO101CommandGate, NoDispatch
from cross_backend.local_metadata import load_policy_metadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch-threads', type=int, default=12)
    parser.add_argument('--execute-gripper-one-shot', action='store_true')
    parser.add_argument('--gripper-max-delta', type=float, default=.5)
    args = parser.parse_args()
    if args.torch_threads not in (1, 2, 4, 6, 8, 12):
        parser.error('unsupported thread count')
    if not .5 <= args.gripper_max_delta <= 3.0:
        parser.error('gripper-max-delta must be between 0.5 and 3.0')
    torch.set_num_threads(args.torch_threads)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path('artifacts/safety') / f'live-act-{stamp}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {'mode': 'live_observation_act_no_dispatch', 'status': 'failed',
              'dispatch_enabled': False, 'steps': [], 'hardware_connected': False,
              'torch_threads': args.torch_threads}
    try:
        root = Path(r'C:\Users\Murphy\.cache\huggingface\lerobot\Murphy-Lin\lerobot101_dataset_a')
        checkpoint = Path(r'D:\project\lerobot-seeed\outputs\train\act_test\checkpoints\000020\pretrained_model')
        path = Path(r'C:\Users\Murphy\.cache\huggingface\lerobot\calibration\robots\so_follower\so101_follower_arm.json')
        raw = path.read_bytes()
        calibration = json.loads(raw)
        report.update(checkpoint=str(checkpoint), calibration_sha256=hashlib.sha256(raw).hexdigest())
        names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper']
        meta = load_policy_metadata(root)
        for feature in ('observation.state', 'action'):
            if meta.features[feature]['names'] != [n+'.pos' for n in names]:
                raise RuntimeError('Dataset joint ordering mismatch')
        config = ACTConfig.from_pretrained(checkpoint)
        config.pretrained_path, config.device = checkpoint, 'cpu'
        policy = make_policy(config, ds_meta=meta).eval()
        pre, post = make_pre_post_processors(config, pretrained_path=str(checkpoint), dataset_stats=meta.stats)
        report['hardware_reader_environment'] = 'official_lerobot_0.6.1'
        lower, upper = [-100]*5+[0], [100]*6
        report['candidate_limits'] = {'lower': lower, 'upper': upper, 'speed_normalized_per_s': [2]*6,
                                      'scale': .1, 'hypothetical_dt_s': .05}
        capture_dir = output.with_suffix('')
        worker = subprocess.Popen([r'D:\project\lerobot\.venv\Scripts\python.exe',
            '-u', 'scripts/capture_live_observation.py', '--serve', '--output-dir', str(capture_dir)]
            + (['--allow-gripper-one-shot'] if args.execute_gripper_one_shot else []),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        ready = json.loads(worker.stdout.readline())
        if ready.get('status') != 'ready':
            raise RuntimeError(f'Capture worker failed to start: {ready}')
        for index in range(3):
            t0 = time.perf_counter()
            worker.stdin.write(json.dumps({'command':'capture'})+'\n')
            worker.stdin.flush()
            response = json.loads(worker.stdout.readline())
            if response.get('status') != 'captured':
                raise RuntimeError(f'Capture failed: {response}')
            capture_path = Path(response['path'])
            report['hardware_connected'] = True
            with np.load(capture_path) as captured:
                bgr, state = captured['bgr'], captured['state']
                raw_state = dict(zip(names, captured['raw'].tolist()))
                image_time, state_time = float(captured['image_time']), float(captured['state_time'])
                report['last_observed_torque'] = captured['torque'].tolist()
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            batch = pre({'observation.state': torch.from_numpy(state).unsqueeze(0),
                         'observation.images.front': torch.from_numpy(rgb).permute(2,0,1).float().div(255).unsqueeze(0)})
            infer_start = time.perf_counter()
            with torch.inference_mode():
                chunk = post(policy.predict_action_chunk(batch))
            infer_end = time.perf_counter()
            action = chunk[0,0].cpu().numpy()
            record = {'index': index, 'raw_state': raw_state, 'state': state.tolist(),
                      'action': action.tolist(), 'chunk_shape': list(chunk.shape),
                      'inference_ms': (infer_end-infer_start)*1000,
                      'state_age_at_output_ms': (infer_end-state_time)*1000,
                      'capture_process_to_image_ms': (image_time-t0)*1000,
                      'image_state_read_skew_ms': (state_time-image_time)*1000}
            report['steps'].append(record)
            gate = SO101CommandGate(lower, upper, [2]*6, scale=.1)
            try:
                candidate = gate.prepare(action, state, state, .05)
                record['candidate_action'] = candidate.tolist()
                record['candidate_gate'] = 'passed'
                try:
                    NoDispatch().send(candidate)
                except RuntimeError as exc:
                    record['dispatch_blocked'] = str(exc)
            except ValueError as exc:
                record['candidate_gate'] = 'rejected'
                record['gate_error'] = str(exc)
            record['freshness_gate_100ms'] = infer_end-state_time <= .1
            if index == 0:
                if not cv2.imwrite(str(output.with_suffix('.png')), bgr):
                    raise RuntimeError('Could not save camera evidence')
        report['status'] = 'inference_completed'
        report['software_checks_passed'] = all(s['candidate_gate']=='passed' and s['freshness_gate_100ms'] for s in report['steps'])
        report['ready_for_dispatch'] = False
        if args.execute_gripper_one_shot:
            selected = report['steps'][-1]
            report['mode'] = 'live_act_gripper_one_shot'
            if selected['candidate_gate'] != 'passed' or selected['state_age_at_output_ms'] > 300:
                raise RuntimeError('One-shot timing or candidate gate failed')
            one_shot_gripper = float(selected['state'][5] + np.clip(
                .1 * (selected['action'][5] - selected['state'][5]),
                -args.gripper_max_delta, args.gripper_max_delta))
            report['one_shot_candidate_gripper'] = one_shot_gripper
            report['gripper_max_delta'] = args.gripper_max_delta
            worker.stdin.write(json.dumps({'command':'execute_gripper',
                'expected_state':selected['state'],
                'candidate_gripper':one_shot_gripper,
                'max_delta':args.gripper_max_delta,
                'max_raw_delta':45 if args.gripper_max_delta > .5 else 8,
                'max_raw_excursion':55 if args.gripper_max_delta > .5 else 12})+'\n')
            worker.stdin.flush()
            execution = json.loads(worker.stdout.readline())
            report['execution'] = execution
            if execution.get('status') != 'executed':
                raise RuntimeError(f'One-shot execution failed: {execution}')
            report['status'] = 'passed'
        worker.stdin.write(json.dumps({'command':'stop'})+'\n')
        worker.stdin.flush()
        worker.wait(timeout=10)
        report['capture_worker_exit_code'] = worker.returncode
    except BaseException:
        report['status'] = 'failed'
        report['error_trace'] = traceback.format_exc()
    finally:
        if 'worker' in locals() and worker.poll() is None:
            worker.terminate()
            worker.wait(timeout=10)
        if 'worker' in locals():
            stderr = worker.stderr.read()
            if stderr:
                report['capture_worker_stderr'] = stderr
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(output, report['status'])
    return 0 if report['status'] in ('inference_completed', 'passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
