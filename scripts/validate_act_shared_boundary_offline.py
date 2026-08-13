"""Run local ACT on a saved SO101 frame through the backend-neutral fake boundary."""
import os
for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    os.environ[key] = '1'
os.environ['WANDB_MODE'] = 'disabled'

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import socket
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(sample, checkpoint, output, device):
    import numpy as np
    import torch
    from cross_backend.execution_contract import ActionRequest, action_request_from_policy, execute_from_observation
    from cross_backend.offline_backend_adapters import SavedSO101FakeBackendAdapter
    from cross_backend.move_pot_policy import ACTMovePotAdapter, MovePotObservation, CAMERAS, file_sha256

    original_connect = socket.socket.connect
    def offline(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError('Network disabled for offline validation')
        return original_connect(sock, address)
    socket.socket.connect = offline
    torch.set_num_threads(2)
    source_hash = file_sha256(sample)
    backend = SavedSO101FakeBackendAdapter(sample)
    observation = backend.observe().validate(backend.capabilities)
    policy_observation = MovePotObservation(
        {'observation.images.' + name: observation.images_rgb[name] for name in backend.capabilities.cameras},
        np.asarray(observation.joint_position, dtype=np.float32), observation.observation_id,
        {'observation.images.' + name: observation.camera_frame_ids[name] for name in backend.capabilities.cameras},
        {'observation.images.' + name: observation.camera_capture_ns[name] for name in backend.capabilities.cameras},
        observation.state_capture_ns, task='move pot')
    adapter = ACTMovePotAdapter(checkpoint, device)
    adapter.reset()
    chunk = adapter.predict_chunk(policy_observation)
    if chunk.source['observation_id'] != observation.observation_id or chunk.actions.shape != (50, 6):
        raise AssertionError('ACT prediction source or shape mismatch')
    target = tuple(float(x) for x in chunk.actions[0])
    request = action_request_from_policy(observation, backend.capabilities,
        policy_joint_names=adapter.action_names, policy_action_units=adapter.action_units,
        target=target, backend_to_policy={name: name + '.pos' for name in observation.joint_names})
    _, accepted, no_stop = execute_from_observation(backend, observation, request)
    if not accepted.accepted or no_stop is not None or backend.fake_actions != [target]:
        raise AssertionError('ACT target did not cross fake boundary exactly once')
    rejected_backend = SavedSO101FakeBackendAdapter(sample, reject=True)
    rejected_observation = rejected_backend.observe()
    _, rejected, rejection_stop = execute_from_observation(rejected_backend, rejected_observation, request)
    if rejected.accepted or not rejection_stop.completed or rejected_backend.fake_actions:
        raise AssertionError('Backend rejection did not latch stop')
    wrong_units_backend = SavedSO101FakeBackendAdapter(sample)
    wrong_units_observation = wrong_units_backend.observe()
    wrong_units = ActionRequest(observation.observation_id, observation.joint_names, ('rad',) * 6, target)
    try:
        execute_from_observation(wrong_units_backend, wrong_units_observation, wrong_units)
    except ValueError:
        if not wrong_units_backend.stopped or wrong_units_backend.fake_actions:
            raise AssertionError('Incompatible units did not stop before dispatch')
    else:
        raise AssertionError('Incompatible units were accepted')
    if file_sha256(sample) != source_hash or any(file_sha256(checkpoint / name) != digest
                                                   for name, digest in adapter.checkpoint_manifest.items()):
        raise AssertionError('Source data or model changed during validation')
    result = {
        'status': 'passed', 'scope': 'single saved observation, ACT first predicted action, fake backend only',
        'hardware_dispatches': 0, 'task_success_evaluated': False,
        'source': {'sample': str(sample.resolve()), 'sha256': source_hash,
                   'observation_id': observation.observation_id,
                   'camera_frame_ids': observation.camera_frame_ids,
                   'camera_capture_ns': observation.camera_capture_ns,
                   'state_capture_ns': observation.state_capture_ns},
        'policy': {'name': 'ACT', 'checkpoint': str(checkpoint.resolve()),
                   'checkpoint_manifest': adapter.checkpoint_manifest,
                   'predicted_steps': adapter.predicted_steps, 'executed_candidate_offset': 0,
                   'prediction_source': chunk.source, 'forward_ms': chunk.forward_ms,
                   'processing_and_forward_ms': chunk.processing_and_forward_ms},
        'action': asdict(request), 'accepted_receipt': asdict(accepted),
        'rejected_receipt': asdict(rejected), 'rejection_stop': asdict(rejection_stop),
        'wrong_units_rejected_before_dispatch': True, 'fake_actions_accepted': len(backend.fake_actions),
        'fake_actions_rejected': len(rejected_backend.fake_actions),
    }
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'hardware_dispatches': 0,
                      'fake_actions_accepted': len(backend.fake_actions),
                      'backend_rejection_stopped': rejection_stop.completed,
                      'wrong_units_rejected': True}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'executed_script.py').write_bytes(Path(__file__).read_bytes())
    try:
        main(args.sample, args.checkpoint, args.output, args.device)
    except BaseException:
        (args.output / 'failure.txt').write_text(traceback.format_exc())
        raise
