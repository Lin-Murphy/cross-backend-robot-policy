"""Offline bridge audit of a saved SO101 capture; never imports a motor driver."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from cross_backend.move_pot_policy import MovePotObservation, MovePotChunk, ReplayChunkQueue, CAMERAS, JOINTS, UNITS
from cross_backend.degree_execution import DegreeCandidateGate, DegreeGateRejected, pop_checked


def audit(capture, output):
    output.mkdir(parents=True, exist_ok=False)
    summary = json.loads((capture / 'summary.json').read_text())
    if summary['status'] != 'passed' or summary['dispatches'] != 0:
        raise ValueError('Requires a successful no-dispatch capture')
    results = []
    for sample in summary['samples']:
        index = sample['index']
        path = capture / f'sample-{index:02d}.npz'
        with np.load(path, allow_pickle=False) as data:
            obs = MovePotObservation(
                images_rgb={key: data[key] for key in CAMERAS}, state=data['state'],
                observation_id=f'{capture.name}:{index}',
                frame_ids={key: f'{capture.name}:{index}:{key}' for key in CAMERAS},
                camera_capture_ns={key: None for key in CAMERAS}, state_capture_ns=None)
            obs.validate()
            np.testing.assert_array_equal(obs.state, np.asarray(sample['state'], dtype=np.float32))
            source = dict(observation_id=obs.observation_id, state=obs.state.tolist(),
                          frame_ids=obs.frame_ids, camera_capture_ns=dict(obs.camera_capture_ns),
                          state_capture_ns=None, joint_names=list(JOINTS), joint_units=list(UNITS),
                          capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          candidate_kind='synthetic_current_position_probe_not_policy_output',
                          clock_kind=None, clock_session=None)
            now = time.monotonic_ns()
            queue = ReplayChunkQueue(output / f'events-{index:02d}.jsonl', f'readonly-audit-{index}')
            gate = DegreeCandidateGate(None, None, None)
            try:
                queue.enqueue(MovePotChunk(np.repeat(obs.state[None], 2, axis=0), source, now, now, 0., 0.), 2)
                try:
                    pop_checked(queue, gate, source, time.monotonic_ns())
                except DegreeGateRejected as error:
                    assert 'missing_limits' in error.reasons
                    assert queue.failed and gate.latched
                    results.append({'sample': index, 'observation_contract': 'passed', 'reasons': error.reasons})
                else:
                    raise AssertionError('Expected refusal without physical limits and capture clocks')
                queue.reset()
                gate.reset(None)
            finally:
                queue.close()
    for path in output.glob('events-*.jsonl'):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert any(e['event'] == 'candidate' and e['gate_status'] == 'rejected' for e in events)
        assert all(not e.get('hardware_dispatched', False) for e in events)
    result = {'status': 'passed', 'backend': 'so101', 'mode': 'saved_readonly_contract_audit',
              'task_outcome': None, 'task_outcome_reason': 'No task trial; no object/contact ground truth',
              'policy_calls': 0, 'dispatches': 0, 'samples': results}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.capture, args.output)
