"""Bounded R3 evidence replay and deterministic schedule checks. No inference/hardware."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cross_backend.degree_execution import DegreeCandidateGate, DegreeGateRejected, DegreeLimits, pop_checked
from cross_backend.move_pot_policy import CAMERAS, JOINTS, UNITS, MovePotChunk, ReplayChunkQueue, file_sha256

MS = 1_000_000
PERIOD = 33_333_333


def synthetic_limits():
    return DegreeLimits((-180.,)*5+(0.,), (180.,)*5+(100.,), (2.,)*6, (100.,)*6,
                        10*MS, 40*MS, 50*MS, 3000*MS, 3000*MS, 'synthetic_process_fixture_not_hardware_limits', True)


def context(capture_ns, session):
    return {'state': [0.]*6, 'joint_names': list(JOINTS), 'joint_units': list(UNITS),
            'state_capture_ns': capture_ns, 'clock_kind': 'synthetic', 'clock_session': session}


def schedule(out, name, latency_ns, prefilled=False, age_limit_ns=None):
    session = 'synthetic-' + name
    limits = synthetic_limits()
    if age_limit_ns is not None:
        limits = replace(limits, max_action_age_ns=age_limit_ns, max_camera_age_ns=age_limit_ns)
    gate = DegreeCandidateGate(limits, session, 'synthetic'); gate.reset(0)
    q = ReplayChunkQueue(out/(name+'.jsonl'), session)
    now = 0; accepted = 0; reasons = []; last_eval = 0
    try:
        for cid in range(2):
            capture_ns = now
            duration = 0 if prefilled and cid == 0 else latency_ns
            source = {**context(capture_ns, session), 'observation_id': f'{session}:{cid}',
                      'frame_ids': {k: f'{session}:{cid}:{k}' for k in CAMERAS},
                      'camera_capture_ns': {k: capture_ns for k in CAMERAS},
                      'synthetic_prediction': True, 'injected_inference_duration_ns': duration}
            q.enqueue(MovePotChunk(np.zeros((50,6)), source, now, now+duration, duration/1e6, duration/1e6), 50)
            ready_ns = now + duration
            for _ in range(50):
                # Synchronous refill; no catch-up burst, no fabricated wall-time sleep.
                now = max(last_eval + PERIOD, ready_ns)
                pop_checked(q, gate, context(now, session), now)
                accepted += 1; last_eval = now
    except DegreeGateRejected as exc:
        reasons = exc.reasons
        assert gate.latched and q.failed
        try: q.pop()
        except ValueError: pass
        else: raise AssertionError('Rejected queue was not latched')
    finally:
        discarded = len(q.pending); q.reset(); gate.reset(now); q.close()
    return {'accepted_candidates': accepted, 'rejection_reasons': reasons, 'discarded_on_reset': discarded,
            'clock_kind': 'synthetic', 'injected_inference_duration_ns': latency_ns,
            'last_evaluation_ns': now, 'limits': asdict(limits), 'hardware_dispatches': 0}


def main(out):
    predictions = ROOT/'artifacts/r2-captured-dual-policy-20260924-attempt01'
    captures = ROOT/'artifacts/r2-live-readonly-20260924-attempt01'
    lock = ROOT/'artifacts/move-pot-evaluation-v1-records'
    protected = list(predictions.glob('*.npz')) + list(predictions.glob('*events.jsonl')) + list(captures.glob('*.npz')) + list(lock.rglob('*.json'))
    before = {str(p.relative_to(ROOT)): file_sha256(p) for p in protected}
    result = {'scope': 'R3 first offline gate and timing slice; not full R3 acceptance',
              'hardware_dispatches': 0, 'inference_runs': 0, 'saved_input_replay': [], 'synthetic_schedules': {}}
    for model in ('act', 'smolvla'):
        rows = [json.loads(line) for line in (predictions/f'{model}-events.jsonl').read_text().splitlines()]
        rows = [r for r in rows if r['event'] == 'prediction']
        assert len(rows) == 3
        for i, row in enumerate(rows):
            with np.load(predictions/f'{model}-{i:02d}-actions.npz', allow_pickle=False) as data:
                actions = data['actions'].copy()
            with np.load(captures/f'sample-{i:02d}.npz', allow_pickle=False) as data:
                state = data['state'].tolist()
            np.testing.assert_array_equal(actions, row['actions'])
            assert row['source']['observation_npz_sha256'] == file_sha256(captures/f'sample-{i:02d}.npz')
            np.testing.assert_array_equal(state, row['source']['state'])
            # Keep all historical source timestamps unchanged (unknown exposure remains None).
            source = row['source']
            current = {'state': state, 'state_capture_ns': None, 'joint_names': list(JOINTS), 'joint_units': list(UNITS)}
            for mode in ('missing-physical-limits', 'synthetic-limits-diagnostic'):
                gate = DegreeCandidateGate(None if mode == 'missing-physical-limits' else synthetic_limits(),
                                           'r3-process-monotonic-session', 'monotonic')
                gate.reset(time.perf_counter_ns())
                event_path = out/f'{model}-{i:02d}-{mode}.jsonl'
                q = ReplayChunkQueue(event_path, f'{model}-{i}:{mode}')
                q.enqueue(MovePotChunk(actions, source, row['prediction_start_ns'], row['prediction_end_ns'],
                                       row['forward_ms'], row['processing_and_forward_ms']), 50)
                try:
                    pop_checked(q, gate, current, time.perf_counter_ns())
                except DegreeGateRejected as exc:
                    reasons = exc.reasons
                    assert any('capture_unknown' in reason for reason in reasons)
                    assert mode != 'missing-physical-limits' or 'missing_limits' in reasons
                    assert q.failed and gate.latched
                else: raise AssertionError('Unknown saved capture times must reject')
                q.reset(); gate.reset(None); q.close()
                result['saved_input_replay'].append({'model': model, 'sample': i, 'mode': mode,
                    'rejection_reasons': reasons, 'event_file': event_path.name, 'discarded_actions': 49})
    scenarios = [('act-like-sync', 6_300_000, False, None),
                 ('smolvla-like-sync', 113_400_000, False, None),
                 ('smolvla-like-prefilled-refill', 113_400_000, True, None),
                 ('act-like-source-expiry', 6_300_000, False, 100*MS),
                 ('smolvla-like-source-expiry', 113_400_000, True, 100*MS)]
    for name, delay, prefilled, age in scenarios:
        result['synthetic_schedules'][name] = schedule(out, name, delay, prefilled, age)
    schedules = result['synthetic_schedules']
    assert schedules['act-like-sync']['accepted_candidates'] == 100
    assert schedules['smolvla-like-sync']['accepted_candidates'] == 0
    assert schedules['smolvla-like-prefilled-refill']['accepted_candidates'] == 50
    assert schedules['act-like-source-expiry']['accepted_candidates'] == 3
    assert schedules['smolvla-like-source-expiry']['accepted_candidates'] == 3
    assert 'control_period_outside_limits' in schedules['smolvla-like-prefilled-refill']['rejection_reasons']
    for path in out.glob('*.jsonl'):
        for line in path.read_text().splitlines():
            e = json.loads(line)
            assert not e.get('hardware_dispatched', False)
            assert e.get('dispatch_monotonic_ns') is None
    assert all(file_sha256(ROOT/p) == digest for p, digest in before.items())
    result['protected_inputs_sha256'] = before
    result['protected_inputs_unchanged'] = True
    result['status'] = 'passed'
    (out/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'saved_rejections': len(result['saved_input_replay']),
                      'synthetic_schedules': {k:v['accepted_candidates'] for k,v in schedules.items()},
                      'protected_files': len(before), 'hardware_dispatches': 0}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    main(args.output)
