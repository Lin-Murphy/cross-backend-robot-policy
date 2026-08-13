"""Verify ordered real SO101 shared observations/actions against audited raw packets."""
import argparse
import hashlib
import json
from pathlib import Path

NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(run, calibration_summary):
    summary_path = run / 'summary.json'
    events_path = run / 'hardware-events.jsonl'
    summary = json.loads(summary_path.read_text())
    calibration = json.loads(calibration_summary.read_text())['hardware_calibration']
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    wanted = {'shared_backend_observation', 'legacy_goal_candidate',
              'legacy_goal_packet_transmitted', 'shared_backend_action_receipt'}
    ordered = [e for e in events if e['event'] in wanted]
    if summary['shared_boundary_enabled'] is not True:
        raise ValueError('missing shared boundary')
    count = summary['raw_goal_packets_transmitted']
    if len(ordered) < count * 4:
        raise ValueError('shared/raw packet count mismatch')
    tail = ordered[count*4:]
    observation_ids = set()
    max_source_to_receipt_ms = 0.0
    for index in range(count):
        observation, candidate, packet, shared = ordered[index*4:index*4+4]
        if [x['event'] for x in (observation, candidate, packet, shared)] != [
                'shared_backend_observation', 'legacy_goal_candidate',
                'legacy_goal_packet_transmitted', 'shared_backend_action_receipt']:
            raise ValueError(f'event order mismatch at packet {index}')
        oid = observation['observation_id']
        if oid in observation_ids or shared['observation_id'] != oid:
            raise ValueError(f'observation provenance mismatch at packet {index}')
        observation_ids.add(oid)
        if tuple(observation['joint_names']) != NAMES or tuple(observation['joint_units']) != (
                'deg', 'deg', 'deg', 'deg', 'deg', 'calibrated_0_100'):
            raise ValueError(f'joint schema mismatch at packet {index}')
        if set(observation['camera_frame_ids']) != {'follower', 'camera2'} or any(
                not observation['camera_frame_ids'][name].startswith(oid + ':')
                for name in ('follower', 'camera2')):
            raise ValueError(f'camera provenance mismatch at packet {index}')
        receipt = shared['receipt']
        if not receipt['accepted'] or receipt['physical_dispatches'] != 1 or                 receipt['ack_scope'] != 'sync_transport_return' or shared['stop'] is not None:
            raise ValueError(f'bad shared receipt at packet {index}')
        target = receipt['target']
        if len(target) != len(NAMES):
            raise ValueError(f'target length mismatch at packet {index}')
        raw = []
        for j, name in enumerate(NAMES):
            low = calibration[name]['range_min']
            high = calibration[name]['range_max']
            if j < 5:
                raw.append(int(target[j] * 4095 / 360 + (low + high) / 2))
            else:
                raw.append(int(min(100, max(0, target[j])) * (high - low) / 100 + low))
        if raw != packet['raw'] or raw != [candidate['raw_ids_values'][str(j)]
                                             for j in range(1, 7)]:
            raise ValueError(f'calibrated target differs from raw transport at packet {index}')
        if not observation['host_ns'] <= candidate['host_ns'] <= packet['host_ns'] <= receipt['backend_clock_ns']:
            raise ValueError(f'host clock order mismatch at packet {index}')
        max_source_to_receipt_ms = max(max_source_to_receipt_ms,
                                       (receipt['backend_clock_ns'] - observation['host_ns']) / 1e6)
    rejected = [e for e in events if e['event'] == 'legacy_trial_guard_rejected']
    rejection_stage = None
    if summary['status'] == 'gate_rejected':
        if len(rejected) != 1:
            raise ValueError('missing or duplicate guard rejection')
        if len(tail) == 2 and [e['event'] for e in tail] == [
                'shared_backend_observation', 'legacy_goal_candidate'] and \
                tail[1]['reasons'] and rejected[0]['reasons'] == tail[1]['reasons']:
            rejection_stage = 'candidate'
        elif not tail:
            # An encoder/rate violation can reject the source before a shared
            # observation or policy candidate exists.
            raw_feedback = [e for e in events if e['event'] == 'legacy_raw_feedback']
            if not raw_feedback or not raw_feedback[-1]['source_observation'] or \
                    raw_feedback[-1]['reasons'] != rejected[0]['reasons'] or \
                    (ordered and raw_feedback[-1]['host_ns'] <= ordered[-1]['receipt']['backend_clock_ns']):
                raise ValueError('trailing rejected feedback mismatch')
            rejection_stage = 'feedback'
        else:
            raise ValueError('trailing rejected candidate mismatch')
        last_event_ns = rejected[0]['host_ns']
    else:
        if tail or rejected:
            raise ValueError('unexpected trailing candidate or rejection')
        last_event_ns = ordered[-1]['receipt']['backend_clock_ns']
    stop_events = [e for e in events if e['event'] == 'legacy_stop_hold_transport_return']
    if len(stop_events) != 1 or stop_events[0]['host_ns'] <= last_event_ns:
        raise ValueError('missing or early stop hold')
    return {'status': 'passed', 'run': str(run), 'paired_observations_actions_packets': count,
            'trailing_rejected_candidate': rejection_stage == 'candidate',
            'rejection_stage': rejection_stage,
            'unique_observation_ids': len(observation_ids),
            'max_observation_event_to_receipt_ms': max_source_to_receipt_ms,
            'stop_hold_after_last_receipt': True,
            'camera_sensor_timestamps_verified': False,
            'raw_transport_is_per_motor_ack': False,
            'summary_sha256': digest(summary_path), 'events_sha256': digest(events_path),
            'calibration_summary_sha256': digest(calibration_summary)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--calibration-summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.calibration_summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': result['status'],
                      'paired_observations_actions_packets': result['paired_observations_actions_packets']}))
