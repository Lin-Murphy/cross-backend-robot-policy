"""Offline comparison of real pilot and simulation fixture completion milestones."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cross_backend.completion_record import CompletionRecord


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(real_dir, operator_path, scene_path, sim_dir, output):
    raw_path = real_dir / 'summary.json'
    event_path = real_dir / 'hardware-events.jsonl'
    real = json.loads(raw_path.read_text())
    operator = json.loads(operator_path.read_text())
    scene = json.loads(scene_path.read_text())
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    if real['status'] != 'gate_rejected' or not any(e['event'] == 'legacy_trial_guard_rejected' for e in events):
        raise ValueError('real run interruption evidence mismatch')
    if not operator['task_completed_by_operator'] or operator['returned_to_initial_pose'] is not False:
        raise ValueError('operator evidence mismatch')
    if any(sample['raw_positions'] != scene['samples'][0]['raw_positions'] for sample in scene['samples']):
        raise ValueError('post-run raw pose not stable')
    sim_path = sim_dir / 'summary.json'
    trace_path = sim_dir / 'events.jsonl'
    sim = json.loads(sim_path.read_text())
    last = json.loads(trace_path.read_text().splitlines()[-1])
    facts = last['facts']
    if sim['status'] != 'success' or last['task_status'] != 'success' or \
       [x['event'] for x in sim['task_events']] != ['started', 'lift_confirmed', 'success'] or \
       facts['zone'] != 'inside' or not facts['supported_on_mat'] or facts['any_jaw_contact']:
        raise ValueError('simulation task success evidence mismatch')
    task_required = ('object_on_target', 'object_released')
    cycle_required = task_required + ('robot_withdrawn', 'returned_to_start', 'normal_end')
    real_record = CompletionRecord('so101', 'move pot', 'aborted', {
        'object_on_target': {'state': 'yes', 'source': 'operator_report_only'},
        'object_released': {'state': 'unknown', 'source': 'no_contact_or_support_measurement'},
        'robot_withdrawn': {'state': 'no', 'source': 'operator_report_and_post_run_images'},
        'returned_to_start': {'state': 'no', 'source': 'operator_report_and_raw_pose'},
        'normal_end': {'state': 'no', 'source': 'guard_rejection'},
    }, task_required, cycle_required, {'raw_summary': digest(raw_path),
        'hardware_events': digest(event_path), 'operator': digest(operator_path), 'post_scene': digest(scene_path)}).to_dict()
    sim_record = CompletionRecord('mujoco', 'move pot', 'success', {
        'object_on_target': {'state': 'yes', 'source': 'sim_evaluator_zone_and_support'},
        'object_released': {'state': 'yes', 'source': 'sim_evaluator_no_jaw_contact'},
        'robot_withdrawn': {'state': 'unknown', 'source': 'not_evaluated'},
        'returned_to_start': {'state': 'unknown', 'source': 'not_evaluated'},
        'normal_end': {'state': 'yes', 'source': 'sim_task_terminal_success'},
    }, task_required, cycle_required, {'sim_summary': digest(sim_path), 'sim_trace': digest(trace_path)}).to_dict()
    if real_record['task_state'] != 'unknown' or real_record['full_cycle_state'] != 'no' or \
       sim_record['task_state'] != 'yes' or sim_record['full_cycle_state'] != 'unknown':
        raise RuntimeError('completion distinction collapsed')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'comparison.json').write_text(json.dumps({'real': real_record, 'simulation': sim_record,
        'formal_comparison': False, 'purpose': 'interface and evidence semantics only'}, ensure_ascii=False, indent=2) + '\n')
    print('passed: real task unknown/full cycle no; simulation task yes/full cycle unknown')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--real', type=Path, required=True)
    p.add_argument('--operator', type=Path, required=True)
    p.add_argument('--post-scene', type=Path, required=True)
    p.add_argument('--simulation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.real, a.operator, a.post_scene, a.simulation, a.output)
