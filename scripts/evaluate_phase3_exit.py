"""Evaluate the recorded Phase 3 bounded ACT hardware trace without rerunning hardware."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('artifacts/safety/phase3-exit.json'))
    args = parser.parse_args()
    trace = json.loads(args.trace.read_text(encoding='utf-8'))
    execution = trace['execution']
    samples = [execution['start_raw']] + execution['movement_samples_raw'] + execution['return_samples_raw']
    excursion = max(samples) - min(samples)
    checks = {
        'mode_is_act_one_shot': trace['mode'] == 'live_act_gripper_one_shot',
        'hardware_was_connected': trace['hardware_connected'] is True,
        'three_candidate_gates_passed': len(trace['steps']) == 3 and all(s['candidate_gate'] == 'passed' for s in trace['steps']),
        'bounded_one_shot_age_300ms': all(s['state_age_at_output_ms'] <= 300 for s in trace['steps']),
        'raw_command_delta_at_most_8': abs(execution['target_raw'] - execution['start_raw']) <= 8,
        'observable_motion_at_least_2': excursion >= 2,
        'excursion_at_most_12': excursion <= 12,
        'return_error_at_most_8': execution['return_error_ticks'] <= 8,
        'all_torques_off': not any(execution['final_torque']),
        'only_gripper_commanded': trace.get('mode') == 'live_act_gripper_one_shot',
    }
    report = {
        'source_trace': str(args.trace),
        'source_status': trace['status'],
        'known_source_metric_bug': 'source observed_excursion omitted start_raw',
        'recomputed_excursion_ticks': excursion,
        'checks': checks,
        'passed': all(checks.values()),
        'boundary': 'bounded gripper-only ACT smoke; not continuous rollout or task success',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"phase 3 exit {'passed' if report['passed'] else 'failed'}: {args.output}")
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
