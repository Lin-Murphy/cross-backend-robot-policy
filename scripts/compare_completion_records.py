"""Compare two backend-neutral task milestone records without claiming paired performance."""
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


def validated(path):
    source = json.loads(path.read_text())
    if source.get('schema_version') != 1:
        raise ValueError('unsupported completion schema')
    record = CompletionRecord(source['backend'], source['task'], source['run_status'],
                              source['stages'], tuple(source['task_required']),
                              tuple(source['full_cycle_required']), source['evidence_sha256']).to_dict()
    if record != source:
        raise ValueError('completion record does not match derived milestones')
    return record


def compare(left_path, right_path):
    left, right = validated(left_path), validated(right_path)
    if left['task'] != right['task'] or left['task_required'] != right['task_required'] or \
       left['full_cycle_required'] != right['full_cycle_required']:
        raise ValueError('incompatible task or completion rules')
    if left['backend'] == right['backend']:
        raise ValueError('comparison requires distinct backends')
    names = list(dict.fromkeys(left['task_required'] + left['full_cycle_required']))
    return {
        'schema_version': 1,
        'task': left['task'],
        'task_required': left['task_required'],
        'full_cycle_required': left['full_cycle_required'],
        'left': left,
        'right': right,
        'milestone_states': {name: {
            'left': left['stages'][name]['state'], 'right': right['stages'][name]['state']}
            for name in names},
        'task_states': {'left': left['task_state'], 'right': right['task_state']},
        'full_cycle_states': {'left': left['full_cycle_state'], 'right': right['full_cycle_state']},
        'record_sources': {'left': {'path': str(left_path), 'sha256': digest(left_path)},
                           'right': {'path': str(right_path), 'sha256': digest(right_path)}},
        'formal_performance_comparison': False,
        'paired_initial_state_verified': False,
        'same_policy_verified': False,
        'purpose': 'shared task semantics and evidence provenance across backends',
    }


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--left', type=Path, required=True)
    p.add_argument('--right', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = compare(args.left, args.right)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'passed', 'task_states': result['task_states'],
                      'full_cycle_states': result['full_cycle_states']}))
