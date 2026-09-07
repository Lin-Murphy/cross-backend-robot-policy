#!/usr/bin/env python3
"""Prepare an explicit, unstarted sim/real study; never accesses devices."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cross_backend.evaluation import digest, write_json
from cross_backend.paired_backend_study import RESIDUALS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', type=int, default=10, choices=range(1, 11))
    a = p.parse_args()
    source = json.loads(a.model_protocol.read_text())
    if source.get('backend') != 'mujoco' or set(source['model_manifests']) != {'act', 'smolvla'}:
        p.error('Require the ACT/SmolVLA move-pot model protocol')
    a.output.mkdir(parents=True, exist_ok=False)
    cases = [f'P{i:02d}' for i in range(1, a.cases + 1)]
    protocol = {'schema_version': 1, 'study_id': 'move-pot-measured-sim-real-20260928',
                'task': 'move_pot', 'status': 'draft', 'created_utc': datetime.now(timezone.utc).isoformat(),
                'frozen_utc': None, 'cases': cases, 'policies': ['act', 'smolvla'],
                'checkpoint_sha256': {n: source['model_manifests'][n]['model.safetensors'] for n in ('act', 'smolvla')},
                'model_protocol': {'path': str(a.model_protocol.resolve()), 'sha256': digest(a.model_protocol)},
                'success_rule': 'grasp object outside green mat and release fully inside without assistance',
                'timing': {'control_hz': 30, 'policy_seconds': 18},
                'alignment': None, 'pair_tolerances': {name: None for name in RESIDUALS},
                'scope': 'Prospective measured baseline pairs; not the historical 40-slot delay study.',
                'hardware_authorization': 'No motion authorization is provided by this draft or analysis tools.',
                'initial_state_rule': 'Measure joint state, object/mat pose and both camera views for each P case; freeze correspondence before running either policy.',
                'outcome_rule': 'Success, failure, stop, and unknown retained. A failed task is valid evidence. No reuse of unpaired past runs as P cases.'}
    write_json(a.output / 'protocol.json', protocol)
    reference = {'path': 'protocol.json', 'sha256': digest(a.output / 'protocol.json')}
    write_json(a.output / 'pairs.json', {'schema_version': 1, 'protocol': reference,
               'pairs': [{'case_id': case, 'policy': policy, 'simulation': None, 'real': None, 'binding': None}
                         for case in cases for policy in protocol['policies']]})
    write_json(a.output / 'initial-measurements.json', {'status': 'unmeasured', 'cases': [
        {'case_id': case, 'robot_state_deg_and_gripper': None, 'object_pose_in_table_frame': None,
         'mat_pose_in_table_frame': None, 'follower_photo': None, 'camera2_photo': None,
         'measured_by': None, 'measured_utc': None} for case in cases]})
    print(a.output / 'pairs.json')


if __name__ == '__main__': main()
