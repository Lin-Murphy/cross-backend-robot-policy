"""No-motion shared boundary check on saved SO101 data and MuJoCo."""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cross_backend.execution_contract import ActionRequest, execute_one
from cross_backend.offline_backend_adapters import SimJointBackendAdapter, SavedSO101FakeBackendAdapter
from cross_backend.tape_sim_backend import TapeSimBackend


def hold(observation):
    return ActionRequest(observation.observation_id, observation.joint_names,
                         observation.joint_units, observation.joint_position)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(sample, scene, candidate_path, output):
    output.mkdir(parents=True, exist_ok=False)
    candidate_data = json.loads(candidate_path.read_text())
    if candidate_data['saved_observation_sha256'] != sha(sample) or \
       candidate_data['fake_actions'] != 0:
        raise ValueError('offline policy candidate source mismatch')
    candidate = tuple(candidate_data['candidate_degrees_or_gripper_percent'])
    real = SavedSO101FakeBackendAdapter(sample)
    real_obs, real_action, real_receipt, real_stop = execute_one(real, lambda observation:
        ActionRequest(observation.observation_id, observation.joint_names,
                      observation.joint_units, candidate))
    if real.fake_actions != [real_action.target] or real_receipt.physical_dispatches != 0:
        raise RuntimeError('saved real fake dispatch mismatch')
    rejected = SavedSO101FakeBackendAdapter(sample, reject=True)
    _, _, reject_receipt, reject_stop = execute_one(rejected, hold)
    if reject_receipt.accepted or not rejected.stopped or rejected.fake_actions or not reject_stop.completed:
        raise RuntimeError('fake rejection not latched/stopped')
    simulator = TapeSimBackend(scene)
    try:
        simulator.reset()
        sim = SimJointBackendAdapter(simulator)
        sim_obs, sim_action, sim_receipt, sim_stop = execute_one(sim, hold)
        # Model's degree/percent action must not slip into a radian simulator.
        incompatible = SimJointBackendAdapter(simulator)
        try:
            execute_one(incompatible, lambda observation: ActionRequest(
                observation.observation_id, observation.joint_names,
                real.capabilities.action_units, candidate))
        except ValueError:
            if not incompatible.stopped:
                raise RuntimeError('incompatible policy action did not stop simulator adapter')
        else:
            raise RuntimeError('incompatible policy action was accepted')
        if not sim_receipt.accepted or simulator.index != 1:
            raise RuntimeError('simulation did not advance exactly once')
    finally:
        simulator.close()
    def describe_observation(observation):
        row = asdict(observation)
        row['images_rgb'] = {name: {'shape': list(image.shape),
            'sha256': hashlib.sha256(image.tobytes()).hexdigest()}
            for name, image in observation.images_rgb.items()}
        return row
    result = {'status': 'passed_shared_capture_action_receipt_stop_boundary',
        'physical_dispatches': 0, 'real_fake_actions': len(real.fake_actions),
        'real_rejection_stopped': rejected.stopped, 'real_rejection_stop_receipt': asdict(reject_stop),
        'incompatible_sim_policy_action_rejected': incompatible.stopped, 'sim_steps': 1,
        'so101': {'capabilities': asdict(real.capabilities), 'observation': describe_observation(real_obs),
                  'receipt': asdict(real_receipt), 'stop': None if real_stop is None else asdict(real_stop)},
        'simulation': {'capabilities': asdict(sim.capabilities), 'observation': describe_observation(sim_obs),
                       'receipt': asdict(sim_receipt), 'stop': None if sim_stop is None else asdict(sim_stop)},
        'evidence_sha256': {'saved_so101_sample': sha(sample), 'saved_policy_candidate': sha(candidate_path),
                            'simulation_scene': sha(scene)}}
    (output / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'physical_dispatches', 'real_fake_actions', 'real_rejection_stopped', 'sim_steps')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--saved-so101', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--saved-candidate', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.saved_so101, a.scene, a.saved_candidate, a.output)
