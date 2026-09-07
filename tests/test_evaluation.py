"""Unified lifecycle tests: task failures, partial runs, identity and device gates."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from cross_backend.evaluation import aggregate, load_config, make_plan, normalize, run_evaluation


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = {'schema_version': 1, 'backend': 'aloha', 'task': 'transfer_cube',
                       'policies': [{'name': 'act', 'checkpoint': 'weights'}],
                       'episodes': 2, 'first_seed': 1000}
        (self.root / 'weights').mkdir()
        (self.root / 'weights/model.safetensors').write_bytes(b'fixture only')

    def load(self):
        path = self.root / 'config.json'
        path.write_text(json.dumps(self.config))
        return load_config(path)

    def job(self, policy='act'):
        raw = self.root / policy
        raw.mkdir()
        return {'policy': policy, 'episode_index': 0, 'raw_directory': str(raw)}

    def write_rewards(self, job, rewards):
        data = {'per_task': [{'metrics': {'max_rewards': rewards, 'successes': [False] * len(rewards)}}]}
        (Path(job['raw_directory']) / 'eval_info.json').write_text(json.dumps(data))

    def test_configuration_paths_and_no_side_effect_plan(self):
        c = self.load()
        self.assertEqual(c['policies'][0]['checkpoint'], str(self.root / 'weights'))
        plan = make_plan(c, self.root / 'output')
        self.assertFalse((self.root / 'output').exists())
        self.assertIn('--eval.n_episodes=2', plan['jobs'][0]['argv'])

    def test_virtualenv_executable_symlink_is_preserved(self):
        bindir = self.root / 'venv/bin'
        bindir.mkdir(parents=True)
        executable = bindir / 'python'
        executable.symlink_to(sys.executable)
        self.config['python'] = 'venv/bin/python'
        c = self.load()
        self.assertEqual(c['python'], str(executable))
        self.assertEqual(make_plan(c, self.root / 'out')['jobs'][0]['argv'][0], str(bindir / 'lerobot-eval'))

    def test_reject_invalid_configurations(self):
        cases = [{'episodes': True}, {'episodes': 0}, {'task': 'other'}, {'unknown': 1},
                 {'policies': [{'name': 'smolvla', 'checkpoint': 'weights'}]},
                 {'scene': 'scene.xml'}, {'policies': []}]
        for change in cases:
            with self.subTest(change=change):
                original = self.config.copy()
                self.config.update(change)
                with self.assertRaises(ValueError): self.load()
                self.config = original

    def test_so101_never_implicitly_executes(self):
        self.config = {'schema_version': 1, 'backend': 'so101', 'task': 'move_pot',
                       'policies': [{'name': 'act'}]}
        c = self.load()
        with self.assertRaises(ValueError): make_plan(c, self.root / 'out')
        validation = make_plan(c, self.root / 'out', validate_only=True)
        self.assertIn('--validate-only', validation['jobs'][0]['argv'])
        self.assertNotIn('--execute-approved-once', validation['jobs'][0]['argv'])
        execution = make_plan(c, self.root / 'out', approved=True)
        self.assertIn('--shared-boundary', execution['jobs'][0]['argv'])
        self.config['episodes'] = 2
        with self.assertRaises(ValueError): self.load()

    def test_common_reward_rule_overrides_native_success(self):
        c = self.load()
        job = self.job()
        self.write_rewards(job, [0, 4])
        rows = normalize(job, c)
        self.assertEqual([r['outcome'] for r in rows], [False, True])
        self.assertEqual(aggregate(rows, c['policies'])[0]['success_rate_known'], .5)

    def test_episode_count_and_nonfinite_reward_rejected(self):
        c = self.load(); job = self.job()
        for rewards in ([4], [4, float('nan')]):
            self.write_rewards(job, rewards)
            with self.assertRaises(ValueError): normalize(job, c)

    def test_dot_seed_mismatch_rejected(self):
        c = self.load(); job = self.job('dot')
        data = {'per_episode': [{'seed': 0, 'max_reward': 4, 'success': False}]}
        (Path(job['raw_directory']) / 'eval_info.json').write_text(json.dumps(data))
        with self.assertRaises(ValueError): normalize(job, c)

    def test_mujoco_plan_uses_selected_policy_and_explicit_resources(self):
        self.config.update(backend='mujoco', task='move_pot', calibration='calibration.json',
                           sim_python='python3', initial_conditions=['initial1.json', 'initial2.json'])
        c = self.load(); jobs = make_plan(c, self.root / 'out')['jobs']
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0]['argv'][0], 'python3')
        self.assertIn(str(self.root / 'initial2.json'), jobs[1]['argv'])
        self.assertIn('--checkpoint', jobs[0]['argv'])
        self.assertEqual(jobs[1]['seed'], 1001)

    def test_mujoco_failure_and_unknown_are_distinct(self):
        c = self.load(); c.update(backend='mujoco', task='move_pot')
        job = self.job(); raw = Path(job['raw_directory']) / 'act'; raw.mkdir()
        for status, expected in [('timeout', False), ('incomplete', False), ('uncertain', None),
                                 ('stopped_error_or_gate', None), ('success', True)]:
            (raw / 'summary.json').write_text(json.dumps({'status': status}))
            self.assertIs(normalize(job, c)[0]['outcome'], expected)

    def test_real_unknown_does_not_become_failure(self):
        c = self.load(); c.update(backend='so101', task='move_pot')
        job = self.job()
        (Path(job['raw_directory']) / 'summary.json').write_text(json.dumps(
            {'backend': 'so101', 'policy': 'act', 'status': 'gate_rejected', 'raw_goal_packets_transmitted': 50}))
        rows = normalize(job, c)
        report = aggregate(rows, c['policies'])[0]
        self.assertIsNone(report['success_rate_known'])
        self.assertEqual(report['unknown_outcomes'], 1)

    def test_full_lifecycle_task_failure_is_successful_execution_and_no_overwrite(self):
        c = self.load(); output = self.root / 'out'; plan = make_plan(c, output)
        # Real subprocess fixture exercises capture, return codes and normalization without ML dependencies.
        fixture = self.root / 'fixture.py'
        fixture.write_text("import json,sys\nfrom pathlib import Path\np=Path(sys.argv[1]);p.mkdir()\n"
                           "(p/'eval_info.json').write_text(json.dumps({'per_task':[{'metrics':"
                           "{'max_rewards':[0,0],'successes':[False,False]}}]}))\n")
        plan['jobs'][0]['argv'] = [sys.executable, str(fixture), plan['jobs'][0]['raw_directory']]
        site_probe = subprocess.CompletedProcess([], 0, stdout='/tmp', stderr='')
        with patch('cross_backend.evaluation.subprocess.run', return_value=site_probe):
            result = run_evaluation(plan, output)
        self.assertEqual(result['execution_status'], 'completed')
        self.assertEqual(result['comparison'][0]['success_rate_known'], 0)
        self.assertEqual(json.loads((output / 'results.json').read_text()), result)
        with self.assertRaises(FileExistsError): run_evaluation(plan, output)

    def test_hardware_validation_does_not_count_as_a_missing_episode(self):
        self.config = {'schema_version': 1, 'backend': 'so101', 'task': 'move_pot',
                       'policies': [{'name': 'smolvla'}]}
        c = self.load(); output = self.root / 'out'
        plan = make_plan(c, output, validate_only=True)
        def fixture(argv, **kwargs):
            self.assertIn('--validate-only', argv)
            raw = Path(plan['jobs'][0]['raw_directory']); raw.mkdir()
            (raw / 'validation.json').write_text(json.dumps({'status': 'validated_no_hardware'}))
            class Process:
                def wait(self): return 0
            return Process()
        with patch('cross_backend.evaluation.subprocess.Popen', side_effect=fixture):
            result = run_evaluation(plan, output)
        self.assertEqual(result['execution_status'], 'completed')
        self.assertEqual(result['episodes'], [])
        self.assertEqual(result['comparison'][0]['missing_episodes'], 0)
        self.assertEqual(result['comparison'][0]['expected_episodes'], 0)

    def test_one_error_does_not_discard_other_policy_evidence(self):
        c = self.load(); c['policies'].append({'name': 'dot', 'checkpoint': str(self.root / 'missing')})
        output = self.root / 'out'; plan = make_plan(c, output)
        def fixture(argv, **kwargs):
            raw = Path(plan['jobs'][0]['raw_directory']); raw.mkdir()
            self.write_rewards(plan['jobs'][0], [4, 0])
            class Process:
                def wait(self): return 0
            return Process()
        with patch('cross_backend.evaluation.subprocess.run', return_value=subprocess.CompletedProcess([], 0, stdout='/tmp')), \
             patch('cross_backend.evaluation.subprocess.Popen', side_effect=fixture):
            result = run_evaluation(plan, output)
        self.assertEqual(result['execution_status'], 'error')
        self.assertEqual(len(result['episodes']), 2)
        self.assertEqual(result['jobs'][1]['execution_status'], 'error')
        self.assertIn('FileNotFoundError', result['jobs'][1]['error'])


if __name__ == '__main__':
    unittest.main()
