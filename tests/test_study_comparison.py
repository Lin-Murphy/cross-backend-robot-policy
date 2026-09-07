import json
from pathlib import Path
import tempfile
import unittest

from cross_backend.evaluation import digest, load_config, make_plan, normalize
from cross_backend.study_comparison import analyze_model_comparison, paired_statistics
from cross_backend.paired_backend_study import analyze_sim_real, ALIGNMENT_COMPONENTS, RESIDUALS


class ModelComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.run = self.root / 'run'; self.run.mkdir()
        for name in ('act', 'dot'):
            cp = self.root / name; cp.mkdir(); (cp / 'model.safetensors').write_bytes(name.encode())
        config = {'schema_version': 1, 'backend': 'aloha', 'task': 'transfer_cube', 'episodes': 2, 'first_seed': 1000,
                  'policies': [{'name': n, 'checkpoint': str(self.root / n)} for n in ('act', 'dot')]}
        self.config = self.root / 'aloha-config.json'; self.save(self.config, config)
        c = load_config(self.config); plan = make_plan(c, self.run)
        episodes = []; jobs = []
        for job in plan['jobs']:
            raw = Path(job['raw_directory']); raw.mkdir(parents=True)
            data = ({'per_task': [{'metrics': {'max_rewards': [0, 4], 'successes': [False, True]}}]}
                    if job['policy'] == 'act' else {'per_episode': [{'seed': 1000, 'max_reward': 4, 'success': False}, {'seed': 1001, 'max_reward': 4, 'success': False}]})
            self.save(raw / 'eval_info.json', data)
            episodes += normalize(job, c)
            jobs.append({'id': job['id'], 'policy': job['policy'], 'execution_status': 'completed',
                         'checkpoint_sha256': digest(self.root / job['policy'] / 'model.safetensors')})
        self.save(self.run / 'plan.json', plan)
        self.save(self.run / 'results.json', {'execution_status': 'completed', 'episodes': episodes, 'jobs': jobs})
        self.protocol = self.root / 'aloha-protocol.json'
        self.save(self.protocol, {'study_id': 'synthetic-test-only', 'backend': 'aloha', 'task': 'transfer_cube',
                 'scope': 'synthetic fixture, not experimental data', 'episodes_per_policy': 2, 'first_seed': 1000,
                 'config_sha256': digest(self.config), 'code_sha256': {},
                 'model_manifests': {n: {'model.safetensors': digest(self.root / n / 'model.safetensors')} for n in ('act', 'dot')}})

    @staticmethod
    def save(path, data): path.write_text(json.dumps(data))

    def test_rebuild_and_pair_from_native_evidence(self):
        r = analyze_model_comparison(self.run, self.protocol)
        self.assertTrue(r['evidence_verified']); self.assertEqual(r['paired']['pairs'], 2)
        self.assertEqual(r['paired']['difference_right_minus_left'], .5)

    def test_tampered_normalized_outcome_rejected(self):
        path = self.run / 'results.json'; data = json.loads(path.read_text())
        data['episodes'][0]['outcome'] = True; self.save(path, data)
        with self.assertRaisesRegex(ValueError, 'reproduce'): analyze_model_comparison(self.run, self.protocol)

    def test_changed_checkpoint_rejected(self):
        (self.root / 'act/model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'Checkpoint'): analyze_model_comparison(self.run, self.protocol)

    def test_paired_statistics_handle_empty_and_discordant_pairs(self):
        self.assertIsNone(paired_statistics([])['difference_right_minus_left'])
        r = paired_statistics([(False, True)] * 4)
        self.assertEqual(r['difference_right_minus_left'], 1)
        self.assertEqual(r['mcnemar_exact_two_sided'], .125)
        with self.assertRaises(ValueError): paired_statistics([(None, False)])


class SimRealTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model = self.ref('model.bin', b'fixture model')
        evidence = self.ref('measured.txt', b'SYNTHETIC TEST MEASUREMENTS ONLY')
        alignment = {'scene_sha256':'a'*64,'robot_calibration_sha256':'b'*64,'reviewer': 'test', 'reviewed_utc': '2026-09-28T00:00:00Z'}
        alignment.update({c: {'status': 'measured_and_reviewed', 'evidence': [evidence]} for c in ALIGNMENT_COMPONENTS})
        self.protocol_data = {'schema_version': 1, 'study_id': 'synthetic-test-only', 'task': 'move_pot',
                              'status': 'frozen', 'frozen_utc': '2026-09-28T00:00:00Z', 'cases': ['P01'], 'policies': ['act'],
                              'checkpoint_sha256': {'act': self.model['sha256']}, 'success_rule': 'placement_without_help',
                              'pair_tolerances': {n: 1 for n in RESIDUALS}, 'timing': {'control_hz': 30, 'policy_seconds': 20},
                              'alignment': self.ref('alignment.json', alignment)}
        initial = self.ref('initial.json', {'scene_sha256': 'a' * 64})
        first = self.ref('first.npz', b'synthetic array fixture')
        events = self.ref('events.jsonl', b'synthetic events fixture')
        media = self.ref('video.mp4', b'synthetic media fixture, not a real recording')
        profile = self.ref('profile.json', {'fps': 30, 'max_episode_seconds': 20,'model_path':str(self.root)})
        summary = self.ref('real-summary.json', {'policy': 'act', 'backend': 'so101', 'status': 'ended_unreviewed','model_path':str(self.root),'calibration_sha256':'b'*64,
                          'raw_goal_packets_transmitted': 3, 'profile_sha256': profile['sha256']})
        self.pair = {'case_id': 'P01', 'policy': 'act', 'simulation': {
            'summary': self.ref('sim-summary.json', {'trial_id': 'fixture-act', 'physical_alignment_verified': True,'alignment_spec_sha256':self.protocol_data['alignment']['sha256'],
                              'initial_file_sha256': initial['sha256'], 'status': 'timeout'}),
            'initial': initial, 'budget': self.ref('budget.json', {'control_hz': 30, 'max_control_ticks': 600}),
            'checkpoint_manifest': self.ref('checkpoint.json', {'model.safetensors': self.model['sha256']}),
            'video': media, 'events': events}, 'real': {
            'summary': summary, 'first_observation': first, 'events': events, 'checkpoint': self.model,
            'profile': profile, 'follower_video': media, 'camera2_video': media,
            'assessment': self.ref('assessment.json', {'trial_id':self.root.name,'policy': 'act', 'task_success': False, 'task_evidence': 'synthetic reviewer fixture'}),
            'audit': self.ref('audit.json', {'status': 'passed', 'summary_sha256': summary['sha256'],
                    'events_sha256': events['sha256'], 'paired_observations_actions_packets': 3})},
            'binding': self.ref('binding.json', {'case_id': 'P01', 'policy': 'act', 'reviewer': 'test',
                'reviewed_utc': '2026-09-28T00:00:00Z', 'success_rule': 'placement_without_help',
                'measured_residuals': {n: .1 for n in RESIDUALS}, 'measurement_evidence': [evidence],
                'simulation_initial_sha256': initial['sha256'], 'scene_sha256': 'a' * 64,
                'real_first_observation_sha256': first['sha256'], 'timing_review': 'matched_with_declared_backend_differences'})}

    def ref(self, name, data):
        path = self.root / name
        path.write_bytes(data if isinstance(data, bytes) else json.dumps(data).encode())
        return {'path': name, 'sha256': digest(path)}

    def analyze(self):
        protocol = self.ref('protocol.json', self.protocol_data)
        self.ref('manifest.json', {'protocol': protocol, 'pairs': [self.pair]})
        return analyze_sim_real(self.root / 'manifest.json')

    def test_failed_tasks_are_valid_pairs(self):
        result = self.analyze()
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(result['groups']['act']['agreement_rate'], 1)
        self.assertEqual(result['groups']['act']['both_failure'], 1)

    def test_nominal_mapping_is_not_upgraded_by_matching_names(self):
        path = self.root / 'sim-summary.json'; data = json.loads(path.read_text())
        data['physical_alignment_verified'] = False
        self.pair['simulation']['summary'] = self.ref(path.name, data)
        result = self.analyze()
        self.assertEqual(result['eligible_pairs'], 0)
        self.assertIn('Nominal simulation', result['pairs'][0]['reasons'][0])

    def test_different_timing_is_not_a_matched_pair(self):
        self.protocol_data['timing']['policy_seconds'] = 18
        result = self.analyze()
        self.assertEqual(result['eligible_pairs'], 0)
        self.assertIn('time budget', result['pairs'][0]['reasons'][0])

    def test_unknown_outcome_is_not_imputed(self):
        self.pair['real']['assessment'] = self.ref('assessment.json', {'trial_id':self.root.name,'policy': 'act', 'task_success': None})
        result = self.analyze()
        self.assertEqual(result['eligible_pairs'], 0)
        self.assertIsNone(result['groups']['act']['agreement_rate'])

    def test_missing_pairs_stay_missing(self):
        self.pair['real'] = None; self.pair['binding'] = None
        result = self.analyze()
        self.assertEqual(result['excluded_pairs'], 1)
        self.assertIn('real_not_recorded', result['pairs'][0]['reasons'])

    def test_hash_tamper_and_mismatched_checkpoints_excluded(self):
        (self.root / 'events.jsonl').write_bytes(b'changed')
        result = self.analyze(); self.assertEqual(result['eligible_pairs'], 0)
        self.assertIn('changed', result['pairs'][0]['reasons'][0])

    def test_missing_alignment_withholds_statistics(self):
        self.protocol_data['alignment'] = None
        result = self.analyze()
        self.assertEqual(result['eligible_pairs'], 0)
        self.assertTrue(result['global_issues'])


if __name__ == '__main__': unittest.main()
