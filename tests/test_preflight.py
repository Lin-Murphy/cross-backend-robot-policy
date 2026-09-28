import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from cross_backend.preflight import check_config, probe_environment


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cp = self.root / 'checkpoint'; self.cp.mkdir()
        self.cfg = {'backend':'mujoco', 'task':'move_pot', 'policies':[{'name':'act','checkpoint':str(self.cp)}],
                    'python':sys.executable, 'sim_python':sys.executable, 'scene':str(self.root/'scene.xml'),
                    'calibration':str(self.root/'calibration.json')}
        self.metadata = {'type':'act','chunk_size':50,'n_action_steps':50,'n_obs_steps':1,'input_features':{'observation.state':{'shape':[6]},
            'observation.images.follower':{'shape':[3,480,640]},'observation.images.camera2':{'shape':[3,480,640]}},
            'output_features':{'action':{'shape':[6]}}}
        self.write_metadata()
        (self.cp/'model.safetensors').write_bytes(b'fixture: not loaded')
        for name in ('policy_preprocessor.json','policy_postprocessor.json'):
            (self.cp/name).write_text(json.dumps({'steps':[]}))
        (self.root/'scene.xml').write_text('<mujoco><asset/></mujoco>')
        source=Path(__file__).parent/'fixtures/so101-calibration.json'
        (self.root/'calibration.json').write_text(source.read_text())

    def write_metadata(self):
        (self.cp/'config.json').write_text(json.dumps(self.metadata))

    def check(self):
        with patch('cross_backend.preflight.probe_environment',return_value={'missing':[]}), \
             patch('cross_backend.preflight.shutil.which',return_value=sys.executable):
            return check_config(self.cfg, self.root/'new-output')

    def test_valid_checks_create_no_output_and_load_no_weights(self):
        self.assertEqual(self.check()['status'],'passed')
        self.assertFalse((self.root/'new-output').exists())

    def test_collects_missing_files_and_dimension_errors(self):
        (self.cp/'model.safetensors').unlink()
        self.metadata['output_features']['action']['shape']=[14];self.write_metadata()
        (self.root/'scene.xml').unlink()
        result=self.check();errors=[c for c in result['checks'] if c['status']=='error']
        self.assertEqual(len(errors),3)
        self.assertIn('expected action shape [6]',str(errors))

    def test_processor_state_and_mesh_references_checked(self):
        (self.cp/'policy_preprocessor.json').write_text(json.dumps({'steps':[{'state_file':'missing.safetensors'}]}))
        (self.root/'scene.xml').write_text('<mujoco><asset><mesh file="missing.stl"/></asset></mujoco>')
        result=self.check()
        self.assertEqual(result['status'],'failed')
        self.assertIn('missing.safetensors',str(result));self.assertIn('missing.stl',str(result))

    def test_existing_output_and_camera_mismatch_rejected(self):
        (self.root/'new-output').mkdir()
        self.metadata['input_features']['observation.images.camera2']['shape']=[3,224,224];self.write_metadata()
        result=self.check()
        self.assertIn('Output already exists',str(result));self.assertIn('camera features',str(result))

    def test_dependency_probe_and_timeout_are_actionable(self):
        with patch('cross_backend.preflight.probe_environment',side_effect=subprocess.TimeoutExpired('python',30)):
            result=check_config({'backend':'so101','policies':[{'name':'act'}]})
        self.assertEqual(result['status'],'failed')
        self.assertFalse(result['hardware_access'])
        self.assertEqual(probe_environment(sys.executable,['json','policybridge_missing_module'])['missing'],['policybridge_missing_module'])

    def test_cli_check_never_launches_evaluation(self):
        from scripts import evaluate
        config={'schema_version':1, 'backend':'so101','task':'move_pot','policies':[{'name':'act'}]}
        p=self.root/'config.json';p.write_text(json.dumps(config))
        with patch.object(sys,'argv',['evaluate.py','--config',str(p),'--check-environment']), \
             patch.object(evaluate,'check_config',return_value={'status':'passed'}), \
             patch.object(evaluate,'run_evaluation') as run, patch('builtins.print'):
            self.assertEqual(evaluate.main(),0)
            run.assert_not_called()

    def test_cli_failed_preflight_blocks_model_launch(self):
        from scripts import evaluate
        config={'schema_version':1,'backend':'aloha','task':'transfer_cube',
                'policies':[{'name':'act','checkpoint':str(self.cp)}]}
        p=self.root/'config.json';p.write_text(json.dumps(config))
        with patch.object(sys,'argv',['evaluate.py','--config',str(p),'--output',str(self.root/'out')]), \
             patch.object(evaluate,'check_config',return_value={'status':'failed'}), \
             patch.object(evaluate,'run_evaluation') as run, patch('builtins.print'):
            self.assertEqual(evaluate.main(),2)
            run.assert_not_called()
