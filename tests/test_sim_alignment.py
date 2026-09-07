import json
from pathlib import Path
import tempfile
import unittest
from cross_backend.evaluation import digest, load_config, make_plan
from cross_backend.sim_alignment import load_sim_alignment
from cross_backend.paired_backend_study import ALIGNMENT_COMPONENTS

class AlignmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.file=self.root/'alignment.json'
        measured=self.root/'measurements.txt';measured.write_text('synthetic measurement fixture, not physical evidence')
        self.spec={'schema_version':1,'scene_sha256':'a'*64,'robot_calibration_sha256':'b'*64,'reviewer':'test','reviewed_utc':'2026-09-28',
                   'verified_coordinate_mapping':{'scale':[.01]*6,'offset':[0]*6,'policy_lower':[-10]*6,'policy_upper':[10]*6,
                   'sim_lower':[-1]*6,'sim_upper':[1]*6,'provenance':'synthetic fixture with verified flag for test only','validation':'physical_alignment_verified'}}
        self.spec.update({k:{'status':'measured_and_reviewed','evidence':[{'path':'measurements.txt','sha256':digest(measured)}]} for k in ALIGNMENT_COMPONENTS})
    def load(self):
        self.file.write_text(json.dumps(self.spec))
        class Backend:scene_sha256='a'*64
        return load_sim_alignment(self.file,Backend())
    def test_reviewed_mapping_is_used(self):
        m,meta=self.load();self.assertAlmostEqual(m.to_sim([1]*6)[0],.01)
        self.assertTrue(meta['physical_alignment_verified'])
    def test_missing_component_cannot_enable_physical_alignment(self):
        self.spec['camera_extrinsics']['status']='candidate'
        with self.assertRaises(ValueError):self.load()
    def test_wrong_scene_and_changed_measurements_refused(self):
        self.spec['scene_sha256']='b'*64
        with self.assertRaises(ValueError):self.load()
        self.spec['scene_sha256']='a'*64
        (self.root/'measurements.txt').write_text('changed')
        with self.assertRaises(ValueError):self.load()
    def test_timing_and_alignment_options_are_forwarded(self):
        c={'schema_version':1,'backend':'mujoco','task':'move_pot','policies':[{'name':'act','checkpoint':'cp'}],
           'calibration':'cal.json','initial_conditions':['initial.json'],'alignment_spec':'alignment.json','max_control_ticks':540}
        path=self.root/'config.json';path.write_text(json.dumps(c));cfg=load_config(path)
        job=make_plan(cfg,self.root/'out')['jobs'][0]
        self.assertIn('--alignment-spec',job['argv']);self.assertIn('540',job['argv'])
        c.pop('initial_conditions');path.write_text(json.dumps(c))
        with self.assertRaises(ValueError):load_config(path)
        c.pop('alignment_spec');c['max_control_ticks']=True;path.write_text(json.dumps(c))
        with self.assertRaises(ValueError):load_config(path)

if __name__=='__main__':unittest.main()
