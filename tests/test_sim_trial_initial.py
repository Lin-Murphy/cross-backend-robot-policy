import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from cross_backend.sim_trial_initial import apply_initial_condition, load_initial_condition, initial_geometry_sha256
from cross_backend.tape_sim_backend import TapeSimBackend

SCENE=Path(__file__).resolve().parents[1]/'assets/so101/tape.xml'


class SimTrialInitialTests(unittest.TestCase):
    def setUp(self):
        self.b=TapeSimBackend(SCENE);self.b.reset()
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'initial.json'
        self.record={'schema_version':1,'backend':'mujoco','split':'C','initial_condition_id':'C01',
                     'scene_sha256':self.b.scene_sha256,'joint_unit':'sim_radian_unaligned',
                     'robot_q_rad':[0]*6,'tape_xy_m':[-.18,-.15],
                     'mat_xy_m':[-.18,.08],'tape_quaternion_wxyz':[1,0,0,0]}
    def tearDown(self):self.b.close();self.tmp.cleanup()
    def write(self):self.path.write_text(json.dumps(self.record)+'\n')
    def test_valid_reset_records_identity(self):
        self.write();info=apply_initial_condition(self.path,self.b,expected_split='C')
        self.assertEqual(info['initial_condition_id'],'C01')
        self.assertEqual(len(info['initial_file_sha256']),64)
        np.testing.assert_allclose(self.b.data.qpos[self.b.qadr],np.zeros(6))
        np.testing.assert_allclose(self.b.model.geom_pos[self.b.model.geom('green_mat').id,:2],[-.18,.08])
    def test_wrong_split_scene_unit_or_nan_does_not_mutate(self):
        self.b.step_sim_targets(np.zeros(6));before=self.b.snapshot()
        for key,value in [('split','V'),('scene_sha256','a'*64),('joint_unit','degree'),
                          ('robot_q_rad',[0,0,0,0,0,999])]:
            old=self.record[key];self.record[key]=value;self.write()
            with self.assertRaises(ValueError):load_initial_condition(self.path,self.b,expected_split='C')
            self.assertEqual(self.b.snapshot(),before)
            self.record[key]=old
    def test_geometry_identity_ignores_label_and_quaternion_sign(self):
        self.write();a=initial_geometry_sha256(self.record)
        other=dict(self.record,split='V',initial_condition_id='V01',
                   tape_quaternion_wxyz=[-1,0,0,0])
        self.assertEqual(a,initial_geometry_sha256(other))
        changed=dict(self.record,mat_xy_m=[-.16,.08])
        self.assertNotEqual(a,initial_geometry_sha256(changed))

    def test_t_id_requires_p_prefix(self):
        self.record['split']='T';self.record['initial_condition_id']='V01';self.write()
        with self.assertRaisesRegex(ValueError,'ID for split'):
            load_initial_condition(self.path,self.b,expected_split='T')


if __name__=='__main__':unittest.main()
