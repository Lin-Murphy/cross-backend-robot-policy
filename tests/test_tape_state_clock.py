import unittest
from pathlib import Path
import numpy as np
from cross_backend.tape_sim_backend import TapeSimBackend
SCENE=Path(__file__).resolve().parents[1]/'assets/so101/tape-base-offset.xml'
class StateClockTests(unittest.TestCase):
    def setUp(self):self.b=TapeSimBackend(SCENE,1/30);self.b.reset()
    def tearDown(self):self.b.close()
    def test_thirty_hz_no_cumulative_drift(self):
        times=[]
        for _ in range(300):
            s=self.b.step_sim_targets(np.zeros(6));times.append(s['sim_ns']);self.assertGreaterEqual(s['control_lateness_ns'],0);self.assertLess(s['control_lateness_ns'],5_000_000)
        self.assertEqual(times[:3],[35_000_000,70_000_000,100_000_000]);self.assertEqual(times[-1],10_000_000_000)
    def test_restore_replays_contacts_and_positions(self):
        self.b.reset(mat_xy=[-.2,.09],tape_xy=[-.17,-.14])
        for _ in range(20):self.b.step_sim_targets(np.zeros(6))
        snapshot=self.b.snapshot();first=[]
        for _ in range(20):first.append(self.b.step_sim_targets([.02,0,0,0,0,.1]))
        self.b.reset();self.b.restore(snapshot)
        for prior in first:
            again=self.b.step_sim_targets([.02,0,0,0,0,.1]);np.testing.assert_allclose(again['qpos'],prior['qpos'],rtol=0,atol=1e-12);np.testing.assert_allclose(again['qvel'],prior['qvel'],rtol=0,atol=1e-12);self.assertEqual(again['sim_ns'],prior['sim_ns']);self.assertEqual(again['contacts'],prior['contacts'])
    def test_invalid_reset_preserves_state(self):
        self.b.step_sim_targets(np.zeros(6));before=self.b.snapshot()
        for kwargs in [dict(robot_q=[10]*6),dict(tape_xy=[float('nan'),0]),dict(tape_quaternion=[0,0,0,0])]:
            with self.assertRaises(ValueError):self.b.reset(**kwargs)
            self.assertEqual(self.b.snapshot(),before)
    def test_reset_restores_default_mat(self):
        self.b.reset(mat_xy=[.4,.4]);self.b.reset();np.testing.assert_array_equal(self.b.model.geom_pos[self.b.model.geom('green_mat').id],self.b.default_mat_pos)
    def test_bad_snapshot_clock_is_atomic(self):
        self.b.step_sim_targets(np.zeros(6));before=self.b.snapshot();bad=dict(before);bad['index']=100
        with self.assertRaises(ValueError):self.b.restore(bad)
        self.assertEqual(self.b.snapshot(),before)
    def test_wrong_identity_rejected(self):
        snap=self.b.snapshot();snap['scene_sha256']='wrong'
        with self.assertRaises(ValueError):self.b.restore(snap)
if __name__=='__main__':unittest.main()
