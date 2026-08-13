import unittest
import numpy as np
from cross_backend.sim_policy_mapping import CoordinateMapping,mapping_from_spec,simulation_observation
from cross_backend.move_pot_policy import CAMERAS

class MappingTests(unittest.TestCase):
    def setUp(self):
        self.m=CoordinateMapping((np.pi/180,-np.pi/180,np.pi/180,np.pi/180,np.pi/180,.01),(.1,.2,0,0,0,-.2),(-90,)*5+(0,), (90,)*5+(100,),(-2,)*6,(2,)*6,'synthetic signed/offset fixture','synthetic_test')
    def test_known_units_sign_and_gripper(self):
        q=self.m.to_sim([0,90,0,0,0,100]);self.assertAlmostEqual(q[0],.1);self.assertAlmostEqual(q[1],.2-np.pi/2);self.assertAlmostEqual(q[5],.8)
        np.testing.assert_allclose(self.m.to_policy(q),[0,90,0,0,0,100],atol=1e-12)
    def test_no_clip_and_bad_units(self):
        for a in [[0]*5+[101],[float('nan')]*6,[0]*5]:
            with self.assertRaises(ValueError):self.m.to_sim(a)
        with self.assertRaises(ValueError):self.m.to_sim([0]*6,joint_units=('rad',)*6)
    def test_missing_mapping_refuses(self):
        with self.assertRaises(ValueError):mapping_from_spec({'mapping':{'body':'guess zero'}})
    def test_sim_bounds_also_apply(self):
        with self.assertRaises(ValueError):self.m.to_policy([3]*6)
    def test_named_packet_no_ground_truth_leak(self):
        class Data:time=.5;qpos=np.array([.1,.2,0,0,0,.3])
        class Backend:
            data=Data();index=25;qadr=list(range(6))
            def render(self):return {n:np.zeros((480,640,3),dtype=np.uint8) for n in CAMERAS}
        obs,meta=simulation_observation(Backend(),self.m,'synthetic-episode')
        obs.validate();self.assertEqual(obs.state_capture_ns,500_000_000)
        self.assertAlmostEqual(obs.state[-1],50);self.assertFalse(meta['policy_dispatch_authorized'])
        self.assertEqual(meta['clock_domain'],'simulation');self.assertFalse(hasattr(obs,'tape_position'))
if __name__=='__main__':unittest.main()
