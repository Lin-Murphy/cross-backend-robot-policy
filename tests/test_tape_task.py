import unittest
from dataclasses import replace
from cross_backend.tape_task import TapeFacts,TapeTaskEvaluator
START=TapeFacts(0,'outside',0,0,False,True,False,False)
LIFT=TapeFacts(100,'outside',.02,0,False,False,True,True)
PLACE=TapeFacts(200,'inside',0,0,True,False,False,False)
class TaskTests(unittest.TestCase):
    def run_seq(self,*frames):
        e=TapeTaskEvaluator()
        for f in frames:e.update(f)
        return e
    def test_lift_place_success(self):self.assertEqual(self.run_seq(START,LIFT,PLACE).status,'success')
    def test_push_not_success(self):self.assertEqual(self.run_seq(START,PLACE).finish(),'incomplete')
    def test_touch_one_jaw_not_lift(self):self.assertEqual(self.run_seq(START,replace(LIFT,both_jaws_contact=False),PLACE).finish(),'incomplete')
    def test_release_high_then_landing_is_drop(self):
        e=self.run_seq(START,LIFT,replace(PLACE,sim_ns=150,supported_on_mat=False,bottom_clearance_m=.1),PLACE)
        self.assertEqual(e.status,'drop')
    def test_low_release_allowed(self):
        e=self.run_seq(START,LIFT,replace(PLACE,sim_ns=150,supported_on_mat=False,bottom_clearance_m=.003),PLACE)
        self.assertEqual(e.status,'success')
    def test_boundary_not_success(self):self.assertEqual(self.run_seq(START,LIFT,replace(PLACE,zone='boundary')).finish(),'uncertain')
    def test_outside_placement(self):self.assertEqual(self.run_seq(START,LIFT,replace(PLACE,zone='outside',supported_on_mat=False,supported_on_table=True)).status,'placement_outside')
    def test_fast_contact_not_success(self):self.assertEqual(self.run_seq(START,LIFT,replace(PLACE,speed_m_s=.5)).finish(),'incomplete')
    def test_holding_not_success(self):self.assertEqual(self.run_seq(START,LIFT,replace(PLACE,any_jaw_contact=True)).finish(),'incomplete')
    def test_collision_latches(self):
        e=self.run_seq(START,LIFT,replace(PLACE,forbidden_contact=True),replace(PLACE,sim_ns=300));self.assertEqual(e.status,'collision')
    def test_initial_inside_or_airborne_rejected(self):
        for f in [replace(START,zone='inside'),replace(START,supported_on_table=False)]:self.assertEqual(self.run_seq(f).status,'invalid_initial_state')
    def test_timeout(self):self.assertEqual(self.run_seq(START,replace(PLACE,sim_ns=20_000_000_000)).status,'timeout')
    def test_invalid_time_and_nan(self):
        for f in [replace(LIFT,sim_ns=0),replace(LIFT,speed_m_s=float('nan'))]:self.assertEqual(self.run_seq(START,f).status,'input_error')
    def test_reset_forgets_lift(self):
        e=self.run_seq(START,LIFT);e.reset();e.update(START);e.update(PLACE);self.assertEqual(e.finish(),'incomplete')
if __name__=='__main__':unittest.main()
