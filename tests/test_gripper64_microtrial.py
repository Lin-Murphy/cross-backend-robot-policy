import unittest
from cross_backend.gripper64_microtrial import NAMES, OFFSETS, TrialStop, restricted_bus_class, run_trial
from test_gripper_microtrial import Clock, FakeBase

class Gripper64Tests(unittest.TestCase):
    def setUp(self):
        self.bus=restricted_bus_class(FakeBase)();self.clock=Clock();self.events=[]
        self.ref=dict(self.bus.state);self.cal={n:{'range_min':1000,'range_max':3000} for n in NAMES}
    def run_it(self):return run_trial(self.bus,self.ref,self.cal,self.clock.now,self.clock.sleep,self.events.append)
    def test_full_round_trip_and_readbacks(self):
        r=self.run_it()
        self.assertEqual(r['acknowledged_goals'],[2000+x for x in OFFSETS])
        self.assertEqual(len(self.bus.writes),128)
        self.assertEqual(r['return_error_ticks'],0)
        self.assertEqual(r['gripper_max_observed_excursion_ticks'],64)
        self.assertLess(r['elapsed_s'],20)
        times=[e['monotonic_s'] for e in self.events if e['event']=='write_attempt']
        self.assertTrue(all(b-a>=.1 for a,b in zip(times,times[1:])))
        self.assertGreaterEqual(times[64]-times[63],1)
        self.assertEqual(len([e for e in self.events if e['event']=='goal_readback']),128)
        self.assertTrue(all(w[:3]==(42,2,6) for w in self.bus.writes))
        self.clock.sleep(.2)
        with self.assertRaises(TrialStop):self.bus.write_gripper_goal(2001)
    def test_mismatched_goal_stops_after_one(self):
        original=self.bus.read
        self.bus.read=lambda k,n,**kw: 1999 if k=='Goal_Position' else original(k,n,**kw)
        with self.assertRaisesRegex(TrialStop,'Goal register mismatch'):self.run_it()
        self.assertEqual(len(self.bus.writes),1)
    def test_drift_stops_without_return(self):
        for name,delta in [('gripper',69),('shoulder_pan',5)]:
            self.setUp();original=self.bus._write
            def drift(*a,**kw):
                original(*a,**kw);self.bus.state[name]=2000+delta
            self.bus._write=drift
            original_read=self.bus.read
            self.bus.read=lambda k,n,**kw: self.bus.writes[-1][3] if k=='Goal_Position' else original_read(k,n,**kw)
            with self.assertRaisesRegex(TrialStop,'Excess displacement'):self.run_it()
            self.assertEqual(len(self.bus.writes),1)
    def test_changed_reference_no_write(self):
        self.bus.state['gripper']+=1
        with self.assertRaises(TrialStop):self.run_it()
        self.assertFalse(self.bus.writes)
    def test_deadline_and_scope(self):
        self.bus.arm_trial(2000,self.clock.now)
        with self.assertRaises(TrialStop):self.bus.write_gripper_goal(2065)
        with self.assertRaises(TrialStop):self.bus._write(42,2,5,2001)
        for name in ('write','sync_write','enable_torque','disable_torque','write_calibration'):
            with self.assertRaises(TrialStop):getattr(self.bus,name)('anything')
        self.clock.sleep(20)
        with self.assertRaises(TrialStop):self.bus.write_gripper_goal(2001)
        self.assertFalse(self.bus.writes)
    def test_failed_write_no_retry(self):
        self.bus.failure=True
        with self.assertRaises(ConnectionError):self.run_it()
        self.assertEqual(len(self.bus.writes),1)
        self.clock.sleep(.2)
        with self.assertRaises(TrialStop):self.bus.write_gripper_goal(2002)
    def test_slow_goal_read(self):
        original=self.bus.read
        def slow(k,n,**kw):
            if k=='Goal_Position':self.clock.sleep(.051)
            return original(k,n,**kw)
        self.bus.read=slow
        with self.assertRaisesRegex(TrialStop,'Goal read exceeded'):self.run_it()
        self.assertEqual(len(self.bus.writes),1)

if __name__=='__main__':unittest.main()
