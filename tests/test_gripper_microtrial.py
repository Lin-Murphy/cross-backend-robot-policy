import unittest
from cross_backend.gripper_microtrial import NAMES,TrialStop,restricted_bus_class,run_trial

class Clock:
    def __init__(self):self.t=0
    def now(self):return self.t
    def sleep(self,t):self.t+=t
class FakeBase:
    def __init__(self):self.state={n:2000 for n in NAMES};self.writes=[];self.failure=None;self.delay=0;self.clock=None
    def write(self,name,motor,value,**kwargs):return self._write(42,2,6,value,**{'num_retry':kwargs['num_retry']})
    def _write(self,addr,length,motor_id,value,**kwargs):
        self.writes.append((addr,length,motor_id,value))
        if self.failure:raise ConnectionError('injected failure')
        self.state['gripper']=value
    def read(self,name,motor,**kwargs):
        if self.clock:self.clock.sleep(self.delay)
        return self.state[motor]
    def disconnect(self,disable_torque=True):self.disconnect_value=disable_torque

class Tests(unittest.TestCase):
    def setUp(self):
        self.bus=restricted_bus_class(FakeBase)();self.clock=Clock();self.events=[]
        self.ref=dict(self.bus.state);self.cal={n:{'range_min':1000,'range_max':3000} for n in NAMES}
    def run_it(self):return run_trial(self.bus,self.ref,self.cal,self.clock.now,self.clock.sleep,self.events.append)
    def test_normal_exact_sequence_return_and_intervals(self):
        r=self.run_it();self.assertEqual(r['acknowledged_goals'],[2001,2002,2003,2004,2003,2002,2001,2000]);self.assertEqual(r['return_error_ticks'],0)
        times=[e['monotonic_s'] for e in self.events if e['event']=='write_attempt']
        self.assertTrue(all(b-a>=.1 for a,b in zip(times,times[1:])));self.assertTrue(all(w[:3]==(42,2,6) for w in self.bus.writes))
    def test_public_writes_and_torque_blocked(self):
        for name in ('write','sync_write','_sync_write','write_calibration','configure_motors','enable_torque','disable_torque'):
            with self.assertRaises(TrialStop):getattr(self.bus,name)('anything')
        with self.assertRaises(TrialStop):self.bus.disconnect(True)
        self.bus.disconnect();self.assertFalse(self.bus.disconnect_value)
    def test_lowlevel_without_permit(self):
        with self.assertRaises(TrialStop):self.bus._write(42,2,6,2001)
        self.assertEqual(self.bus.writes,[])
    def test_wrong_sequence_early_and_rearm(self):
        self.bus.arm_trial(2000,self.clock.now)
        with self.assertRaises(TrialStop):self.bus.write_gripper_goal(2004)
        self.bus.write_gripper_goal(2001)
        with self.assertRaises(TrialStop):self.bus.write_gripper_goal(2002)
        with self.assertRaises(TrialStop):self.bus.arm_trial(2000,self.clock.now)
    def test_changed_state_refuses_before_write(self):
        self.bus.state['wrist_flex']+=1
        with self.assertRaises(TrialStop):self.run_it()
        self.assertFalse(self.bus.writes)
    def test_failure_stops_without_return(self):
        self.bus.failure=True
        with self.assertRaises(ConnectionError):self.run_it()
        self.assertEqual(len(self.bus.writes),1)
        self.clock.sleep(.2)
        with self.assertRaises(TrialStop):self.bus.write_gripper_goal(2002)
    def test_slow_state_read(self):
        self.bus.clock=self.clock;self.bus.delay=.01
        with self.assertRaises(TrialStop):self.run_it()
        self.assertFalse(self.bus.writes)
    def test_displacement_abort_no_blind_return(self):
        original=self.bus._write
        def drift(*a,**kw):original(*a,**kw);self.bus.state['shoulder_pan']+=5
        self.bus._write=drift
        with self.assertRaises(TrialStop):self.run_it()
        self.assertEqual(len(self.bus.writes),1)
    def test_deadline_blocks_write(self):
        self.bus.arm_trial(2000,self.clock.now);self.clock.sleep(5)
        with self.assertRaises(TrialStop):self.bus.write_gripper_goal(2001)
        self.assertFalse(self.bus.writes)
    def test_bad_final_readback(self):
        original=self.bus._write
        def wrong(*a,**kw):original(*a,**kw);self.bus.state['gripper']=2003
        self.bus._write=wrong
        with self.assertRaisesRegex(TrialStop,'Return error'):self.run_it()
        self.assertEqual(len(self.bus.writes),8)

class StationaryTests(unittest.TestCase):
    def test_static_tracking_error_is_not_drift(self):
        from cross_backend.gripper_microtrial import verify_stationary_preflight
        ref={n:2000 for n in NAMES};cal={n:{'range_min':1000,'range_max':3000} for n in NAMES}
        class Bus:
            def read(self,k,n,**kwargs):return 1983 if k=='Goal_Position' and n=='elbow_flex' else 2000
        info=verify_stationary_preflight(Bus(),ref,cal,lambda t:None)
        self.assertEqual(info['static_goal_minus_position']['elbow_flex'],-17)
    def test_changed_goal_and_actual_posture_reject(self):
        from cross_backend.gripper_microtrial import verify_stationary_preflight
        ref={n:2000 for n in NAMES};cal={n:{'range_min':1000,'range_max':3000} for n in NAMES}
        for field in ('Goal_Position','Present_Position'):
            class Bus:
                count=0
                def read(self,k,n,**kwargs):
                    self.count+=1
                    return 2001 if self.count>12 and k==field and n=='gripper' else 2000
            with self.assertRaises(TrialStop):verify_stationary_preflight(Bus(),ref,cal,lambda t:None)

if __name__=='__main__':unittest.main()
