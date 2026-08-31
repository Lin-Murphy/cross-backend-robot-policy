import json
import unittest
from pathlib import Path
from cross_backend.so101_dispatch_backend import (NAMES,PhysicalDispatchLimits,
    SO101BoundedDispatcher,DispatchRejected,PartialDispatchError,calibrated_to_raw,restricted_policy_bus_class)


class FakeBus:
    def __init__(self,position):
        self.reg={n:{'Present_Position':v,'Goal_Position':v,'Torque_Enable':1,'Goal_Velocity':0} for n,v in zip(NAMES,position)}
        self.writes=[]
    def read(self,register,name,normalize=False,num_retry=0):return self.reg[name][register]
    def write(self,register,name,value,normalize=False,num_retry=0):
        self.writes.append((register,name,value));self.reg[name][register]=value


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.cal=json.loads((Path(__file__).parent/'fixtures/so101-calibration.json').read_text())
        self.position=(2076,1399,2634,3122,1969,2287)
        self.bus=FakeBus(self.position);self.events=[];self.clock=[1_000_000_000]
        self.limits=PhysicalDispatchLimits(
            tuple(self.cal[n]['range_min'] for n in NAMES),tuple(self.cal[n]['range_max'] for n in NAMES),
            (12,)*6,(120.0,)*6,(160.0,)*6,(20,)*6,(100,)*6,
            100_000_000,200_000_000,100_000_000,'test fixture only','development_pilot')
        self.runner=SO101BoundedDispatcher(self.bus,self.cal,self.limits,self.events.append,lambda:self.clock[0])
        self.current=[(self.position[i]-(self.cal[n]['range_min']+self.cal[n]['range_max'])/2)*360/4095
                      if i<5 else (self.position[i]-self.cal[n]['range_min'])*100/(self.cal[n]['range_max']-self.cal[n]['range_min'])
                      for i,n in enumerate(NAMES)]

    def test_requires_velocity_configuration_and_rejects_before_goals(self):
        with self.assertRaisesRegex(DispatchRejected,'velocity_limit_not_configured'):
            self.runner.dispatch(self.current,950_000_000)
        self.assertEqual(self.bus.writes,[])

    def test_first_real_policy_delta_rejected_without_goal_writes(self):
        self.runner.configure_velocity()
        self.assertEqual(len(self.bus.writes),6)
        action=[4.142,-40.570,47.069,51.767,-5.757,31.188]
        with self.assertRaises(DispatchRejected) as cm:self.runner.dispatch(action,950_000_000)
        self.assertIn('target_from_feedback_step:elbow_flex',cm.exception.reasons)
        self.assertFalse(any(register=='Goal_Position' for register,_,_ in self.bus.writes))
        self.assertTrue(self.runner.latched)

    def test_small_target_acknowledged_and_too_fast_next_step_rejected(self):
        self.runner.configure_velocity()
        result=self.runner.dispatch(self.current,950_000_000)
        self.assertEqual(result['raw'],self.position)
        self.assertEqual(len([x for x in self.events if x['event']=='dispatch_complete']),1)
        self.clock[0]+=50_000_000
        with self.assertRaisesRegex(DispatchRejected,'dispatch_period_outside_limits'):
            self.runner.dispatch(self.current,1_000_000_000)

    def test_stop_hold_preserves_torque_and_logs_measured_goals(self):
        self.runner.configure_velocity()
        result=self.runner.stop_hold()
        self.assertEqual(result['held_motors'],list(NAMES))
        self.assertTrue(self.runner.latched)
        self.assertEqual(len([x for x in self.bus.writes if x[0]=='Goal_Position']),6)
        self.assertTrue(all(self.bus.reg[n]['Torque_Enable']==1 for n in NAMES))

    def test_partial_write_failure_reports_already_written_motor(self):
        self.runner.configure_velocity()
        original=self.bus.write
        def fail_second_goal(register,name,value,normalize=False,num_retry=0):
            if register=='Goal_Position' and name=='shoulder_lift':raise OSError('simulated serial failure')
            return original(register,name,value,normalize=normalize,num_retry=num_retry)
        self.bus.write=fail_second_goal
        with self.assertRaises(PartialDispatchError) as cm:
            self.runner.dispatch(self.current,950_000_000)
        self.assertEqual(cm.exception.acknowledged_motors,('shoulder_pan',))
        self.assertTrue(self.runner.latched)
        self.assertEqual(len([e for e in self.events if e['event']=='partial_dispatch_failure']),1)

    def test_low_level_bus_allowlist_blocks_other_registers_and_profile_escape(self):
        class FakeBase:
            def __init__(self):
                self.motors={n:type('M',(),{'id':i})() for i,n in enumerate(NAMES,1)}
                self.writes=[]
            def write(self,register,name,value,normalize=False,num_retry=0):
                address=42 if register=='Goal_Position' else 46
                return self._write(address,2,self.motors[name].id,value,num_retry=num_retry)
            def _write(self,addr,length,motor_id,value,*,num_retry=0,raise_on_error=True,err_msg=''):
                self.writes.append((addr,length,motor_id,value))
        bus=restricted_policy_bus_class(FakeBase,self.limits)()
        with self.assertRaises(DispatchRejected):bus.write('Operating_Mode','shoulder_pan',0)
        with self.assertRaises(DispatchRejected):bus.write('Goal_Position','shoulder_pan',0)
        with self.assertRaises(DispatchRejected):bus.write('Goal_Velocity','shoulder_pan',999)
        bus.write('Goal_Velocity','shoulder_pan',100)
        bus.write('Goal_Position','shoulder_pan',self.position[0])
        self.assertEqual(bus.writes,[(46,2,1,100),(42,2,1,self.position[0])])

    def test_calibration_overflow_rejected_before_write(self):
        self.runner.configure_velocity()
        action=list(self.current);action[3]=1000
        with self.assertRaisesRegex(DispatchRejected,'action_outside_calibration:wrist_flex'):
            self.runner.dispatch(action,950_000_000)
        self.assertFalse(any(register=='Goal_Position' for register,_,_ in self.bus.writes))


if __name__=='__main__':unittest.main()
