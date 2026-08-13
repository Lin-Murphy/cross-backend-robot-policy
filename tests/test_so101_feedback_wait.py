import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from cross_backend.so101_dispatch_backend import NAMES,PhysicalDispatchLimits,DispatchRejected
from cross_backend.so101_feedback_wait import wait_for_feedback_window


class FakeBus:
    def __init__(self,position,config):
        self.reg={n:{'Present_Position':position[i],'Goal_Position':position[i],
                     'Torque_Enable':1,'Goal_Velocity':config['goal_velocity_raw'][i]}
                  for i,n in enumerate(NAMES)}
        self.writes=[]
    def read(self,register,name,normalize=False,num_retry=0):return self.reg[name][register]
    def write(self,*args,**kwargs):self.writes.append((args,kwargs));raise AssertionError('wait must not write')


class FeedbackWaitTest(unittest.TestCase):
    def setUp(self):
        self.config=json.loads(Path('configs/so101-smolvla-feedback-wait-pilot-20260927.json').read_text())
        self.cal=json.loads(Path('artifacts/so101-bounded-smolvla-pilot-20260927-postcheck/summary.json').read_text())['hardware_calibration']
        keys=PhysicalDispatchLimits.__dataclass_fields__
        limits=PhysicalDispatchLimits(**{k:tuple(self.config[k]) if isinstance(self.config[k],list) else self.config[k]
                                         for k in keys}).validate(self.cal)
        self.start=list(self.config['reference_raw']);self.bus=FakeBus(self.start,self.config)
        self.dispatcher=SimpleNamespace(calibration=self.cal,limits=limits,last_dispatch_ns=1_000_000_000)
        self.now=[1_105_000_000];self.events=[]

    def action_for(self,raw):
        action=[]
        for i,n in enumerate(NAMES):
            c=self.cal[n]
            action.append((raw[i]-(c['range_min']+c['range_max'])/2)*360/4095+.00001 if i<5
                          else (raw[i]-c['range_min'])*100/(c['range_max']-c['range_min'])+.00001)
        return action

    def test_waits_for_slow_feedback_without_writing(self):
        target=self.start.copy();target[2]=2405
        self.bus.reg['elbow_flex']['Goal_Position']=2497
        def advance(seconds):
            self.now[0]+=round(seconds*1e9)
            self.bus.reg['elbow_flex']['Present_Position']-=20
        result=wait_for_feedback_window(self.bus,self.dispatcher,self.action_for(target),1_000_000_000,
            max_wait_ns=750_000_000,record=self.events.append,clock=lambda:self.now[0],sleep=advance)
        self.assertGreater(result['wait_elapsed_ns'],0)
        self.assertLessEqual(abs(result['raw_target'][2]-result['feedback']['elbow_flex']),120)
        self.assertEqual(self.bus.writes,[])
        self.assertGreater(len(self.events),1)

    def test_stalled_feedback_times_out_with_no_write(self):
        target=self.start.copy();target[2]=2405
        self.bus.reg['elbow_flex']['Goal_Position']=2497
        with self.assertRaisesRegex(DispatchRejected,'feedback_window_timeout'):
            wait_for_feedback_window(self.bus,self.dispatcher,self.action_for(target),1_000_000_000,
                max_wait_ns=100_000_000,record=self.events.append,clock=lambda:self.now[0],
                sleep=lambda seconds:self.now.__setitem__(0,self.now[0]+round(seconds*1e9)))
        self.assertEqual(self.bus.writes,[])

    def test_changed_torque_rejects_without_write(self):
        self.bus.reg['elbow_flex']['Torque_Enable']=0
        with self.assertRaisesRegex(DispatchRejected,'torque_changed:elbow_flex'):
            wait_for_feedback_window(self.bus,self.dispatcher,self.action_for(self.start),1_000_000_000,
                max_wait_ns=750_000_000,record=self.events.append,clock=lambda:self.now[0])
        self.assertEqual(self.bus.writes,[])

    def test_stale_source_rejects_without_write(self):
        with self.assertRaisesRegex(DispatchRejected,'host_observation_age_unknown_or_exceeded'):
            wait_for_feedback_window(self.bus,self.dispatcher,self.action_for(self.start),-1_000_000_000,
                max_wait_ns=750_000_000,record=self.events.append,clock=lambda:self.now[0])
        self.assertEqual(self.bus.writes,[])


if __name__=='__main__':unittest.main()
