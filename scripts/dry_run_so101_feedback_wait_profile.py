"""Counterfactual first two saved actions on a simulated tracking bus; NO devices."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cross_backend.so101_dispatch_backend import NAMES,PhysicalDispatchLimits,SO101BoundedDispatcher
from cross_backend.so101_feedback_wait import wait_for_feedback_window


def main(out):
    cfgpath=Path('configs/so101-smolvla-feedback-wait-pilot-20260927.json')
    predpath=Path('artifacts/so101-post-abort-no-motion-prediction-20260927/prediction.npz')
    cfg=json.loads(cfgpath.read_text())
    cal=json.loads(Path('artifacts/so101-bounded-smolvla-pilot-20260927-postcheck/summary.json').read_text())['hardware_calibration']
    keys=PhysicalDispatchLimits.__dataclass_fields__
    limits=PhysicalDispatchLimits(**{k:tuple(cfg[k]) if isinstance(cfg[k],list) else cfg[k] for k in keys}).validate(cal)
    actions=np.load(predpath)['actions'];now=[1_000_000_000];events=[]
    class FakeBus:
        def __init__(self):
            self.reg={n:{'Present_Position':cfg['reference_raw'][i],
                         'Goal_Position':cfg['reference_raw'][i],
                         'Torque_Enable':1,'Goal_Velocity':0} for i,n in enumerate(NAMES)}
            self.writes=[]
        def read(self,register,name,normalize=False,num_retry=0):return self.reg[name][register]
        def write(self,register,name,value,normalize=False,num_retry=0):
            assert register in ('Goal_Position','Goal_Velocity')
            self.writes.append((register,name,value));self.reg[name][register]=value
        def advance(self,seconds):
            now[0]+=round(seconds*1e9)
            step=max(1,int(320*seconds))
            for n in NAMES:
                p=self.reg[n]['Present_Position'];g=self.reg[n]['Goal_Position']
                self.reg[n]['Present_Position']=p+min(step,g-p) if g>=p else p+max(-step,g-p)
    bus=FakeBus();dispatch=SO101BoundedDispatcher(bus,cal,limits,events.append,lambda:now[0])
    dispatch.configure_velocity();rows=[];source=now[0]-500_000_000
    for i in range(2):
        if i:bus.advance(.105)
        wait=wait_for_feedback_window(bus,dispatch,actions[i],source,max_wait_ns=cfg['feedback_wait_max_ns'],
            poll_ns=cfg['feedback_poll_ns'],record=events.append,clock=lambda:now[0],sleep=bus.advance)
        result=dispatch.dispatch(actions[i],source)
        rows.append({'index':i,'raw':result['raw'],'wait_ms':wait['wait_elapsed_ns']/1e6,
                     'feedback_at_wait_ready':wait['feedback']})
    payload={'status':'passed_fake_bus_only','real_hardware_dispatches':0,
             'accepted_actions':len(rows),'assumed_servo_rate_ticks_s':320,
             'source_prediction_sha256':hashlib.sha256(predpath.read_bytes()).hexdigest(),
             'profile_sha256':hashlib.sha256(cfgpath.read_bytes()).hexdigest(),'rows':rows}
    (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    (out/'events.jsonl').write_text(''.join(json.dumps({**e,'fake_bus_only':True},default=str)+'\n' for e in events))
    print(json.dumps(payload,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);main(a.output)
