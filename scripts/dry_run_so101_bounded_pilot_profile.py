"""Exercise actual saved first-ten model actions against bounded dispatcher on a fake bus."""
import hashlib
import json
from pathlib import Path
import numpy as np
from cross_backend.so101_dispatch_backend import (NAMES,PhysicalDispatchLimits,
    SO101BoundedDispatcher,calibrated_to_raw)

PROFILE=Path('configs/so101-smolvla-bounded-pilot-20260926.json')
CAPTURE=Path('artifacts/so101-pose-alignment-postcheck-20260926')
PREDICTION=Path('artifacts/so101-postpose-smolvla-no-motion-20260926')
OUT=Path('artifacts/so101-bounded-pilot-fakebus-20260927-v4')


class FakeBus:
    def __init__(self,position):
        self.reg={n:{'Present_Position':position[n],'Goal_Position':position[n],
                     'Torque_Enable':1,'Goal_Velocity':0} for n in NAMES}
    def read(self,register,name,normalize=False,num_retry=0):return self.reg[name][register]
    def write(self,register,name,value,normalize=False,num_retry=0):
        if register not in ('Goal_Position','Goal_Velocity'):raise AssertionError('unexpected write')
        self.reg[name][register]=value
    def follow_goals_ideally(self):
        for n in NAMES:self.reg[n]['Present_Position']=self.reg[n]['Goal_Position']


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(PROFILE.read_text())
    s=json.loads((CAPTURE/'summary.json').read_text())
    cal=s['hardware_calibration']
    keys=PhysicalDispatchLimits.__dataclass_fields__
    limits=PhysicalDispatchLimits(**{k:tuple(cfg[k]) if isinstance(cfg[k],list) else cfg[k] for k in keys}).validate(cal)
    action=np.load(PREDICTION/'prediction.npz',allow_pickle=False)['actions'][:10]
    now=[1_000_000_000]
    bus=FakeBus(s['samples'][0]['raw_positions'])
    events=[]
    def record(event):events.append({**event,'hardware_dispatched':False,'fake_bus_only':True})
    dispatch=SO101BoundedDispatcher(bus,cal,limits,record,lambda:now[0])
    dispatch.configure_velocity()
    rows=[]
    for index,target in enumerate(action):
        if index:now[0]+=105_000_000
        result=dispatch.dispatch(target,now[0]-500_000_000)
        bus.follow_goals_ideally()
        rows.append({'index':index,'raw':result['raw'],'virtual_goal_accepted':True})
    (OUT/'events.jsonl').write_text(''.join(json.dumps(e,default=str)+'\n' for e in events))
    payload={'status':'passed_fake_bus_only','source_prediction_sha256':hashlib.sha256((PREDICTION/'prediction.npz').read_bytes()).hexdigest(),
             'profile_sha256':hashlib.sha256(PROFILE.read_bytes()).hexdigest(),'accepted_first_ten':len(rows),
             'real_hardware_dispatches':0,'rows':rows}
    (OUT/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({k:payload[k] for k in ('status','accepted_first_ten','real_hardware_dispatches')},indent=2))


if __name__=='__main__':main()
