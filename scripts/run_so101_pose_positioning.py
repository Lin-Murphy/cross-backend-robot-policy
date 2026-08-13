"""One-use, logged six-joint alignment to historical pilot episode 0 start. REAL MOTION."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback

from cross_backend.so101_pose_positioning import (NAMES,REFERENCE,TARGET,STEP_COUNT,PERIOD_S,
    PositioningStop,restricted_bus_class,verify_plan,waypoint)
from capture_move_pot_readonly import PORT,CAL,CAL_SHA


def main(out):
    from lerobot.motors import Motor,MotorCalibration,MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    raw=CAL.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CAL_SHA:raise PositioningStop('calibration hash changed')
    calibration=json.loads(raw)
    plan=verify_plan(calibration)
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    bus=restricted_bus_class(FeetechMotorsBus)(port=PORT,
        motors={n:Motor(i,'sts3215',MotorNormMode.RANGE_0_100 if n=='gripper' else MotorNormMode.DEGREES) for i,n in enumerate(NAMES,1)},
        calibration={n:MotorCalibration(**v) for n,v in calibration.items()})
    summary={'status':'failed','source_calibration_sha256':CAL_SHA,'dispatch_count':0,'steps_completed':0,
             'torque_initial':None,'torque_final':None,'target_raw':dict(zip(NAMES,TARGET))}
    stream=(out/'events.jsonl').open('w',buffering=1)
    def event(kind,**fields):stream.write(json.dumps({'event':kind,'monotonic_ns':time.monotonic_ns(),**fields})+'\n')
    def read(register):return {n:bus.read(register,n,normalize=False,num_retry=0) for n in NAMES}
    def check_state(expected,max_tracking=24):
        start=time.monotonic()
        pos=read('Present_Position');end=time.monotonic()
        event('feedback',raw=pos,read_duration_s=end-start)
        if end-start>.05:raise PositioningStop('slow position read')
        for n,goal in zip(NAMES,expected):
            value=pos[n]
            if not calibration[n]['range_min']<=value<=calibration[n]['range_max']:
                raise PositioningStop('state outside calibration: '+n)
            if abs(value-goal)>max_tracking:raise PositioningStop('tracking error: '+n)
        return pos
    def write(register,name,value):
        event('write_attempt',register=register,motor=name,value=value)
        bus.allowed_write(register,name,value)
        summary['dispatch_count']+=1
        event('write_acknowledged',register=register,motor=name,value=value)
        actual=bus.read(register,name,normalize=False,num_retry=0)
        event('write_readback',register=register,motor=name,value=actual)
        if actual!=value:raise PositioningStop('register readback mismatch: '+name)
    def stop_at_present():
        try:
            pos=read('Present_Position')
            torque=read('Torque_Enable')
            for n in NAMES:
                if torque[n]==1 and calibration[n]['range_min']<=pos[n]<=calibration[n]['range_max']:
                    write('Goal_Position',n,pos[n])
            event('stop_hold_targets_written',raw=pos,torque_before=torque)
        except BaseException as exc:
            event('stop_hold_failed',error=repr(exc))
    try:
        bus.connect()
        actual={n:asdict(v) for n,v in bus.read_calibration().items()}
        if actual!=calibration:raise PositioningStop('hardware calibration changed')
        for sample in range(3):
            positions=read('Present_Position');goals=read('Goal_Position');torque=read('Torque_Enable')
            event('preflight',sample=sample,positions=positions,goals=goals,torque=torque)
            if any(abs(positions[n]-REFERENCE[i])>2 for i,n in enumerate(NAMES)):
                raise PositioningStop('fresh pose differs from fixed reference')
            if any(torque[n]!=0 for n in NAMES):raise PositioningStop('torque not all off at preflight')
            if sample==0:initial_goals=goals
            elif goals!=initial_goals:raise PositioningStop('existing goals changed')
            time.sleep(.1)
        summary['torque_initial']=torque
        # Initial goal writes may automatically enable torque on this servo. Each is checked immediately.
        for i,n in enumerate(NAMES):
            write('Goal_Position',n,REFERENCE[i])
            torque=read('Torque_Enable');event('torque_after_initial_goal',motor=n,raw=torque)
            if any(torque[k] not in (0,1) for k in NAMES):raise PositioningStop('invalid torque state')
            if any(torque[NAMES[j]]!=0 for j in range(i+1,len(NAMES))):
                raise PositioningStop('uninitialized motor unexpectedly enabled')
            check_state(REFERENCE,4)
        for n in NAMES:
            if read('Torque_Enable')[n]==0:write('Torque_Enable',n,1)
            check_state(REFERENCE,4)
        if any(v!=1 for v in read('Torque_Enable').values()):raise PositioningStop('torque enable mismatch')
        event('movement_start',reference=dict(zip(NAMES,REFERENCE)),target=dict(zip(NAMES,TARGET)))
        start=time.monotonic();last=REFERENCE;last_position=check_state(last,4)
        for index in range(1,STEP_COUNT+1):
            due=start+index*PERIOD_S
            while time.monotonic()<due:time.sleep(min(.01,due-time.monotonic()))
            now=time.monotonic()
            if now-due>.05:raise PositioningStop('control period overrun')
            target=waypoint(index)
            if any(abs(v-old)>1 for v,old in zip(target,last)):raise PositioningStop('step bound violated')
            for i,n in enumerate(NAMES):
                if target[i]!=last[i]:write('Goal_Position',n,target[i])
            if any(v!=1 for v in read('Torque_Enable').values()):raise PositioningStop('torque changed during movement')
            position=check_state(target,24)
            if any(abs(position[n]-last_position[n])>8 for n in NAMES):
                raise PositioningStop('measured displacement >8 ticks per control period')
            last_position=position;last=target;summary['steps_completed']=index
            event('step_completed',index=index,target=dict(zip(NAMES,target)),position=position)
        time.sleep(.5)
        final=check_state(TARGET,6)
        summary['final_raw']=final;summary['status']='completed_pose_alignment'
        event('alignment_completed',raw=final)
    except BaseException as exc:
        summary['failure']=repr(exc);event('stopped_failure',error=repr(exc),traceback=traceback.format_exc())
        if bus.is_connected and summary['dispatch_count']:stop_at_present()
        raise
    finally:
        try:
            if bus.is_connected:
                summary['torque_final']=read('Torque_Enable')
                bus.disconnect(disable_torque=False)
        finally:
            (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
            stream.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('pose alignment deadline')))
    signal.alarm(90)
    try:main(args.output)
    finally:signal.alarm(0)
