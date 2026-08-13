"""One explicitly authorized microtrial. Hardcoded gripper-only bounds; no models."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.gripper_microtrial import NAMES,TrialStop,restricted_bus_class,run_trial,verify_stationary_preflight
CAL=Path('/home/murphy/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower_arm.json')
EXPECTED='f911b3b1e57ed0d513d27eeb7bbd743086b09a03f9babd4900b3a734c0594434'
PORT='/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B7B016049-if00'


def main(out, repeat=False):
    from lerobot.motors import Motor,MotorCalibration,MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    proposal=ROOT/'artifacts/r3-operator-review-20260924T225104Z/proposal.json'
    p=json.loads(proposal.read_text());ref={n:p['current_calibration_margins'][n]['raw'] for n in NAMES}
    assert ref==dict(zip(NAMES,(2087,1441,2632,3139,2021,2290)))
    assert hashlib.sha256(CAL.read_bytes()).hexdigest()==EXPECTED
    cal=json.loads(CAL.read_text());assert all(cal[n]['id']==i for i,n in enumerate(NAMES,1))
    bus=restricted_bus_class(FeetechMotorsBus)(port=PORT,motors={n:Motor(i,'sts3215',MotorNormMode.RANGE_0_100 if n=='gripper' else MotorNormMode.DEGREES) for i,n in enumerate(NAMES,1)},calibration={n:MotorCalibration(**v) for n,v in cal.items()})
    result={'status':'preflight','authorization':('User: 再试试，没看清; one identical repeat, no larger amplitude.' if repeat else 'User: 可以，你来执行; reviewed gripper +4 tick and return sequence.'),
            'proposal_sha256':hashlib.sha256(proposal.read_bytes()).hexdigest(),'calibration_sha256':EXPECTED,
            'torque_writes':0,'other_joint_goal_writes':0,'policy_used':False,'automatic_retry':False}
    stream=(out/'events.jsonl').open('x')
    def record(event):
        stream.write(json.dumps(event,allow_nan=False)+'\n');stream.flush()
    def timeout(*_):raise TrialStop('Hard timeout: no further writes')
    signal.signal(signal.SIGALRM,timeout);signal.signal(signal.SIGTERM,timeout);signal.alarm(30)
    try:
        print('Opening restricted bus for fresh read-only preflight',flush=True);bus.connect()
        actual={n:asdict(v) for n,v in bus.read_calibration().items()};result['hardware_calibration']=actual
        if any(cal[n][k]!=actual[n][k] for n in NAMES for k in ('id','homing_offset','range_min','range_max')):raise TrialStop('Calibration mismatch')
        torque={n:bus.read('Torque_Enable',n,normalize=False,num_retry=0) for n in NAMES};result['torque_before']=torque
        if any(v!=1 for v in torque.values()):raise TrialStop('Torque state changed; refuse to enable it')
        result['stationary_preflight']=verify_stationary_preflight(bus,ref,cal,time.sleep)
        if repeat:
            for count in (3,2,1):
                print(f'Observation countdown: {count}',flush=True)
                time.sleep(1)
            result['stationary_preflight_after_countdown']=verify_stationary_preflight(bus,ref,cal,time.sleep)
            if any(bus.read('Torque_Enable',n,normalize=False,num_retry=0)!=1 for n in NAMES):raise TrialStop('Torque changed during countdown')
        print('Fresh preflight passed; executing only the approved gripper sequence',flush=True)
        signal.setitimer(signal.ITIMER_REAL,5)
        result['trial']=run_trial(bus,ref,cal,time.monotonic,time.sleep,record)
        signal.setitimer(signal.ITIMER_REAL,0)
        result['status']=result['trial']['status']
    except BaseException as exc:
        result.update(status='stopped_failure',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
        print('STOPPED: '+str(exc),flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL,3)
        try:
            if bus.is_connected:
                result['torque_after']={n:bus.read('Torque_Enable',n,normalize=False,num_retry=0) for n in NAMES}
                result['final_raw_readback']={n:bus.read('Present_Position',n,normalize=False,num_retry=0) for n in NAMES}
                if result.get('torque_before')!=result['torque_after']:
                    result['status']='stopped_failure';result['torque_changed_externally']=True
        except BaseException as exc:result['cleanup_read_error']=str(exc);result['status']='stopped_failure'
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            try:
                if bus.is_connected:bus.disconnect(disable_torque=False)
                result['serial_closed']=not bus.is_connected
            except BaseException as exc:result['disconnect_error']=str(exc);result['status']='stopped_failure'
            stream.close()
            events=[json.loads(line) for line in (out/'events.jsonl').read_text().splitlines()]
            result['write_attempts']=sum(e['event']=='write_attempt' for e in events)
            result['acknowledged_writes']=sum(e['event']=='write_acknowledged' for e in events)
            result['calibration_unchanged']=hashlib.sha256(CAL.read_bytes()).hexdigest()==EXPECTED
            (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
    if result['status']!='passed_bounded_return_check':raise SystemExit(1)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--execute-approved-once',action='store_true')
    group.add_argument('--repeat-authorized-once',action='store_true')
    args=parser.parse_args()
    root=ROOT/'artifacts/r3-gripper-authorized-20260924'
    if args.repeat_authorized_once:
        prior=json.loads((root/'hardware-attempt02/summary.json').read_text())
        if prior['status']!='passed_bounded_return_check' or prior['acknowledged_writes']!=8 or not prior['serial_closed']:
            raise SystemExit('Prior trial not cleanly completed; no repeat')
        out=root/'hardware-attempt03-user-repeat'
    else:
        prior=json.loads((root/'hardware-attempt01/summary.json').read_text())
        if prior['write_attempts']!=0 or prior['acknowledged_writes']!=0:raise SystemExit('Prior motion attempted; refuse another trial')
        if prior.get('error')!='Existing goal differs from reference posture':raise SystemExit('Unexpected prior failure; refuse retry')
        out=root/'hardware-attempt02'
    out.mkdir(parents=True,exist_ok=False)
    (out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    (out/'microtrial_source.py').write_bytes((ROOT/'src/cross_backend/gripper_microtrial.py').read_bytes())
    main(out,args.repeat_authorized_once)
