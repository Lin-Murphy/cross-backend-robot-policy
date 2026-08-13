"""Read-only follow-up to stopped preflight; no allowed motor writes."""
import json
from pathlib import Path
import signal
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.capture_move_pot_readonly import readonly_class,PORT,CAL,CAL_SHA,NAMES
import hashlib

def main():
    from lerobot.motors import Motor,MotorCalibration,MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    out=ROOT/'artifacts/r3-gripper-authorized-20260924/no-response-registers.json'
    if out.exists():raise RuntimeError('No overwrite')
    assert hashlib.sha256(CAL.read_bytes()).hexdigest()==CAL_SHA
    cal=json.loads(CAL.read_text())
    bus=readonly_class(FeetechMotorsBus)(port=PORT,motors={n:Motor(i,'sts3215',MotorNormMode.RANGE_0_100 if n=='gripper' else MotorNormMode.DEGREES) for i,n in enumerate(NAMES,1)},calibration={n:MotorCalibration(**v) for n,v in cal.items()})
    result={'status':'running','motor_writes':0,'samples':[]}
    def stop(*_):raise TimeoutError('Read-only time limit')
    signal.signal(signal.SIGALRM,stop);signal.alarm(15)
    try:
        bus.connect()
        result['configuration_raw']={n:{k:bus.read(k,n,normalize=False,num_retry=0) for k in ('Firmware_Major_Version','Firmware_Minor_Version','CW_Dead_Zone','CCW_Dead_Zone','Goal_Position','Goal_Time','Goal_Velocity','Acceleration','Operating_Mode','Torque_Enable','Torque_Limit','Max_Torque_Limit','Min_Voltage_Limit','Max_Voltage_Limit','Minimum_Startup_Force','Lock')} for n in NAMES}
        for i in range(10):
            start=time.monotonic_ns()
            rows={n:{k:bus.read(k,n,normalize=False,num_retry=0) for k in ('Present_Position','Present_Velocity','Present_Load','Present_Voltage','Present_Temperature','Present_Current','Moving','Status','Goal_Position')} for n in NAMES}
            result['samples'].append({'index':i,'start_ns':start,'end_ns':time.monotonic_ns(),'motors':rows})
            time.sleep(.1)
        result['torque_after']={n:bus.read('Torque_Enable',n,normalize=False,num_retry=0) for n in NAMES}
        result['status']='passed_readonly'
    except BaseException as exc:result['error']=str(exc);result['status']='failed';raise
    finally:
        try:
            if bus.is_connected:bus.disconnect(disable_torque=False)
        finally:
            signal.alarm(0);out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'configuration_raw':result['configuration_raw'],'position_span':{n:max(r['motors'][n]['Present_Position'] for r in result['samples'])-min(r['motors'][n]['Present_Position'] for r in result['samples']) for n in NAMES},'first_feedback':result['samples'][0]['motors'],'status':'passed_readonly','motor_writes':0},indent=2))

if __name__=='__main__':main()
