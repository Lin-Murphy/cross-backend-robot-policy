"""Read-only SO101 register snapshot; motor writes are forbidden."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from capture_move_pot_readonly import PORT,CAL,CAL_SHA,NAMES,readonly_class


def main(out):
    from lerobot.motors import Motor,MotorCalibration,MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    raw=CAL.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CAL_SHA:raise RuntimeError('calibration_changed')
    expected=json.loads(raw)
    bus=readonly_class(FeetechMotorsBus)(port=PORT,
        motors={n:Motor(i,'sts3215',MotorNormMode.RANGE_0_100 if n=='gripper' else MotorNormMode.DEGREES)
                for i,n in enumerate(NAMES,1)},
        calibration={n:MotorCalibration(**c) for n,c in expected.items()})
    result={'status':'failed','port':PORT,'calibration_sha256':CAL_SHA,'motor_writes_enabled':False,
            'dispatches':0,'samples':[]}
    try:
        bus.connect()
        if {n:asdict(v) for n,v in bus.read_calibration().items()}!=expected:
            raise RuntimeError('hardware_calibration_changed')
        for index in range(3):
            sample={'index':index,'host_start_ns':time.perf_counter_ns()}
            for register in ('Present_Position','Goal_Position','Goal_Velocity','Torque_Enable'):
                sample[register]={n:bus.read(register,n,normalize=False,num_retry=0) for n in NAMES}
            sample['host_end_ns']=time.perf_counter_ns();result['samples'].append(sample)
            time.sleep(.1)
        result['status']='passed'
    finally:
        if bus.is_connected:bus.disconnect(disable_torque=False)
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    main(a.output)
