"""Seeed hardware reader for official-runtime inference; no motor writes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import cv2
import numpy as np
import sys
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


class ReadOnlyBus(FeetechMotorsBus):
    def write(self, *a, **kw):
        raise RuntimeError('Motor writes forbidden')
    def sync_write(self, *a, **kw):
        raise RuntimeError('Motor writes forbidden')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--serve', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--allow-gripper-one-shot', action='store_true')
    args = parser.parse_args()
    if not args.serve and args.output is None:
        parser.error('--output is required unless --serve is used')
    if args.serve and args.output_dir is None:
        parser.error('--output-dir is required with --serve')
    names = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
    path = Path(r'C:\Users\Murphy\.cache\huggingface\lerobot\calibration\robots\so_follower\so101_follower_arm.json')
    calibration = json.loads(path.read_bytes())
    bus = ReadOnlyBus(port='COM4', motors={n:Motor(i,'sts3215',MotorNormMode.RANGE_0_100 if n=='gripper' else MotorNormMode.RANGE_M100_100) for i,n in enumerate(names,1)}, calibration={n:MotorCalibration(**calibration[n]) for n in names})
    camera = None
    try:
        bus.connect()
        if {n:asdict(c) for n,c in bus.read_calibration().items()} != calibration:
            raise RuntimeError('Calibration mismatch')
        camera = cv2.VideoCapture(1)
        # Do not force MJPG: this camera is discovered and captured as YUY2 by
        # lerobot-find-cameras. Let the official OpenCV backend negotiate it.
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        for _ in range(3):
            ok,bgr = camera.read()
            if not ok:
                raise RuntimeError('Camera read failed')
        def capture(output):
            ok,bgr = camera.read()
            image_time = time.perf_counter()
            if not ok or bgr.shape != (480,640,3):
                raise RuntimeError('Camera read failed or shape mismatch')
            raw = bus.sync_read('Present_Position',normalize=False)
            for n in names:
                if not calibration[n]['range_min'] <= raw[n] <= calibration[n]['range_max']:
                    raise RuntimeError(f'Position out of range: {n}={raw[n]}')
            normalized = bus._normalize({calibration[n]['id']:raw[n] for n in names})
            state_time = time.perf_counter()
            torque = [bus.read('Torque_Enable',n,normalize=False) for n in names]
            np.savez(output, bgr=bgr, state=np.array([normalized[i] for i in range(1,7)],dtype=np.float32), raw=np.array([raw[n] for n in names]), torque=torque, image_time=image_time,state_time=state_time)
        if args.serve:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            print(json.dumps({'status':'ready'}), flush=True)
            index = 0
            executed = False
            for line in sys.stdin:
                command = json.loads(line)
                if command.get('command') == 'stop':
                    break
                if command.get('command') == 'execute_gripper':
                    if not args.allow_gripper_one_shot or executed:
                        raise RuntimeError('Gripper one-shot is not armed or already used')
                    executed = True
                    expected = np.asarray(command['expected_state'], dtype=float)
                    candidate = float(command['candidate_gripper'])
                    max_delta = float(command.get('max_delta', .5))
                    max_raw_delta = int(command.get('max_raw_delta', 8))
                    max_raw_excursion = int(command.get('max_raw_excursion', 12))
                    if not .5 <= max_delta <= 3.0 or not 8 <= max_raw_delta <= 45 or not 12 <= max_raw_excursion <= 55:
                        raise RuntimeError('Execution limits outside approved envelope')
                    if expected.shape != (6,) or not np.isfinite(expected).all() or not np.isfinite(candidate):
                        raise RuntimeError('Invalid execution request')
                    raw_now = bus.sync_read('Present_Position', normalize=False)
                    normalized_now = bus._normalize({calibration[n]['id']:raw_now[n] for n in names})
                    state_now = np.array([normalized_now[i] for i in range(1,7)])
                    if np.max(np.abs(state_now-expected)) > .5:
                        raise RuntimeError('State drift before one-shot')
                    torques = [bus.read('Torque_Enable',n,normalize=False) for n in names]
                    if any(torques):
                        raise RuntimeError('Expected all torques off')
                    if not 0 <= candidate <= 100 or abs(candidate-state_now[5]) > max_delta + 1e-6:
                        raise RuntimeError('Candidate outside gripper gate')
                    target_raw = bus._unnormalize({6:candidate})[6]
                    start_raw = raw_now['gripper']
                    if abs(target_raw-start_raw) > max_raw_delta:
                        raise RuntimeError('Raw target exceeds configured bound')
                    movement, returning = [], []
                    try:
                        FeetechMotorsBus.write(bus,'Goal_Position','gripper',start_raw,normalize=False)
                        FeetechMotorsBus.write(bus,'Torque_Enable','gripper',1,normalize=False)
                        FeetechMotorsBus.write(bus,'Goal_Position','gripper',target_raw,normalize=False)
                        for _ in range(5):
                            time.sleep(.1); p=bus.read('Present_Position','gripper',normalize=False); movement.append(p)
                            if abs(p-start_raw)>max_raw_excursion: raise RuntimeError('Excursion exceeded configured bound')
                        FeetechMotorsBus.write(bus,'Goal_Position','gripper',start_raw,normalize=False)
                        for _ in range(20):
                            time.sleep(.1); p=bus.read('Present_Position','gripper',normalize=False); returning.append(p)
                            if abs(p-start_raw)>max_raw_excursion: raise RuntimeError('Return excursion exceeded configured bound')
                    finally:
                        FeetechMotorsBus.write(bus,'Torque_Enable','gripper',0,normalize=False)
                    result={'status':'executed','start_raw':start_raw,'target_raw':target_raw,
                            'movement_samples_raw':movement,'return_samples_raw':returning,
                            'observed_excursion_ticks':max([start_raw]+movement+returning)-min([start_raw]+movement+returning),
                            'return_error_ticks':abs(returning[-1]-start_raw),
                            'max_delta_normalized':max_delta,'max_raw_delta':max_raw_delta,
                            'max_raw_excursion':max_raw_excursion,
                            'final_torque':[bus.read('Torque_Enable',n,normalize=False) for n in names]}
                    if result['observed_excursion_ticks'] < 2 or result['return_error_ticks'] > 8 or any(result['final_torque']):
                        result['status']='execution_failed'
                    print(json.dumps(result), flush=True)
                    continue
                if command.get('command') != 'capture':
                    raise RuntimeError('Unknown command')
                output = args.output_dir / f'observation-{index}.npz'
                capture(output)
                print(json.dumps({'status':'captured','path':str(output.resolve())}), flush=True)
                index += 1
        else:
            capture(args.output)
    finally:
        if camera is not None:
            camera.release()
        if bus.is_connected:
            bus.disconnect(disable_torque=False)


if __name__=='__main__':
    main()
