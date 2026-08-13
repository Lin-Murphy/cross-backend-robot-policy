"""Explicitly armed, gripper-only raw-position transport smoke (not ACT)."""
import argparse
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import asdict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--settle-seconds', type=float, default=.2)
    args = parser.parse_args()
    if not .2 <= args.settle_seconds <= 2:
        parser.error('settle-seconds must be between 0.2 and 2')
    if not args.execute:
        parser.error('--execute required after physical readiness confirmation')
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    calibration = json.loads(Path(r'C:\Users\Murphy\.cache\huggingface\lerobot\calibration\robots\so_follower\so101_follower_arm.json').read_text())
    bus = FeetechMotorsBus(port='COM4', motors={'gripper': Motor(6, 'sts3215', MotorNormMode.RANGE_0_100)})
    report = {'policy': 'scripted_gripper_micro_smoke', 'is_act': False, 'steps': [], 'status': 'failed'}
    report['settle_seconds'] = args.settle_seconds
    report['settling_samples'] = []
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path('artifacts/safety') / f'gripper-micro-{stamp}.json'
    torque_touched = False
    try:
        bus.connect()
        actual = asdict(bus.read_calibration()['gripper'])
        if actual != calibration['gripper']:
            raise RuntimeError('Calibration changed')
        if bus.read('Torque_Enable', 'gripper', normalize=False) != 0:
            raise RuntimeError('Expected initial torque off')
        if bus.read('Operating_Mode', 'gripper', normalize=False) != 0:
            raise RuntimeError('Expected position mode')
        start = bus.read('Present_Position', 'gripper', normalize=False)
        report['initial_raw'] = start
        if not actual['range_min']+20 < start < actual['range_max']-20:
            raise RuntimeError('Insufficient calibration margin')
        # Seed current position before enabling only the gripper.
        bus.write('Goal_Position', 'gripper', start, normalize=False)
        torque_touched = True
        bus.enable_torque('gripper')
        previous_time = time.perf_counter()
        for offset in list(range(1,9)) + list(range(7,-1,-1)):
            time.sleep(.1)
            now = time.perf_counter()
            if now-previous_time > .3:
                raise RuntimeError('Control interval exceeded 300 ms')
            measured = bus.read('Present_Position', 'gripper', normalize=False)
            if abs(measured-start) > 12:
                raise RuntimeError('Measured excursion exceeded 12 ticks')
            goal = start+offset
            bus.write('Goal_Position', 'gripper', goal, normalize=False)
            feedback = bus.read('Present_Position', 'gripper', normalize=False)
            report['steps'].append({'goal_raw':goal, 'feedback_raw':feedback, 'dt':now-previous_time})
            if abs(feedback-start) > 12:
                raise RuntimeError('Feedback excursion exceeded 12 ticks')
            previous_time = now
        settle_start = time.perf_counter()
        while time.perf_counter() - settle_start < args.settle_seconds:
            time.sleep(.1)
            position = bus.read('Present_Position', 'gripper', normalize=False)
            report['settling_samples'].append({'elapsed_s': time.perf_counter()-settle_start, 'position_raw': position})
            if abs(position-start) > 12:
                raise RuntimeError('Settling excursion exceeded 12 ticks')
        final = bus.read('Present_Position', 'gripper', normalize=False)
        report['final_raw'] = final
        report['observed_span_ticks'] = max(s['feedback_raw'] for s in report['steps'])-min(s['feedback_raw'] for s in report['steps'])
        if abs(final-start) > 4 or report['observed_span_ticks'] < 2:
            raise RuntimeError('Return or observable movement check failed')
        report['status'] = 'passed'
    except BaseException:
        report['error_trace'] = traceback.format_exc()
    finally:
        try:
            if bus.is_connected:
                if torque_touched:
                    bus.disable_torque('gripper')
                    report['final_torque'] = bus.read('Torque_Enable', 'gripper', normalize=False)
                    if report['final_torque'] != 0:
                        raise RuntimeError('Torque-off verification failed')
                bus.disconnect(disable_torque=False)
        except BaseException:
            report['cleanup_error'] = traceback.format_exc()
            report['status'] = 'failed'
        output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(output, report['status'])
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
