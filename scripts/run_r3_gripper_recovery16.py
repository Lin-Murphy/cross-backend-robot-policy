"""One-time, separately approved gripper-only recovery probe."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cross_backend.gripper_recovery_microtrial import (NAMES, TrialStop, restricted_bus_class,
                                                        run_trial, verify_disabled_preflight)

CAL = Path('/home/murphy/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower_arm.json')
EXPECTED = 'f911b3b1e57ed0d513d27eeb7bbd743086b09a03f9babd4900b3a734c0594434'
PORT = '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B7B016049-if00'
OUT = ROOT / 'artifacts/r3-gripper-recovery16-20260926/hardware-attempt03-auto-torque-aware'


def main(out):
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    if hashlib.sha256(CAL.read_bytes()).hexdigest() != EXPECTED:
        raise TrialStop('Calibration file changed')
    cal = json.loads(CAL.read_text())
    if tuple(cal) != NAMES or any(cal[n]['id'] != i for i, n in enumerate(NAMES, 1)):
        raise TrialStop('Calibration identity mismatch')
    bus = restricted_bus_class(FeetechMotorsBus)(
        port=PORT,
        motors={n: Motor(i, 'sts3215', MotorNormMode.RANGE_0_100 if n == 'gripper'
                         else MotorNormMode.DEGREES) for i, n in enumerate(NAMES, 1)},
        calibration={n: MotorCalibration(**v) for n, v in cal.items()})
    result = {'status': 'preflight', 'calibration_sha256': EXPECTED,
              'policy_calls': 0, 'other_joint_goal_writes': 0, 'automatic_retry': False}
    stream = (out / 'events.jsonl').open('x')

    def record(event):
        stream.write(json.dumps(event, allow_nan=False) + '\n')
        stream.flush()

    def timeout(*_):
        raise TrialStop('Hard deadline; no further goal writes')

    signal.signal(signal.SIGALRM, timeout)
    signal.signal(signal.SIGTERM, timeout)
    signal.alarm(15)
    try:
        bus.connect()
        actual = {n: asdict(v) for n, v in bus.read_calibration().items()}
        if any(cal[n][k] != actual[n][k] for n in NAMES
               for k in ('id', 'homing_offset', 'range_min', 'range_max')):
            raise TrialStop('Hardware calibration mismatch')
        torque = {n: bus.read('Torque_Enable', n, normalize=False, num_retry=0) for n in NAMES}
        result['torque_before'] = torque
        if any(value != 0 for value in torque.values()):
            raise TrialStop('Expected all motors torque disabled')
        reference = {n: bus.read('Present_Position', n, normalize=False, num_retry=0) for n in NAMES}
        result['fresh_reference_raw'] = reference
        if any(not cal[n]['range_min'] <= reference[n] <= cal[n]['range_max'] for n in NAMES):
            raise TrialStop('Reference outside calibration')
        if reference['gripper'] + 16 > cal['gripper']['range_max']:
            raise TrialStop('Planned endpoint outside calibration')
        result['stationary_preflight'] = verify_disabled_preflight(bus, reference, cal, time.sleep)
        for count in (3, 2, 1):
            print(f'观察倒计时 {count}', flush=True)
            time.sleep(1)
        result['stationary_preflight_after_countdown'] = verify_disabled_preflight(bus, reference, cal, time.sleep)
        if any(bus.read('Torque_Enable', n, normalize=False, num_retry=0) != 0 for n in NAMES):
            raise TrialStop('Torque changed during countdown')
        record({'event': 'motion_phase_start', 'reference_raw': reference})
        signal.setitimer(signal.ITIMER_REAL, 8)
        result['trial'] = run_trial(bus, reference, cal, time.monotonic, time.sleep, record)
        signal.setitimer(signal.ITIMER_REAL, 0)
        result['status'] = ('sequence_complete_response_detected' if result['trial']['motion_response_detected']
                            else 'sequence_complete_no_position_response_detected')
    except BaseException as exc:
        result.update(status='stopped_failure', error_type=type(exc).__name__, error=str(exc),
                      traceback=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL, 3)
        if bus.is_connected:
            try:
                # This is the only cleanup write; never issue an unverified return goal.
                result['gripper_torque_before_cleanup'] = bus.read('Torque_Enable', 'gripper', normalize=False,
                                                                     num_retry=0)
                if result['gripper_torque_before_cleanup'] == 1 and getattr(bus, '_phase', None) in (
                        'initial_goal_attempted', 'initial_goal_confirmed',
                        'enable_attempted', 'enabled', 'fault', 'disable_attempted'):
                    record({'event': 'write_attempt', 'register': 'Torque_Enable', 'raw': 0, 'motor_id': 6})
                    bus.disable_gripper_only()
                    record({'event': 'write_acknowledged', 'register': 'Torque_Enable', 'raw': 0, 'motor_id': 6})
            except BaseException as exc:
                result['cleanup_error'] = str(exc)
                result['status'] = 'stopped_failure'
            try:
                result['torque_after'] = {n: bus.read('Torque_Enable', n, normalize=False, num_retry=0)
                                          for n in NAMES}
                result['final_raw'] = {n: bus.read('Present_Position', n, normalize=False, num_retry=0)
                                       for n in NAMES}
                if any(value != 0 for value in result['torque_after'].values()):
                    result['status'] = 'stopped_failure'
                    result['cleanup_error'] = 'Torque still enabled: manual power disconnect required'
            except BaseException as exc:
                result['cleanup_read_error'] = str(exc)
                result['status'] = 'stopped_failure'
            finally:
                try:
                    bus.disconnect(disable_torque=False)
                    result['serial_closed'] = not bus.is_connected
                except BaseException as exc:
                    result['disconnect_error'] = str(exc)
                    result['status'] = 'stopped_failure'
        signal.setitimer(signal.ITIMER_REAL, 0)
        stream.close()
        events = [json.loads(line) for line in (out / 'events.jsonl').read_text().splitlines()]
        result['write_attempts'] = sum(e['event'] == 'write_attempt' for e in events)
        result['acknowledged_writes'] = sum(e['event'] == 'write_acknowledged' for e in events)
        result['calibration_unchanged'] = hashlib.sha256(CAL.read_bytes()).hexdigest() == EXPECTED
        (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)
    if not result['status'].startswith('sequence_complete_'):
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-approved-once', action='store_true', required=True)
    parser.parse_args()
    prior = ROOT / 'artifacts/r3-gripper-recovery16-20260926/hardware-attempt02-after-zero-goals/summary.json'
    previous = json.loads(prior.read_text())
    attempts = ROOT / 'artifacts/r3-gripper-recovery16-20260926/hardware-attempt02-after-zero-goals/events.jsonl'
    events = [json.loads(line) for line in attempts.read_text().splitlines()]
    if previous['status'] != 'stopped_failure' or previous['error'] != 'Torque changed before enable':
        raise TrialStop('Previous attempt was not the observed auto-torque stop')
    if previous['write_attempts'] != 1 or previous['acknowledged_writes'] != 1:
        raise TrialStop('Previous attempt had unexpected write count')
    if [e.get('raw') for e in events if e['event'] == 'write_attempt'] != [previous['fresh_reference_raw']['gripper']]:
        raise TrialStop('Previous attempt had movement targets')
    emergency = json.loads((ROOT / 'artifacts/so101-backend-audit-20260926/gripper-emergency-disable.json').read_text())
    if emergency['status'] != 'verified_all_torque_off':
        raise TrialStop('Emergency torque closeout unverified')
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'executed_script.py').write_bytes(Path(__file__).read_bytes())
    (OUT / 'microtrial_source.py').write_bytes((ROOT / 'src/cross_backend/gripper_recovery_microtrial.py').read_bytes())
    main(OUT)
