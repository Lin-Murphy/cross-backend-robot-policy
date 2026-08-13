"""Bounded no-write capture with fixed, recovered move-pot device identities."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback

PORT = '/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B7B016049-if00'
CAL = Path('/home/murphy/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower_arm.json')
CAL_SHA = 'f911b3b1e57ed0d513d27eeb7bbd743086b09a03f9babd4900b3a734c0594434'
CAMERAS = {'follower': ('/dev/v4l/by-id/usb-Astra_Pro_HD_Camera_Astra_Pro_HD_Camera-video-index0', 30),
           'camera2': ('/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0', 60)}
NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')


def readonly_class(base):
    class ReadOnlyBus(base):
        def forbidden(self, *args, **kwargs):
            raise RuntimeError('Motor writes forbidden in read-only capture')
        write = sync_write = _write = _sync_write = forbidden
        write_calibration = configure_motors = enable_torque = disable_torque = forbidden
        _enable_torque = _disable_torque = forbidden
        def disconnect(self, disable_torque=False):
            if disable_torque:
                self.forbidden()
            return super().disconnect(disable_torque=False)
    return ReadOnlyBus


def main(out):
    import cv2
    import numpy as np
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    raw_cal = CAL.read_bytes()
    assert hashlib.sha256(raw_cal).hexdigest() == CAL_SHA, 'Local calibration changed'
    expected = json.loads(raw_cal)
    assert set(expected) == set(NAMES)
    assert all(expected[n]['id'] == i for i, n in enumerate(NAMES, 1))
    result = {'status': 'failed', 'port': PORT, 'calibration_sha256': CAL_SHA,
              'motor_writes_enabled': False, 'dispatches': 0, 'cameras': {}, 'samples': [],
              'timestamp_semantics': 'host read intervals; sensor exposure times unknown'}
    bus = readonly_class(FeetechMotorsBus)(port=PORT, motors={n: Motor(i, 'sts3215',
        MotorNormMode.RANGE_0_100 if n == 'gripper' else MotorNormMode.DEGREES)
        for i, n in enumerate(NAMES, 1)}, calibration={n: MotorCalibration(**v) for n, v in expected.items()})
    caps = {}
    try:
        print('Connecting direct read-only follower bus', flush=True)
        bus.connect()
        actual = {n: asdict(v) for n, v in bus.read_calibration().items()}
        result['hardware_calibration'] = actual
        result['calibration_differences'] = [{ 'motor': n, 'field': k, 'expected': expected[n][k], 'actual': actual[n][k]}
            for n in NAMES for k in ('id', 'homing_offset', 'range_min', 'range_max') if expected[n][k] != actual[n][k]]
        assert not result['calibration_differences'], 'Hardware calibration mismatch'
        result['torque_before'] = {n: bus.read('Torque_Enable', n, normalize=False) for n in NAMES}
        for name, (path, fps) in CAMERAS.items():
            cap = cv2.VideoCapture(path, cv2.CAP_V4L2); caps[name] = cap
            assert cap.isOpened(), 'Cannot open ' + path
            settings = {'fourcc': cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')),
                        'width': cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640),
                        'height': cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480),
                        'fps': cap.set(cv2.CAP_PROP_FPS, fps)}
            result['cameras'][name] = {'path': path, 'requested_fps': fps, 'set_results': settings,
                'reported_fps': cap.get(cv2.CAP_PROP_FPS), 'reported_width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                'reported_height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}
            for _ in range(5):
                ok, _ = cap.read(); assert ok, 'Camera warmup failed'
        for index in range(3):
            arrays, timing = {}, {}
            for name, cap in caps.items():
                start = time.perf_counter_ns(); ok, bgr = cap.read(); end = time.perf_counter_ns()
                assert ok and bgr.shape == (480, 640, 3), 'Invalid camera frame'
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                assert int(rgb.max()) > 10, 'Near-black camera frame'
                arrays['observation.images.' + name] = rgb
                timing[name] = {'read_start_ns': start, 'read_end_ns': end, 'sensor_capture_ns': None}
                assert cv2.imwrite(str(out / f'{index:02d}-{name}.png'), bgr)
            start = time.perf_counter_ns(); positions = bus.sync_read('Present_Position', normalize=False); end = time.perf_counter_ns()
            violations = [n for n in NAMES if not expected[n]['range_min'] <= positions[n] <= expected[n]['range_max']]
            sample = {'index': index, 'camera_reads': timing, 'state_read_start_ns': start, 'state_read_end_ns': end,
                      'raw_positions': positions, 'raw_position_violations': violations}
            result['samples'].append(sample)
            assert not violations, 'Raw position outside calibration range; refuse hidden clipping'
            normalized = bus._normalize({expected[n]['id']: positions[n] for n in NAMES})
            arrays['state'] = np.array([normalized[expected[n]['id']] for n in NAMES], dtype=np.float32)
            sample['state'] = arrays['state'].tolist()
            np.savez_compressed(out / f'sample-{index:02d}.npz', **arrays)
        result['torque_after'] = {n: bus.read('Torque_Enable', n, normalize=False) for n in NAMES}
        assert result['torque_before'] == result['torque_after'], 'Torque state changed externally'
        assert hashlib.sha256(CAL.read_bytes()).hexdigest() == CAL_SHA
        result['status'] = 'passed'
    except BaseException:
        result['failure'] = traceback.format_exc()
        raise
    finally:
        for cap in caps.values():
            cap.release()
        try:
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
        finally:
            (out / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('PASS: three dual-camera/state observations, no motor writes', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'executed_script.py').write_bytes(Path(__file__).read_bytes())
    def timed_out(*_):
        raise TimeoutError('Read-only capture exceeded 45 seconds')
    signal.signal(signal.SIGALRM, timed_out); signal.alarm(45)
    try:
        main(args.output)
    finally:
        signal.alarm(0)
