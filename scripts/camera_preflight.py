"""Bounded camera frame check with no robot connection."""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import cv2

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
output = Path('artifacts/safety') / f'camera-{stamp}.json'
report = {'camera_index': 1, 'status': 'failed', 'frames': [], 'hardware_dispatch': False}
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
try:
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise RuntimeError('Camera 1 did not open')
    for i in range(10):
        start = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f'Frame {i} failed')
        report['frames'].append({'shape': list(frame.shape), 'read_ms': (time.perf_counter()-start)*1000})
    image = output.with_suffix('.jpg')
    if not cv2.imwrite(str(image), frame):
        raise RuntimeError('Could not save frame')
    report['image'] = str(image)
    report['status'] = 'passed' if frame.shape == (480,640,3) else 'shape_mismatch'
except Exception as e:
    report['error'] = repr(e)
finally:
    cap.release()
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(output, report['status'])
raise SystemExit(0 if report['status'] == 'passed' else 1)
