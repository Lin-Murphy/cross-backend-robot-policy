"""Offline checkerboard intrinsics candidate; never opens cameras or motors."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def fit_views(points, size, pattern, square_mm):
    if len(points) < 15:
        raise ValueError('need at least 15 distinct detected views')
    if not np.isfinite(square_mm) or square_mm <= 0:
        raise ValueError('measured square size must be positive and finite')
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[:pattern[0], :pattern[1]].T.reshape(-1, 2) * (square_mm / 1000)
    views = [np.asarray(p, np.float32).reshape(-1, 1, 2) for p in points]
    if any(len(p) != len(obj) or not np.isfinite(p).all() for p in views):
        raise ValueError('invalid corner array')
    # Reject duplicate corner layouts, including re-encoded copies of one image.
    if any(float(np.sqrt(np.mean(np.sum((views[i] - views[j]) ** 2, axis=2)))) < 2.0
           for i in range(len(views)) for j in range(i)):
        raise ValueError('repeated or near-duplicate corner layout; collect distinct poses')
    holdout = list(range(4, len(views), 5))
    train = [i for i in range(len(views)) if i not in holdout]
    rms, k, d, _, _ = cv2.calibrateCamera([obj] * len(train), [views[i] for i in train], size, None, None)
    if not np.isfinite(rms) or not np.isfinite(k).all() or not np.isfinite(d).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('nonfinite or invalid calibration')
    errors = []
    for i, p in enumerate(views):
        ok, r, t = cv2.solvePnP(obj, p, k, d)
        if not ok:
            raise ValueError(f'pose fit failed for view {i}')
        projected, _ = cv2.projectPoints(obj, r, t, k, d)
        error = float(np.sqrt(np.mean(np.sum((projected - p) ** 2, axis=2))))
        if not np.isfinite(error):
            raise ValueError('nonfinite reprojection error')
        errors.append({'view_index': i, 'split': 'holdout' if i in holdout else 'fit', 'rms_px': error})
    return {'status': 'candidate_not_accepted', 'image_size': list(size), 'pattern': list(pattern),
            'square_mm': square_mm, 'K': k.tolist(), 'distortion': d.tolist(), 'fit_rms_px': rms,
            'views': errors, 'holdout_note': 'intrinsics held fixed; board pose fitted on each held-out image',
            'physical_alignment_verified': False, 'extrinsics_verified': False}


def verify_capture_manifest(path, images_dir, camera):
    manifest = json.loads(path.read_text())
    if manifest.get('camera') != camera or manifest.get('status') != 'candidate_capture_complete':
        raise ValueError('capture manifest camera/status mismatch')
    records = manifest.get('records')
    if not isinstance(records, list):
        raise ValueError('capture manifest records missing')
    accepted = [r for r in records if r.get('accepted') is True]
    if len(accepted) != manifest.get('accepted_views') or len(accepted) < 15:
        raise ValueError('capture manifest accepted count mismatch')
    names = [Path(r.get('path', '')).name for r in accepted]
    if len(set(names)) != len(names):
        raise ValueError('capture manifest duplicate accepted image name')
    actual = {p.name for p in images_dir.iterdir()
              if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}}
    if set(names) != actual:
        raise ValueError('capture manifest/image set mismatch')
    for row, name in zip(accepted, names):
        expected = row.get('sha256')
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError('capture manifest image hash invalid')
        found = hashlib.sha256((images_dir/name).read_bytes()).hexdigest()
        if found != expected:
            raise ValueError(f'capture manifest image hash mismatch: {name}')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--images', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--camera', choices=['follower', 'camera2'], required=True)
    a.add_argument('--capture-manifest', type=Path, help='verify accepted photos against read-only capture.json')
    a.add_argument('--square-mm', type=float, required=True, help='measured printed square side')
    a.add_argument('--columns', type=int, default=9, help='inner corners')
    a.add_argument('--rows', type=int, default=6, help='inner corners')
    args = a.parse_args()
    if args.columns < 3 or args.rows < 3 or not np.isfinite(args.square_mm) or args.square_mm <= 0:
        a.error('invalid board dimensions')
    if not args.images.is_dir():
        a.error('image directory does not exist')
    args.output.mkdir(parents=True, exist_ok=False)
    records, points, hashes, size = [], [], set(), None
    pattern = (args.columns, args.rows)
    result = {'status': 'rejected', 'camera': args.camera, 'capture_manifest_verified': False}
    try:
        if args.capture_manifest is not None:
            result['capture_manifest_sha256'] = verify_capture_manifest(args.capture_manifest, args.images, args.camera)
            result['capture_manifest_verified'] = True
        for path in sorted(args.images.iterdir()):
            if path.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}:
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            record = {'path': str(path.resolve()), 'sha256': digest}
            records.append(record)
            im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if im is None:
                record['rejected'] = 'unreadable'
                continue
            shape = (im.shape[1], im.shape[0])
            if size is not None and shape != size:
                record['rejected'] = 'size_mismatch'
                raise ValueError('mixed image sizes; use one camera resolution')
            size = shape
            if digest in hashes:
                record['rejected'] = 'duplicate_bytes'
                continue
            hashes.add(digest)
            ok, corners = cv2.findChessboardCornersSB(im, pattern)
            if not ok:
                record['rejected'] = 'corners_not_found'
                continue
            record['view_index'] = len(points)
            record['corners_px'] = corners.reshape(-1, 2).tolist()
            points.append(corners)
        result.update(fit_views(points, size, pattern, args.square_mm))
    except Exception as exc:
        result.update(status='rejected', reason=f'{type(exc).__name__}: {exc}')
    finally:
        result.update(camera=args.camera, opencv_version=cv2.__version__, hardware_access=False)
        (args.output / 'detections.json').write_text(json.dumps(records, indent=2) + '\n')
        (args.output / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))
    return 0 if result['status'] == 'candidate_not_accepted' else 2


if __name__ == '__main__':
    raise SystemExit(main())
