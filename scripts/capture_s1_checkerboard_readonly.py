"""Capture checkerboard views from one camera only; no robot/motor imports."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import stat
import time
import math

import cv2
import numpy as np


def validate_camera_device(value):
    path=Path(value)
    direct=bool(re.fullmatch(r'/dev/video\d+',value))
    stable=(str(path.parent)=='/dev/v4l/by-id' and path.name.endswith('-video-index0'))
    if not (direct or stable):
        raise ValueError('device must be /dev/videoN or /dev/v4l/by-id/*-video-index0')
    resolved=path.resolve(strict=True)
    if not re.fullmatch(r'/dev/video\d+',str(resolved)) or not stat.S_ISCHR(resolved.stat().st_mode):
        raise ValueError('device link must resolve to a V4L2 video character device')
    return str(resolved)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--device',help='V4L2 camera device, for example /dev/video2')
    source.add_argument('--offline-image',type=Path,help='single-image no-device test')
    parser.add_argument('--camera',choices=('follower','camera2'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--views',type=int,default=20)
    parser.add_argument('--min-views',type=int,default=15)
    parser.add_argument('--width',type=int,default=640)
    parser.add_argument('--height',type=int,default=480)
    parser.add_argument('--columns',type=int,default=9)
    parser.add_argument('--rows',type=int,default=6)
    parser.add_argument('--automatic',action='store_true',help='Sample while the operator moves the board; no robot access')
    parser.add_argument('--duration-seconds',type=float,default=60)
    parser.add_argument('--interval-seconds',type=float,default=1)
    args=parser.parse_args()
    if not math.isfinite(args.duration_seconds) or not 1<=args.duration_seconds<=300 or not math.isfinite(args.interval_seconds) or not .2<=args.interval_seconds<=10:
        parser.error('invalid automatic capture duration or interval')
    if args.automatic and args.offline_image:parser.error('--automatic requires a camera device')
    if not 1<=args.min_views<=args.views<=40 or min(args.width,args.height)<100 or min(args.columns,args.rows)<3:
        parser.error('invalid image, pattern or view count')
    resolved_device=None
    if args.device:
        try:resolved_device=validate_camera_device(args.device)
        except (OSError,ValueError) as exc:parser.error(str(exc))
    if args.offline_image and args.views!=1:
        parser.error('offline image mode requires --views 1')
    args.output.mkdir(parents=True,exist_ok=False)
    accepted_dir=args.output/'accepted';rejected_dir=args.output/'rejected'
    accepted_dir.mkdir();rejected_dir.mkdir()
    records=[];accepted_corners=[];cap=None;status='incomplete'
    try:
        if args.offline_image:
            source_image=cv2.imread(str(args.offline_image),cv2.IMREAD_COLOR)
            if source_image is None:raise RuntimeError('offline image unreadable')
        else:
            cap=cv2.VideoCapture(args.device,cv2.CAP_V4L2)
            if not cap.isOpened():raise RuntimeError('camera cannot be opened')
            cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,args.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,args.height)
        capture_start=time.monotonic();next_capture=capture_start
        while len(accepted_corners)<args.views:
            if args.automatic and time.monotonic()-capture_start>=args.duration_seconds:break
            if args.offline_image:
                bgr=source_image.copy();stamp=time.monotonic_ns()
            else:
                if args.automatic:
                    remaining=next_capture-time.monotonic()
                    if remaining>0:time.sleep(remaining)
                    next_capture=time.monotonic()+args.interval_seconds
                else:
                    try:
                        answer=input(f'{args.camera}: move board to a new pose, Enter=capture, q=finish: ').strip().lower()
                    except EOFError:
                        break
                    if answer=='q':break
                ok=False;bgr=None
                for _ in range(5):ok,bgr=cap.read()
                stamp=time.monotonic_ns()
                if not ok or bgr is None:raise RuntimeError('camera frame read failed')
            if bgr.ndim!=3 or bgr.shape[2]!=3:raise RuntimeError('expected BGR image')
            if not args.offline_image and bgr.shape[:2]!=(args.height,args.width):
                raise RuntimeError(f'camera resolution mismatch: {bgr.shape[:2]}')
            gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            found,corners=cv2.findChessboardCornersSB(gray,(args.columns,args.rows))
            reason=None
            if not found:reason='corners_not_found'
            elif any(np.sqrt(np.mean(np.sum((corners-previous)**2,axis=2)))<2.0 for previous in accepted_corners):
                reason='near_duplicate_corner_layout'
            index=len(records);filename=f'view-{index:03d}.png'
            location=(rejected_dir if reason else accepted_dir)/filename
            if not cv2.imwrite(str(location),bgr):raise RuntimeError('PNG write failed')
            corner_position=None
            if found:
                xy=corners.reshape(-1,2)
                lower=xy.min(axis=0);upper=xy.max(axis=0)
                corner_position={'center_norm':((lower+upper)/2/[bgr.shape[1],bgr.shape[0]]).tolist(),
                                 'bbox_px':[float(lower[0]),float(lower[1]),float(upper[0]),float(upper[1])],
                                 'bbox_area_fraction':float(np.prod(upper-lower)/(bgr.shape[1]*bgr.shape[0]))}
            entry={'index':index,'path':str(location.resolve()),'sha256':hashlib.sha256(location.read_bytes()).hexdigest(),
                   'capture_monotonic_ns':stamp,'width':bgr.shape[1],'height':bgr.shape[0],
                   'corner_count':int(len(corners)) if found else 0,'corner_position':corner_position,
                   'accepted':reason is None,'reason':reason}
            records.append(entry)
            if reason is None:accepted_corners.append(corners)
            print(json.dumps({'accepted':reason is None,'accepted_views':len(accepted_corners),
                              'attempts':len(records),'reason':reason,'corner_position':corner_position},
                             ensure_ascii=False),flush=True)
            if args.offline_image:break
        status='candidate_capture_complete' if len(accepted_corners)>=args.min_views else 'insufficient_distinct_views'
    except Exception as exc:
        status='capture_error';records.append({'error':f'{type(exc).__name__}: {exc}'})
    finally:
        if cap is not None:cap.release()
        result={'status':status,'camera':args.camera,'device':args.device,'resolved_device':resolved_device,'offline_image':str(args.offline_image) if args.offline_image else None,
                'accepted_views':len(accepted_corners),'min_views':args.min_views,'attempts':len(records),
                'pattern_inner_corners':[args.columns,args.rows],'opencv_version':cv2.__version__,
                'automatic':args.automatic,'duration_limit_seconds':args.duration_seconds,'camera_read_only':True,'motor_or_serial_access':False,'calibration_accepted':False,'records':records}
        (args.output/'capture.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        print(json.dumps({'status':status,'accepted_views':len(accepted_corners),'output':str(args.output.resolve())},ensure_ascii=False))
    return 0 if status=='candidate_capture_complete' else 2


if __name__=='__main__':raise SystemExit(main())
