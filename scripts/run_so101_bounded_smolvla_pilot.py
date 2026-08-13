"""One approved SO101/SmolVLA development episode with explicit raw bounds. REAL MOTION."""
import argparse
import os
for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[key]='1'
os.environ['WANDB_MODE']='disabled'
from dataclasses import asdict,fields
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback
import sys
import socket
_original_connect=socket.socket.connect
def _offline_connect(sock,address):
    if sock.family in (socket.AF_INET,socket.AF_INET6):raise RuntimeError('network_disabled')
    return _original_connect(sock,address)
socket.socket.connect=_offline_connect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.so101_dispatch_backend import (NAMES,PhysicalDispatchLimits,SO101BoundedDispatcher,
    DispatchRejected,PartialDispatchError,calibrated_to_raw,restricted_policy_bus_class)
from cross_backend.move_pot_policy import MovePotObservation,ReplayChunkQueue,CAMERAS
from capture_move_pot_readonly import PORT,CAL,CAL_SHA,CAMERAS as DEVICE_CAMERAS

PROFILE=ROOT/'configs/so101-smolvla-bounded-pilot-20260926.json'
CHECKPOINT=Path('/home/murphy/project/lerobot/outputs/train/smolvla_move_pot_dualcam_20260923_v3/checkpoints/040000/pretrained_model')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def load_profile(calibration):
    config=json.loads(PROFILE.read_text())
    names={f.name for f in fields(PhysicalDispatchLimits)}
    limits=PhysicalDispatchLimits(**{k:tuple(config[k]) if isinstance(config[k],list) else config[k] for k in names})
    limits.validate(calibration)
    if config['formal_trial'] is not False or config['action_hz_nominal']!=10 or config['execute_steps_per_chunk']!=10:
        raise ValueError('unexpected approved pilot protocol')
    return config,limits


def main(out):
    import cv2
    import numpy as np
    import torch
    from lerobot.motors import Motor,MotorCalibration,MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus
    from cross_backend.smolvla_move_pot import SmolVLAMovePotAdapter
    raw=CAL.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=CAL_SHA:raise RuntimeError('calibration changed')
    calibration=json.loads(raw)
    config,limits=load_profile(calibration)
    if not CHECKPOINT.is_dir():raise RuntimeError('cached checkpoint missing')
    out.joinpath('frames').mkdir()
    summary={'status':'failed','backend':'so101','task':'move pot','policy':'SmolVLA',
        'scope':'single_bounded_development_pilot_not_formal','task_outcome':None,'formal_trial_count':0,
        'applied_actions':0,'predictions':0,'frames_saved':0,'profile_sha256':digest(PROFILE),
        'calibration_sha256':CAL_SHA,'source_exposure_timestamps_verified':False}
    (out/'profile.json').write_bytes(PROFILE.read_bytes())
    (out/'dispatch_backend.py').write_bytes((ROOT/'src/cross_backend/so101_dispatch_backend.py').read_bytes())
    record_stream=(out/'hardware-events.jsonl').open('x',buffering=1)
    queue=ReplayChunkQueue(out/'queue-events.jsonl','so101-bounded-smolvla-pilot-20260926')
    def record(event):
        record_stream.write(json.dumps(event,allow_nan=False,default=str)+'\n')
    bus=restricted_policy_bus_class(FeetechMotorsBus,limits)(port=PORT,
        motors={n:Motor(i,'sts3215',MotorNormMode.RANGE_0_100 if n=='gripper' else MotorNormMode.DEGREES)
                for i,n in enumerate(NAMES,1)},
        calibration={n:MotorCalibration(**v) for n,v in calibration.items()})
    caps={};dispatcher=None
    def read(register):return {n:bus.read(register,n,normalize=False,num_retry=0) for n in NAMES}
    def capture(index):
        images={};times={}
        for camera,cap in caps.items():
            start=time.perf_counter_ns();ok,bgr=cap.read();end=time.perf_counter_ns()
            if not ok or bgr.shape!=(480,640,3):raise RuntimeError('camera_read_failed:'+camera)
            path=out/'frames'/f'{index:04d}-{camera}.jpg'
            if not cv2.imwrite(str(path),bgr,[cv2.IMWRITE_JPEG_QUALITY,90]):raise RuntimeError('frame_write_failed')
            images['observation.images.'+camera]=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
            times[camera]={'host_read_start_ns':start,'host_read_end_ns':end,
                           'sensor_exposure_ns':None,'sha256':digest(path)}
        start=time.perf_counter_ns();positions=read('Present_Position');end=time.perf_counter_ns()
        for n in NAMES:
            if not calibration[n]['range_min']<=positions[n]<=calibration[n]['range_max']:
                raise RuntimeError('raw_position_outside_calibration:'+n)
        values=bus._normalize({calibration[n]['id']:positions[n] for n in NAMES})
        state=np.array([values[calibration[n]['id']] for n in NAMES],dtype=np.float32)
        observation=MovePotObservation(images,state,f'so101-pilot:{index:04d}',
            {key:f'so101-pilot:{index:04d}:{key}' for key in CAMERAS},
            {key:None for key in CAMERAS},None,task='move pot')
        observation.validate()
        host_end=time.perf_counter_ns()
        record({'event':'observation','index':index,'observation_id':observation.observation_id,
                'raw_position':positions,'state':state.tolist(),'camera_read_times':times,
                'state_host_read_start_ns':start,'state_host_read_end_ns':end,
                'host_observation_end_ns':host_end})
        summary['frames_saved']+=1
        return observation,host_end,positions
    try:
        torch.set_num_threads(2)
        device='cuda' if torch.cuda.is_available() else 'cpu'
        adapter=SmolVLAMovePotAdapter(CHECKPOINT,device,seed=1000)
        adapter.set_execution_steps(10)
        summary['device']=device;summary['model_sha256']=adapter.checkpoint_manifest['model.safetensors']
        bus.connect()
        if {n:asdict(v) for n,v in bus.read_calibration().items()}!=calibration:
            raise RuntimeError('hardware_calibration_changed')
        for camera,(path,fps) in DEVICE_CAMERAS.items():
            cap=cv2.VideoCapture(path,cv2.CAP_V4L2);caps[camera]=cap
            if not cap.isOpened():raise RuntimeError('camera_open_failed:'+camera)
            cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,640);cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
            cap.set(cv2.CAP_PROP_FPS,fps)
            for _ in range(5):
                ok,_=cap.read()
                if not ok:raise RuntimeError('camera_warmup_failed:'+camera)
        for i in range(3):
            pos=read('Present_Position');goal=read('Goal_Position');torque=read('Torque_Enable')
            record({'event':'preflight','sample':i,'position':pos,'goal':goal,'torque':torque})
            if any(torque[n]!=1 for n in NAMES):raise RuntimeError('torque_not_all_enabled')
            for j,n in enumerate(NAMES):
                if abs(pos[n]-config['reference_raw'][j])>config['max_initial_deviation_ticks']:
                    raise RuntimeError('start_pose_changed:'+n)
                if not limits.raw_lower[j]<=pos[n]<=limits.raw_upper[j]:
                    raise RuntimeError('start_outside_development_profile:'+n)
                if abs(goal[n]-pos[n])>20:raise RuntimeError('stale_goal_gap:'+n)
            time.sleep(.1)
        first_obs,first_end,_=capture(0)
        x1,y1,x2,y2=config['follower_clear_roi_xyxy']
        follower_gray=cv2.cvtColor(first_obs.images_rgb['observation.images.follower'],cv2.COLOR_RGB2GRAY)
        dark_fraction=float((follower_gray[y1:y2,x1:x2]<100).mean())
        record({'event':'scene_clear_check','roi_xyxy':[x1,y1,x2,y2],
                'dark_fraction_below_gray_100':dark_fraction,
                'maximum':config['max_roi_dark_fraction_below_gray_100']})
        if dark_fraction>config['max_roi_dark_fraction_below_gray_100']:
            raise RuntimeError('target_mat_obstruction_detected')
        first_chunk=adapter.predict_chunk(first_obs)
        first_chunk.source['host_observation_end_ns']=first_end
        summary['predictions']+=1
        # Refuse the entire first 10-action prefix before any motor-affecting velocity write.
        previous=tuple(pos[n] for n in NAMES)
        for action in first_chunk.actions[:10]:
            goal=calibrated_to_raw(action,calibration)
            for j,n in enumerate(NAMES):
                if not limits.raw_lower[j]<=goal[j]<=limits.raw_upper[j] or \
                   abs(goal[j]-previous[j])>limits.max_target_step_ticks[j]:
                    raise DispatchRejected(['initial_prefix_outside_profile_or_step:'+n])
            previous=goal
        queue.enqueue(first_chunk,10)
        dispatcher=SO101BoundedDispatcher(bus,calibration,limits,record)
        dispatcher.configure_velocity()
        start_wall=time.monotonic();chunk_end=first_end;frame_index=1
        while summary['applied_actions']<config['max_applied_actions'] and \
              time.monotonic()-start_wall<config['max_wall_seconds']:
            if not queue.pending:
                obs,chunk_end,_=capture(frame_index);frame_index+=1
                chunk=adapter.predict_chunk(obs)
                chunk.source['host_observation_end_ns']=chunk_end
                summary['predictions']+=1
                queue.enqueue(chunk,10)
            if dispatcher.last_dispatch_ns is not None:
                due=dispatcher.last_dispatch_ns+105_000_000
                while time.perf_counter_ns()<due:time.sleep(min(.005,(due-time.perf_counter_ns())/1e9))
            # Capture current task evidence before every candidate selection.
            capture(frame_index);frame_index+=1
            source_end=queue.pending[0][3]['host_observation_end_ns']
            def validate(action,event):
                try:result=dispatcher.dispatch(action,source_end)
                except PartialDispatchError as exc:
                    event['hardware_dispatched']=bool(exc.acknowledged_motors)
                    event['partial_dispatch_motors']=list(exc.acknowledged_motors)
                    raise
                event['hardware_dispatched']=True
                event['dispatch_monotonic_ns']=result['dispatch_monotonic_ns']
                event['feedback_after']=result['feedback_after']
            queue.pop(context_validator=validate)
            summary['applied_actions']+=1
        summary['status']='ended_unreviewed'
    except DispatchRejected as exc:
        summary['status']='gate_rejected';summary['rejection_reasons']=list(exc.reasons)
        record({'event':'trial_gate_rejected','reasons':list(exc.reasons)})
    except BaseException as exc:
        summary['status']='aborted';summary['failure']=repr(exc);summary['traceback']=traceback.format_exc()
        record({'event':'trial_aborted','error':repr(exc)})
    finally:
        if dispatcher is not None and bus.is_connected:
            try:summary['stop_hold']=dispatcher.stop_hold()
            except BaseException as exc:
                summary['stop_hold_error']=repr(exc);summary['status']='aborted'
        try:
            if bus.is_connected:
                summary['final_raw']=read('Present_Position')
                summary['torque_final']=read('Torque_Enable')
                summary['goal_velocity_final']=read('Goal_Velocity')
                bus.disconnect(disable_torque=False)
        except BaseException as exc:
            summary['cleanup_error']=repr(exc);summary['status']='aborted'
        for cap in caps.values():cap.release()
        if queue.pending:queue.reset()
        queue.close();record_stream.close()
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary.get(k) for k in ('status','applied_actions','predictions','frames_saved','rejection_reasons','failure')},indent=2),flush=True)
    if summary['status']!='ended_unreviewed':raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute-approved-once',action='store_true',required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('pilot hard deadline')))
    signal.alarm(75)
    try:main(a.output)
    finally:signal.alarm(0)
