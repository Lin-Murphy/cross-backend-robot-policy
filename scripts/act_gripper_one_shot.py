"""One bounded ACT-derived gripper action on SO101, then return and torque off."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback

import cv2
import numpy as np
import torch
from cross_backend.local_metadata import load_policy_metadata
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors


NAMES = ['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
CALIBRATION = Path(r'C:\Users\Murphy\.cache\huggingface\lerobot\calibration\robots\so_follower\so101_follower_arm.json')
DATASET_ROOT = Path(r'C:\Users\Murphy\.cache\huggingface\lerobot\Murphy-Lin\lerobot101_dataset_a')
CHECKPOINT = Path(r'D:\project\lerobot-seeed\outputs\train\act_test\checkpoints\000020\pretrained_model')


def main():
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path('artifacts/safety') / f'act-gripper-one-shot-{stamp}.json'
    report = {'mode':'act_gripper_one_shot', 'status':'failed', 'joint':'gripper',
              'max_command_delta_normalized':1.0, 'max_raw_excursion_ticks':12,
              'return_tolerance_ticks':8, 'max_observation_age_ms':300,
              'other_joints_commanded':False,
              'episode_id': stamp,
              'episode_type': 'phase4_bounded_gripper_rollout',
              'outcome': 'failed'}
    bus = camera = None
    torque_touched = False
    try:
        raw_calibration = CALIBRATION.read_bytes()
        calibration = json.loads(raw_calibration)
        report['calibration_sha256'] = hashlib.sha256(raw_calibration).hexdigest()
        meta = load_policy_metadata(DATASET_ROOT)
        expected_names = [n+'.pos' for n in NAMES]
        if any(meta.features[k]['names'] != expected_names for k in ('observation.state','action')):
            raise RuntimeError('Dataset joint ordering mismatch')
        config = ACTConfig.from_pretrained(CHECKPOINT)
        config.pretrained_path, config.device = CHECKPOINT, 'cpu'
        policy = make_policy(config, ds_meta=meta).eval()
        pre, post = make_pre_post_processors(config, pretrained_path=str(CHECKPOINT), dataset_stats=meta.stats)
        bus = FeetechMotorsBus(port='COM4', motors={
            n:Motor(i,'sts3215',MotorNormMode.RANGE_0_100 if n=='gripper' else MotorNormMode.RANGE_M100_100)
            for i,n in enumerate(NAMES,1)},
            calibration={n:MotorCalibration(**calibration[n]) for n in NAMES})
        bus.connect()
        if {n:asdict(c) for n,c in bus.read_calibration().items()} != calibration:
            raise RuntimeError('Live calibration mismatch')
        initial_torque = {n:bus.read('Torque_Enable',n,normalize=False) for n in NAMES}
        report['initial_torque'] = initial_torque
        if any(initial_torque.values()):
            raise RuntimeError('Expected all torques off')
        camera = cv2.VideoCapture(1)
        # Let the camera negotiate its supported format (currently YUY2).
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,640); camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        for _ in range(3):
            ok,bgr = camera.read()
            if not ok: raise RuntimeError('Camera warmup failed')
        ok,bgr = camera.read(); image_time=time.perf_counter()
        if not ok or bgr.shape != (480,640,3): raise RuntimeError('Camera frame invalid')
        frame_path = output.with_suffix('.png')
        if not cv2.imwrite(str(frame_path), bgr):
            raise RuntimeError('Camera frame save failed')
        report['camera_frame'] = str(frame_path)
        raw_state = bus.sync_read('Present_Position',normalize=False)
        for n in NAMES:
            if not calibration[n]['range_min'] <= raw_state[n] <= calibration[n]['range_max']:
                raise RuntimeError(f'Position out of range: {n}')
        normalized = bus._normalize({calibration[n]['id']:raw_state[n] for n in NAMES})
        state = np.array([normalized[i] for i in range(1,7)],dtype=np.float32)
        state_time=time.perf_counter()
        rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
        batch=pre({'observation.state':torch.from_numpy(state).unsqueeze(0),
                   'observation.images.front':torch.from_numpy(rgb).permute(2,0,1).float().div(255).unsqueeze(0)})
        infer_start=time.perf_counter()
        with torch.inference_mode(): chunk=post(policy.predict_action_chunk(batch))
        infer_end=time.perf_counter()
        action=chunk[0,0].cpu().numpy()
        report.update(state=state.tolist(), raw_state=raw_state, action=action.tolist(),
                      chunk_shape=list(chunk.shape), inference_ms=(infer_end-infer_start)*1000,
                      observation_age_ms=(infer_end-state_time)*1000,
                      image_state_skew_ms=(state_time-image_time)*1000)
        if not np.isfinite(action).all() or chunk.shape[-1] != 6:
            raise RuntimeError('Invalid ACT output')
        if report['observation_age_ms'] > report['max_observation_age_ms']:
            raise RuntimeError('Observation too old for bounded one-shot')
        current_raw=bus.sync_read('Present_Position',normalize=False)
        current_norm=bus._normalize({calibration[n]['id']:current_raw[n] for n in NAMES})
        current=np.array([current_norm[i] for i in range(1,7)])
        report['pre_dispatch_state']=current.tolist()
        if np.max(np.abs(current-state)) > .5:
            raise RuntimeError('State drift before dispatch')
        # Use a slightly larger bounded fraction to overcome the observed
        # gripper deadband while retaining the normalized and raw tick caps.
        desired=float(current[5] + .2*(action[5]-current[5]))
        # Convert the raw 8-tick cap into the current calibration's normalized
        # units; a fixed normalized cap is not equivalent across calibrations.
        gripper_span = calibration['gripper']['range_max'] - calibration['gripper']['range_min']
        raw_cap_normalized = 8.0 * 100.0 / gripper_span
        candidate=float(np.clip(desired, current[5]-min(1.0, raw_cap_normalized),
                                current[5]+min(1.0, raw_cap_normalized)))
        candidate=float(np.clip(candidate,0,100))
        target_raw=bus._unnormalize({6:candidate})[6]
        start_raw=current_raw['gripper']
        if abs(target_raw-start_raw) > 8:
            raise RuntimeError('Raw target exceeds eight ticks')
        report.update(candidate_gripper=candidate,start_raw=start_raw,target_raw=target_raw)
        bus.write('Goal_Position','gripper',start_raw,normalize=False)
        torque_touched=True
        bus.enable_torque('gripper')
        bus.write('Goal_Position','gripper',target_raw,normalize=False)
        samples=[]
        for _ in range(5):
            time.sleep(.1); p=bus.read('Present_Position','gripper',normalize=False); samples.append(p)
            if abs(p-start_raw)>12: raise RuntimeError('Excursion exceeds 12 ticks')
        bus.write('Goal_Position','gripper',start_raw,normalize=False)
        returns=[]
        for _ in range(20):
            time.sleep(.1); p=bus.read('Present_Position','gripper',normalize=False); returns.append(p)
            if abs(p-start_raw)>12: raise RuntimeError('Return excursion exceeds 12 ticks')
        report.update(movement_samples_raw=samples,return_samples_raw=returns,final_raw=returns[-1],
                      observed_excursion_ticks=max(samples+returns)-min(samples+returns),
                      return_error_ticks=abs(returns[-1]-start_raw))
        if report['observed_excursion_ticks'] < 2: raise RuntimeError('No observable motion')
        if report['return_error_ticks'] > 8: raise RuntimeError('Return tolerance failed')
        report['status']='passed'
        report['outcome']='bounded_rollout_pass'
    except BaseException:
        report['error_trace']=traceback.format_exc()
        report['outcome']='bounded_rollout_failed'
    finally:
        if camera is not None: camera.release()
        if bus is not None and bus.is_connected:
            try:
                if torque_touched: bus.disable_torque('gripper')
                report['final_torque']={n:bus.read('Torque_Enable',n,normalize=False) for n in NAMES}
                if any(report['final_torque'].values()): raise RuntimeError('Final torque not zero')
                bus.disconnect(disable_torque=False)
            except BaseException:
                report['cleanup_error']=traceback.format_exc(); report['status']='failed'
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(output,report['status'])
    return 0 if report['status']=='passed' else 1


if __name__=='__main__': raise SystemExit(main())
