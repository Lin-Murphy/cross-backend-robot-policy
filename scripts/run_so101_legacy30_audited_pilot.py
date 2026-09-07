"""One separately approved SO101/SmolVLA 30 fps development rollout. REAL MOTION."""
import argparse
from dataclasses import asdict,fields
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback
for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[key]='1'
os.environ['WANDB_MODE']='disabled'
_original_connect=socket.socket.connect
def _offline_connect(sock,address):
    if sock.family in (socket.AF_INET,socket.AF_INET6):raise RuntimeError('network_disabled')
    return _original_connect(sock,address)
socket.socket.connect=_offline_connect
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.legacy_bus_audit import install_legacy_bus_audit
from cross_backend.legacy_so101_sync_guard import LegacyTrialProfile,LegacySyncGuard,LegacyGuardRejected,NAMES
from cross_backend.execution_contract import ActionRequest, StopReceipt, execute_from_observation
from cross_backend.so101_lerobot_backend_adapter import SO101LeRobotBackendAdapter
from capture_move_pot_readonly import CAL,CAL_SHA
SMOL_PROFILE=ROOT/'configs/so101-smolvla-bidirectional-followwait18-proposal-20260927.json'
ACT_PROFILE=ROOT/'configs/so101-act-shared18-proposal-20260927.json'
ACT_FULL_CYCLE_PROFILE=ROOT/'configs/so101-act-full-cycle-reset-20260928.json'
ACT_PREOPEN_PROFILE=ROOT/'configs/so101-act-full-cycle-preopen-20260928.json'
PROFILE=SMOL_PROFILE


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def install_hold_before_finalize(strategy,hold,record,clock=time.perf_counter_ns):
    # Execute hold at policy-loop exit, before LeRobot's video-save finally block.
    original=strategy._policy_loop
    def protected(*args,**kwargs):
        try:return original(*args,**kwargs)
        finally:
            record({'event':'legacy_policy_loop_exit_before_finalize','host_ns':clock()})
            hold()
    strategy._policy_loop=protected



def install_shared_action_boundary(robot,guard,stop_current,record,trial_id,first_observation_path=None):
    """Wrap existing LeRobot send without a second observation or a new scheduler."""
    original_send=robot.send_action
    adapter=SO101LeRobotBackendAdapter(robot,guard,stop_current,
        approved_trial_id=trial_id,send_action=original_send)
    latest=[None];first_saved=[False]
    def on_observation(observation,started_ns):
        frame=adapter.capture(observation,started_ns=started_ns).validate(adapter.capabilities)
        latest[0]=frame
        if first_observation_path is not None and not first_saved[0]:
            import numpy as np
            np.savez(first_observation_path, **{
                'observation.state':np.asarray([observation[n+'.pos'] for n in NAMES],dtype=np.float32),
                'observation.images.follower':np.asarray(observation['follower']).copy(),
                'observation.images.camera2':np.asarray(observation['camera2']).copy()})
            first_saved[0]=True
            record({'event':'shared_first_live_observation_saved','observation_id':frame.observation_id,
                    'path':str(first_observation_path),'sha256':digest(first_observation_path),
                    'host_ns':time.perf_counter_ns()})
        record({'event':'shared_backend_observation','observation_id':frame.observation_id,
                'joint_names':frame.joint_names,'joint_units':frame.joint_units,
                'camera_frame_ids':frame.camera_frame_ids,'camera_capture_ns':frame.camera_capture_ns,
                'state_capture_ns':frame.state_capture_ns,'host_ns':time.perf_counter_ns()})
    def send(action):
        frame=latest[0]
        if frame is None:
            adapter.stop()
            raise RuntimeError('shared action without a fresh observation')
        if set(action)!=set(n+'.pos' for n in NAMES):
            adapter.stop()
            raise ValueError('shared action must contain exactly six SO101 joints')
        target=tuple(float(action[n+'.pos']) for n in NAMES)
        request=ActionRequest(frame.observation_id,frame.joint_names,frame.joint_units,target)
        _,receipt,stop=execute_from_observation(adapter,frame,request)
        record({'event':'shared_backend_action_receipt','observation_id':frame.observation_id,
                'receipt':asdict(receipt),'stop':None if stop is None else asdict(stop)})
        return action
    robot.send_action=send
    def restore():robot.send_action=original_send
    return on_observation,restore,adapter


def prepare_postconnect_registers(bus,direct_sync,expected_velocity,record):
    """Recover power-reset registers before a policy goal; keep target at present position."""
    torque={n:bus.read('Torque_Enable',n,normalize=False,num_retry=0) for n in NAMES}
    position={n:bus.read('Present_Position',n,normalize=False,num_retry=0) for n in NAMES}
    goal={n:bus.read('Goal_Position',n,normalize=False,num_retry=0) for n in NAMES}
    velocity={n:bus.read('Goal_Velocity',n,normalize=False,num_retry=0) for n in NAMES}
    record({'event':'legacy_postconnect_registers_before_prepare','torque':torque,
            'position':position,'goal':goal,'velocity':velocity})
    if any(torque[n]!=1 for n in NAMES):raise RuntimeError('torque_not_all_enabled')
    if any(abs(goal[n]-position[n])>20 for n in NAMES):
        ids={bus.motors[n].id:int(position[n]) for n in NAMES}
        result=direct_sync(42,2,ids,num_retry=0)
        record({'event':'legacy_startup_current_hold_transport_return','raw_ids_values':ids,
                'return_value':result,'host_ns':time.perf_counter_ns(),'per_motor_acknowledged':False})
        goal={n:bus.read('Goal_Position',n,normalize=False,num_retry=0) for n in NAMES}
        position={n:bus.read('Present_Position',n,normalize=False,num_retry=0) for n in NAMES}
        if any(abs(goal[n]-position[n])>20 for n in NAMES):
            raise RuntimeError('startup_current_hold_gap')
    if velocity!=expected_velocity:
        if any(velocity[n]!=0 for n in NAMES):raise RuntimeError('unexpected_goal_velocity')
        for n in NAMES:
            bus.write('Goal_Velocity',n,expected_velocity[n],normalize=False,num_retry=0)
        record({'event':'legacy_startup_velocity_set','expected':expected_velocity,
                'host_ns':time.perf_counter_ns()})
    actual={n:bus.read('Goal_Velocity',n,normalize=False,num_retry=0) for n in NAMES}
    if actual!=expected_velocity:raise RuntimeError('unexpected_goal_velocity')
    return actual


def install_first_policy_sample_resampler(engine,robot,guard,record,max_samples):
    """Bound stochastic first-action draws before any robot target is sent."""
    if type(max_samples) is not int or not 1<=max_samples<=8:
        raise ValueError('invalid first-action sample limit')
    original_get=engine.get_action
    first_pending=[True]
    def get_action(obs_frame):
        if not first_pending[0]:
            return original_get(obs_frame)
        for draw in range(1,max_samples+1):
            action=original_get(obs_frame)
            if action is None:
                return None
            values=action.detach().cpu().tolist()
            if len(values)!=len(NAMES) or not all(math.isfinite(float(v)) for v in values):
                reasons=['invalid_first_policy_action']
                raw=None
            else:
                raw_by_id=robot.bus._unnormalize({
                    robot.bus.motors[n].id:float(values[i]) for i,n in enumerate(NAMES)})
                ids={int(k):int(v) for k,v in raw_by_id.items()}
                reasons,raw=guard._goal_reasons(time.perf_counter_ns(),42,2,ids)
            record({'event':'shared_first_policy_sample_screen','sample':draw,
                    'max_samples':max_samples,'raw_candidate':raw,'reasons':list(reasons),
                    'hardware_dispatched':False,'host_ns':time.perf_counter_ns()})
            if not reasons:
                first_pending[0]=False
                return action
            if draw<max_samples:
                engine.reset()
        guard.reject(['first_policy_samples_outside_trial_profile'])
    engine.get_action=get_action
    def restore():engine.get_action=original_get
    return restore


def load_profile():
    profile=json.loads(PROFILE.read_text())
    full_cycle=profile.get('trial_kind')=='act_full_cycle_formal_attempt'
    if full_cycle:
        expected_model=ROOT/'artifacts/r1-act-formal-40000-run02/train/checkpoints/040000/pretrained_model'
        calibration=json.loads(CAL.read_text())
        lower=[calibration[n]['range_min'] for n in NAMES]
        upper=[calibration[n]['range_max'] for n in NAMES];upper[3]=3265
        if (profile.get('policy_type')!='act' or Path(profile['model_path']).resolve()!=expected_model.resolve()
            or profile['fps']!=30 or profile['max_episode_seconds']!=18 or profile['max_goal_packets']!=540
            or profile['raw_lower']!=lower or profile['raw_upper']!=upper
            or profile['max_feedback_raw']!=[calibration[n]['range_max'] for n in NAMES]
            or profile['max_from_feedback']!=[4095]*6 or profile['max_from_last_target']!=[4095]*6
            or profile['max_measured_rate_ticks_s']!=[1000000]*6
            or profile['max_gripper_follow_wait_ns']!=0 or profile['max_gripper_cancel_to_feedback_ticks']!=0
            or profile['max_first_action_samples']!=1 or profile['return_max_seconds']!=15
            or profile['return_nominal_ticks_s']!=180 or profile['return_tolerance_ticks']!=20
            or profile['auto_execute'] is not False or profile['calibration_sha256']!=CAL_SHA
            or digest(CAL)!=CAL_SHA):
            raise ValueError('unexpected ACT full-cycle profile')
        if profile.get('preopen_gripper_raw') is not None and (
            profile['preopen_gripper_raw']!=2309 or
            profile.get('preopen_nominal_ticks_s')!=120 or
            profile.get('preopen_tolerance_ticks')!=20):
            raise ValueError('unexpected ACT gripper preopen')
        keys=LegacyTrialProfile.__dataclass_fields__
        return profile,LegacyTrialProfile(**{k:tuple(profile[k]) if isinstance(profile[k],list) else profile[k]
            for k in keys}).validate()
    if profile['trial_kind']!='single_development_rollout_not_formal' or profile['fps']!=30 or \
       profile['max_episode_seconds']!=18 or profile['max_goal_packets']!=540 or \
       profile['max_gripper_follow_wait_ns']!=300_000_000 or \
       profile['max_gripper_cancel_to_feedback_ticks']!=60 or \
       profile['max_first_action_samples']!=(1 if profile.get('policy_type')=='act' else 4) or profile['raw_lower'][1]!=1000 or profile['auto_execute'] is not False or \
       profile['calibration_sha256']!=CAL_SHA or digest(CAL)!=CAL_SHA:
        raise ValueError('unexpected pilot identity or changed calibration')
    if profile.get('policy_type')=='act':
        expected_model=ROOT/'artifacts/r1-act-formal-40000-run02/train/checkpoints/040000/pretrained_model'
        if Path(profile['model_path']).resolve()!=expected_model.resolve() or \
           profile['raw_upper'][3]!=3265 or profile['max_feedback_raw'][3]!=3189 or \
           profile['max_from_feedback']!=[100,160,200,130,60,160] or \
           profile['raw_lower'][4]!=1750 or profile['max_from_last_target']!=[4095]*6 or profile.get('target_step_gate_enabled') is not False or \
           profile['max_measured_rate_ticks_s']!=[1000,2200,3000,1600,800,3000]:
            raise ValueError('unexpected ACT single-trial profile')
    keys=LegacyTrialProfile.__dataclass_fields__
    guard=LegacyTrialProfile(**{k:tuple(profile[k]) if isinstance(profile[k],list) else profile[k]
                                for k in keys}).validate()
    return profile,guard


def return_to_captured_start(bus,direct_sync,start_raw,profile,record,clock=time.perf_counter_ns,sleep=time.sleep):
    """Separate audited, paced raw return after a normally completed policy episode."""
    current=bus.sync_read('Present_Position',normalize=False,num_retry=0)
    target={n:int(start_raw[n]) for n in NAMES}
    delta=max(abs(target[n]-int(current[n])) for n in NAMES)
    seconds=max(4.0,delta/profile['return_nominal_ticks_s'])
    if seconds>profile['return_max_seconds']:
        raise RuntimeError('return_duration_exceeds_declared_limit')
    steps=max(1,math.ceil(seconds*30))
    record({'event':'formal_return_start','from_raw':current,'to_raw':target,
            'duration_seconds':seconds,'steps':steps,'host_ns':clock()})
    started=clock()
    last_send_ns=None
    for step in range(1,steps+1):
        deadline=started+round(step*seconds*1e9/steps)
        if last_send_ns is not None:deadline=max(deadline,last_send_ns+20_000_000)
        remaining=(deadline-clock())/1e9
        if remaining>0:sleep(remaining)
        values={bus.motors[n].id:round(int(current[n])+(target[n]-int(current[n]))*step/steps)
                for n in NAMES}
        for n in NAMES:
            motor=bus.motors[n].id
            if not profile['raw_lower'][motor-1]<=values[motor]<=profile['raw_upper'][motor-1]:
                raise RuntimeError('return_target_outside_calibration:'+n)
        result=direct_sync(42,2,values,num_retry=0)
        last_send_ns=clock()
        record({'event':'formal_return_goal_transport_return','step':step,'steps':steps,
                'raw_ids_values':values,'return_value':result,'host_ns':last_send_ns,
                'per_motor_acknowledged':False})
    sleep(1.0)
    final=bus.sync_read('Present_Position',normalize=False,num_retry=0)
    reached=all(abs(int(final[n])-target[n])<=profile['return_tolerance_ticks'] for n in NAMES)
    hold={bus.motors[n].id:int(final[n]) for n in NAMES}
    result=direct_sync(42,2,hold,num_retry=0)
    record({'event':'formal_return_final_hold_transport_return','raw_ids_values':hold,
            'return_value':result,'host_ns':clock(),'per_motor_acknowledged':False})
    record({'event':'formal_return_end','final_raw':final,'target_raw':target,
            'within_tolerance':reached,'host_ns':clock()})
    if not reached:raise RuntimeError('return_position_not_reached')
    return final


def preopen_gripper(bus,direct_sync,start_raw,profile,record,clock=time.perf_counter_ns,sleep=time.sleep):
    """Audited gripper-only opening before ACT; the final return still targets start_raw."""
    target=profile['preopen_gripper_raw']
    initial=int(start_raw['gripper'])
    if not 1877<=initial<target<=3209:raise RuntimeError('unexpected_gripper_preopen_range')
    seconds=max(2.5,(target-initial)/profile['preopen_nominal_ticks_s'])
    steps=math.ceil(seconds*30)
    record({'event':'formal_gripper_preopen_start','from_raw':dict(start_raw),
            'target_gripper_raw':target,'duration_seconds':seconds,'steps':steps,'host_ns':clock()})
    started=clock();last_send_ns=None
    for step in range(1,steps+1):
        deadline=started+round(step*seconds*1e9/steps)
        if last_send_ns is not None:deadline=max(deadline,last_send_ns+20_000_000)
        remaining=(deadline-clock())/1e9
        if remaining>0:sleep(remaining)
        values={bus.motors[n].id:int(start_raw[n]) for n in NAMES}
        values[bus.motors['gripper'].id]=round(initial+(target-initial)*step/steps)
        result=direct_sync(42,2,values,num_retry=0)
        last_send_ns=clock()
        record({'event':'formal_gripper_preopen_goal_transport_return',
                'step':step,'steps':steps,'raw_ids_values':values,
                'return_value':result,'host_ns':last_send_ns,'per_motor_acknowledged':False})
    sleep(0.5)
    feedback=bus.sync_read('Present_Position',normalize=False,num_retry=0)
    reached=abs(int(feedback['gripper'])-target)<=profile['preopen_tolerance_ticks']
    record({'event':'formal_gripper_preopen_end','feedback_raw':feedback,
            'within_tolerance':reached,'host_ns':clock()})
    if not reached:raise RuntimeError('gripper_preopen_not_reached')
    return feedback


def rollout_arguments(profile,out):
    cameras=profile['camera_paths']
    camera_arg=('{follower: {type: opencv, index_or_path: '+cameras['follower']+
        ', width: 640, height: 480, fps: 30}, camera2: {type: opencv, index_or_path: '+cameras['camera2']+
        ', width: 640, height: 480, fps: 30}}')
    return ['--robot.type=so101_follower','--robot.port='+profile['port'],
            '--robot.id=so101_follower_arm','--robot.use_degrees=true',
            '--robot.disable_torque_on_disconnect=false','--robot.cameras='+camera_arg,
            '--policy.path='+profile['model_path'],'--strategy.type=episodic',
            '--strategy.reset_to_initial_position=false','--inference.type=sync',
            '--dataset.repo_id=local/rollout_so101_legacy30_audit_single','--dataset.root='+str((out/'dataset').resolve()),
            '--dataset.fps=30','--dataset.episode_time_s='+str(profile['max_episode_seconds']),'--dataset.reset_time_s=0',
            '--dataset.num_episodes=1','--dataset.push_to_hub=false','--fps=30',
            '--interpolation_multiplier=1','--task=move pot','--return_to_initial_position=false',
            '--rename_map='+json.dumps({} if profile.get('policy_type')=='act' else {'observation.images.follower':'observation.images.camera1'}),
            '--play_sounds=false','--display_data=false','--device=cuda']


def validate_rollout_config(cfg,profile,out):
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.rollout.configs import EpisodicStrategyConfig
    if not isinstance(cfg.robot,SOFollowerRobotConfig) or cfg.robot.id!='so101_follower_arm' or \
       cfg.robot.port!=profile['port'] or cfg.robot.use_degrees is not True or \
       cfg.robot.disable_torque_on_disconnect is not False or cfg.robot.max_relative_target is not None:
        raise ValueError('unexpected SO101 robot configuration')
    if set(cfg.robot.cameras)!=set(profile['camera_paths']) or any(
        str(cfg.robot.cameras[n].index_or_path)!=profile['camera_paths'][n] or
        cfg.robot.cameras[n].width!=640 or cfg.robot.cameras[n].height!=480 or cfg.robot.cameras[n].fps!=30
        for n in profile['camera_paths']):raise ValueError('unexpected camera configuration')
    if cfg.policy.type!=profile.get('policy_type','smolvla') or Path(cfg.policy.pretrained_path).resolve()!=Path(profile['model_path']).resolve():
        raise ValueError('unexpected local policy')
    if not isinstance(cfg.strategy,EpisodicStrategyConfig) or \
       cfg.strategy.reset_to_initial_position is not False or cfg.inference.type!='sync':
        raise ValueError('unexpected strategy or inference')
    d=cfg.dataset
    if d is None or d.repo_id!='local/rollout_so101_legacy30_audit_single' or \
       d.num_episodes!=1 or d.fps!=30 or d.episode_time_s!=profile['max_episode_seconds'] or d.reset_time_s!=0 or \
       d.push_to_hub is not False or Path(d.root).resolve()!=(out/'dataset').resolve():
        raise ValueError('unexpected recording configuration')
    if cfg.fps!=30 or cfg.interpolation_multiplier!=1 or cfg.return_to_initial_position is not False or \
       cfg.teleop is not None or cfg.play_sounds is not False or cfg.task!='move pot' or cfg.device!='cuda':
        raise ValueError('unexpected runtime configuration')
    expected_map={} if profile.get('policy_type')=='act' else {'observation.images.follower':'observation.images.camera1'}
    if cfg.rename_map!=expected_map:
        raise ValueError('unexpected camera rename map')
    checkpoint=Path(profile['model_path'])
    saved=json.loads((checkpoint/'policy_preprocessor.json').read_text())
    saved_maps=[step['config']['rename_map'] for step in saved['steps']
                if step['registry_name']=='rename_observations_processor']
    features=json.loads((checkpoint/'config.json').read_text())['input_features']
    expected_features=({'observation.images.follower','observation.images.camera2'} if profile.get('policy_type')=='act'
                       else {'observation.images.camera1','observation.images.camera2'})
    if saved_maps!=[cfg.rename_map] or not expected_features.issubset(features):
        raise ValueError('checkpoint visual inputs differ from rollout mapping')


def execute(cfg,profile,limits,out,shared_boundary=False):
    import cv2
    from lerobot.robots.so_follower.so_follower import SOFollower
    from lerobot.rollout.context import build_rollout_context
    from lerobot.rollout.strategies import create_strategy
    from lerobot.utils.utils import init_logging
    from threading import Event
    validate_rollout_config(cfg,profile,out)
    init_logging()
    full_cycle=profile.get('trial_kind')=='act_full_cycle_formal_attempt'
    summary={'status':'failed','scope':('formal_full_cycle_attempt' if full_cycle else 'single_legacy30_development_pilot_not_formal'),
             'backend':'so101','policy':profile.get('policy_type','smolvla'),'task':'move pot','task_outcome':None,
             'formal_trial_count':0,'profile_sha256':digest(PROFILE),'calibration_sha256':CAL_SHA,
             'model_path':profile['model_path'],'source_exposure_timestamps_verified':False,
             'raw_goal_packets_transmitted':0,'per_motor_goal_acknowledgements_verified':False,
             'shared_boundary_enabled':shared_boundary}
    if full_cycle:summary['return_to_start_verified']=False
    stream=(out/'hardware-events.jsonl').open('x',buffering=1)
    def record(event):stream.write(json.dumps(event,allow_nan=False,default=str)+'\n')
    guard=LegacySyncGuard(limits,record,max_gripper_follow_wait_ns=profile['max_gripper_follow_wait_ns'],
        max_gripper_cancel_to_feedback_ticks=profile['max_gripper_cancel_to_feedback_ticks'])
    original_init=SOFollower.__init__;original_get=SOFollower.get_observation;original_calibrate=SOFollower.calibrate
    robot_holder=[None];original_sync=[None];ctx=None;strategy=None
    scene_checked=[False];hold_attempted=[False];shared_hooks=[None];first_sampler_restore=[None]
    def traced_init(robot,config):
        original_init(robot,config);robot_holder[0]=robot
        original_sync[0]=robot.bus._sync_write
        install_legacy_bus_audit(robot.bus,record,goal_guard=guard)
        def refresh_feedback():
            values=robot.bus.sync_read('Present_Position',normalize=False,num_retry=0)
            return {n:int(values[n]) for n in NAMES}
        guard.refresh_feedback=refresh_feedback
    def traced_get(robot):
        started=time.perf_counter_ns();obs=original_get(robot)
        values={robot.bus.motors[n].id:obs[n+'.pos'] for n in NAMES}
        raw_by_id=robot.bus._unnormalize(values)
        raw={n:int(raw_by_id[robot.bus.motors[n].id]) for n in NAMES}
        guard.observe(raw,observed_ns=started)
        if not scene_checked[0]:
            image=obs['follower'];x1,y1,x2,y2=profile['follower_clear_roi_xyxy']
            gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            fraction=float((gray[y1:y2,x1:x2]<100).mean())
            record({'event':'legacy_scene_clear_check','host_ns':time.perf_counter_ns(),
                    'fraction':fraction,'maximum':profile['max_roi_dark_fraction_below_gray_100']})
            if fraction>profile['max_roi_dark_fraction_below_gray_100']:
                guard.reject(['target_mat_obstruction_detected'])
            scene_checked[0]=True
        if shared_hooks[0] is not None:shared_hooks[0][0](obs,started)
        return obs
    def forbidden_calibrate(robot):raise RuntimeError('calibration_change_forbidden')
    SOFollower.__init__=traced_init;SOFollower.get_observation=traced_get;SOFollower.calibrate=forbidden_calibrate
    def hold_current(robot):
        bus=robot.bus
        if hold_attempted[0] or not bus.is_connected or original_sync[0] is None:return
        hold_attempted[0]=True
        positions=bus.sync_read('Present_Position',normalize=False,num_retry=0)
        ids={bus.motors[n].id:int(positions[n]) for n in NAMES}
        record({'event':'legacy_stop_hold_attempt','raw_ids_values':ids,'host_ns':time.perf_counter_ns()})
        result=original_sync[0](42,2,ids,num_retry=0)
        record({'event':'legacy_stop_hold_transport_return','return_value':result,'host_ns':time.perf_counter_ns(),
                'per_motor_acknowledged':False})
        summary['stop_hold_raw']=positions
    try:
        robot_speed_expected=dict(zip(NAMES,profile['expected_goal_velocity_raw']))
        ctx=build_rollout_context(cfg,Event())
        robot=ctx.hardware.robot_wrapper.inner
        hardware_calibration={n:asdict(v) for n,v in robot.bus.read_calibration().items()}
        if hardware_calibration!=json.loads(CAL.read_bytes()):
            raise RuntimeError('hardware_calibration_changed_after_connect')
        actual=prepare_postconnect_registers(robot.bus,original_sync[0],robot_speed_expected,record)
        torque={n:robot.bus.read('Torque_Enable',n,normalize=False,num_retry=0) for n in NAMES}
        position={n:robot.bus.read('Present_Position',n,normalize=False,num_retry=0) for n in NAMES}
        goal={n:robot.bus.read('Goal_Position',n,normalize=False,num_retry=0) for n in NAMES}
        record({'event':'legacy_postconnect_preflight','velocity_expected':robot_speed_expected,
                'velocity_actual':actual,'torque':torque,'position':position,'goal':goal})
        if actual!=robot_speed_expected:raise RuntimeError('unexpected_goal_velocity')
        if any(torque[n]!=1 for n in NAMES):raise RuntimeError('torque_not_all_enabled')
        if any(abs(goal[n]-position[n])>20 for n in NAMES):raise RuntimeError('stale_goal_gap')
        if not scene_checked[0]:raise RuntimeError('scene_check_missing')
        if full_cycle:
            summary['start_raw']=dict(position)
            summary['return_to_start_requested']=True
            record({'event':'formal_full_cycle_start_pose','raw':position,'host_ns':time.perf_counter_ns()})
            if profile.get('preopen_gripper_raw') is not None:
                summary['preopen_feedback_raw']=preopen_gripper(
                    robot.bus,original_sync[0],summary['start_raw'],profile,record)
                summary['gripper_preopen_verified']=True
        if shared_boundary:
            def shared_stop():
                hold_current(robot)
                if not hold_attempted[0] or 'stop_hold_raw' not in summary:
                    raise RuntimeError('current-position hold not confirmed by transport return')
                return StopReceipt(True,'current_position_sync_hold',1,'sync_transport_return')
            shared_hooks[0]=install_shared_action_boundary(robot,guard,shared_stop,record,
                trial_id=out.name,first_observation_path=out/'first-shared-observation.npz')
        if shared_boundary:
            first_sampler_restore[0]=install_first_policy_sample_resampler(
                ctx.policy.inference,robot,guard,record,profile['max_first_action_samples'])
        strategy=create_strategy(cfg.strategy)
        def immediate_hold():
            try:hold_current(robot)
            except BaseException as exc:
                summary['stop_hold_error']=repr(exc)
                record({'event':'legacy_immediate_hold_error','error':repr(exc),'host_ns':time.perf_counter_ns()})
        install_hold_before_finalize(strategy,immediate_hold,record)
        strategy.setup(ctx);strategy.run(ctx)
        if full_cycle:
            if guard.goal_packets<1:raise RuntimeError('formal_attempt_no_policy_goals')
            summary['formal_trial_count']=1
            summary['return_final_raw']=return_to_captured_start(
                robot.bus,original_sync[0],summary['start_raw'],profile,record)
            summary['return_to_start_verified']=True
        summary['status']='ended_unreviewed'
    except LegacyGuardRejected as exc:
        summary['status']='gate_rejected';summary['rejection_reasons']=list(exc.reasons)
    except BaseException as exc:
        summary['status']='aborted';summary['failure']=repr(exc);summary['traceback']=traceback.format_exc()
    finally:
        if full_cycle and guard.goal_packets>0:summary['formal_trial_count']=1
        robot=robot_holder[0]
        if robot is not None and robot.bus.is_connected:
            try:hold_current(robot)
            except BaseException as exc:summary['status']='aborted';summary['stop_hold_error']=repr(exc)
            if 'stop_hold_error' in summary:summary['status']='aborted'
            try:
                summary['final_raw']=robot.bus.sync_read('Present_Position',normalize=False,num_retry=0)
                summary['final_torque']={n:robot.bus.read('Torque_Enable',n,normalize=False,num_retry=0) for n in NAMES}
            except BaseException as exc:summary['final_read_error']=repr(exc)
        if strategy is not None and ctx is not None:
            try:strategy.teardown(ctx)
            except BaseException as exc:summary['status']='aborted';summary['teardown_error']=repr(exc)
        elif robot is not None:
            try:
                for camera in robot.cameras.values():
                    if camera.is_connected:camera.disconnect()
                if robot.bus.is_connected:robot.bus.disconnect(disable_torque=False)
            except BaseException as exc:summary['cleanup_error']=repr(exc)
        if first_sampler_restore[0] is not None:first_sampler_restore[0]()
        if shared_hooks[0] is not None:shared_hooks[0][1]()
        SOFollower.__init__=original_init;SOFollower.get_observation=original_get;SOFollower.calibrate=original_calibrate
        summary['raw_goal_packets_transmitted']=guard.goal_packets
        stream.close();(out/'summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
    print(json.dumps({k:summary.get(k) for k in ('status','raw_goal_packets_transmitted','rejection_reasons','failure')},indent=2),flush=True)
    if summary['status']!='ended_unreviewed':raise SystemExit(1)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--validate-only',action='store_true')
    mode.add_argument('--execute-approved-once',action='store_true')
    p.add_argument('--shared-boundary',action='store_true',help='new audited common observation/action hook; requires separate real-motion approval')
    p.add_argument('--policy',choices=('smolvla','act'),default='smolvla')
    p.add_argument('--full-cycle',action='store_true',help='ACT formal full-cycle profile; needs specific motion approval')
    p.add_argument('--preopen-gripper',action='store_true',help='ACT full-cycle trial with an audited gripper-only opening')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    if args.full_cycle and (args.policy!='act' or not args.shared_boundary):
        p.error('full-cycle requires ACT and the shared boundary')
    if args.preopen_gripper and not args.full_cycle:p.error('preopen requires full-cycle')
    PROFILE=ACT_PREOPEN_PROFILE if args.preopen_gripper else ACT_FULL_CYCLE_PROFILE if args.full_cycle else ACT_PROFILE if args.policy=='act' else SMOL_PROFILE
    if args.policy=='act' and args.execute_approved_once and not args.shared_boundary:
        p.error('ACT real rollout requires the shared boundary')
    profile,limits=load_profile()
    for source,name in ((PROFILE,'profile.json'),(Path(__file__),'executed_script.py'),
                        (ROOT/'src/cross_backend/legacy_bus_audit.py','legacy_bus_audit.py'),
                        (ROOT/'src/cross_backend/legacy_so101_sync_guard.py','legacy_so101_sync_guard.py')):
        (args.output/name).write_bytes(source.read_bytes())
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.configs import parser
    from lerobot.rollout.configs import RolloutConfig
    sys.argv=[sys.argv[0]]+rollout_arguments(profile,args.output)
    @parser.wrap()
    def _run(cfg:RolloutConfig):
        if args.validate_only:
            validate_rollout_config(cfg,profile,args.output)
            result={'status':'validated_no_hardware','profile_sha256':digest(PROFILE),
                    'model_config_type':cfg.policy.type,'robot_config_class':type(cfg.robot).__name__,
                    'fps':cfg.fps,'episodes':cfg.dataset.num_episodes,'push_to_hub':cfg.dataset.push_to_hub,
                    'auto_return':cfg.return_to_initial_position,'auto_reset':cfg.strategy.reset_to_initial_position,
                    'disconnect_disables_torque':cfg.robot.disable_torque_on_disconnect,
                    'cameras':list(cfg.robot.cameras),'rename_map':cfg.rename_map,
                    'shared_boundary_requested':args.shared_boundary,
                    'active_profile_path':str(PROFILE),'raw_lower':profile['raw_lower'],
                    'max_from_last_target':profile['max_from_last_target'],
                    'target_step_gate_enabled':profile.get('target_step_gate_enabled',True)}
            (args.output/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result,indent=2))
        else:execute(cfg,profile,limits,args.output,shared_boundary=args.shared_boundary)
    _run()
