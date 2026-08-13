"""Assemble the SO101 rollout context from a saved observation with no device access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time
from threading import Event

for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):
    os.environ[key]='1'
os.environ['WANDB_MODE']='disabled'
_original_connect=socket.socket.connect
def _offline_connect(sock,address):
    if sock.family in (socket.AF_INET,socket.AF_INET6):
        raise RuntimeError('network_disabled')
    return _original_connect(sock,address)
socket.socket.connect=_offline_connect
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from run_so101_legacy30_audited_pilot import load_profile,rollout_arguments,validate_rollout_config,install_hold_before_finalize
from cross_backend.legacy_so101_sync_guard import LegacyGuardRejected

class SavedObservationRobot:
    name='so_follower'
    robot_type='so_follower'
    def __init__(self,observations):
        self._observations=observations
        self.cameras={'follower':object(),'camera2':object()}
        self.observation_features={**{k:float for k in observations if k.endswith('.pos')},
                                   'follower':(480,640,3),'camera2':(480,640,3)}
        self.action_features={k:float for k in observations if k.endswith('.pos')}
        self.is_connected=False
        self.connect_count=0
        self.action_count=0
        self.reject_after=None
    def connect(self):
        self.connect_count+=1
        self.is_connected=True
    def get_observation(self):
        return self._observations.copy()
    def send_action(self,action):
        self.action_count+=1
        if self.reject_after is not None and self.action_count>=self.reject_after:
            raise LegacyGuardRejected(['offline_injected_rejection'])
        return action
    def disconnect(self):
        self.is_connected=False

def main(out,exercise_short_loop=False,inject_rejection=False,captured=None):
    import numpy as np
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.configs import parser
    from lerobot.rollout.configs import RolloutConfig
    import lerobot.rollout.context as context
    import argparse
    profile,_=load_profile()
    if captured is None:captured=ROOT/'artifacts/so101-legacy30-attempt02-preflight-20260927/sample-00.npz'
    captured=Path(captured).resolve()
    if not captured.is_relative_to(ROOT):raise ValueError('observation must be a local saved workspace sample')
    with np.load(captured) as x:
        state=x['state']
        names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
        observations={n+'.pos':float(v) for n,v in zip(names,state)}
        observations['follower']=x['observation.images.follower'].copy()
        observations['camera2']=x['observation.images.camera2'].copy()
    fake=SavedObservationRobot(observations)
    original=context.make_robot_from_config
    def fake_factory(_config):return fake
    context.make_robot_from_config=fake_factory
    sys.argv=[sys.argv[0]]+rollout_arguments(profile,out)
    try:
        @parser.wrap()
        def _check(cfg:RolloutConfig):
            validate_rollout_config(cfg,profile,out)
            ctx=context.build_rollout_context(cfg,Event())
            from lerobot.utils.constants import OBS_STR
            from lerobot.utils.feature_utils import build_dataset_frame
            import torch
            engine=ctx.policy.inference
            engine.reset();engine.start()
            processed=ctx.processors.robot_observation_processor(fake.get_observation())
            frame=build_dataset_frame(ctx.data.dataset_features,processed,prefix=OBS_STR)
            candidate=engine.get_action(frame)
            engine.stop()
            if candidate is None or tuple(candidate.shape)!=(6,) or not torch.isfinite(candidate).all():
                raise RuntimeError('offline six-joint policy candidate invalid')
            order=[]
            if exercise_short_loop or inject_rejection:
                from lerobot.rollout.strategies import create_strategy
                cfg.dataset.episode_time_s=15 if inject_rejection else 0.2
                strategy=create_strategy(cfg.strategy)
                if inject_rejection:
                    fake.reject_after=4
                    original_save=ctx.data.dataset.save_episode
                    def timed_save(*args,**kwargs):
                        order.append(('video_save_start',time.perf_counter_ns()))
                        return original_save(*args,**kwargs)
                    ctx.data.dataset.save_episode=timed_save
                    def fake_hold():order.append(('hold',time.perf_counter_ns()))
                    install_hold_before_finalize(strategy,fake_hold,lambda event:order.append((event['event'],event['host_ns'])))
                strategy.setup(ctx)
                try:
                    strategy.run(ctx)
                except LegacyGuardRejected as exc:
                    if not inject_rejection or exc.reasons!=('offline_injected_rejection',):raise
                finally:strategy.teardown(ctx)
                if inject_rejection:
                    holds=[t for n,t in order if n=='hold']
                    saves=[t for n,t in order if n=='video_save_start']
                    if len(holds)!=1 or not saves or holds[0]>=saves[0]:
                        raise RuntimeError('hold did not precede video save')
            result={'status':'injected_guard_hold_precedes_video_save' if inject_rejection else ('short_fake_rollout_complete' if exercise_short_loop else 'context_and_candidate_valid_without_hardware'),
                    'hold_save_order':order,
                    'fake_connects':fake.connect_count,
                    'candidate_degrees_or_gripper_percent':[float(v) for v in candidate],
                    'fake_actions':fake.action_count,'dataset_repo_id':ctx.data.dataset.repo_id,
                    'dataset_root':str(ctx.data.dataset.root),
                    'dataset_features':list(ctx.data.dataset_features),
                    'ordered_action_keys':ctx.data.ordered_action_keys,
                    'rename_map':cfg.rename_map,
                    'saved_observation_sha256':hashlib.sha256(captured.read_bytes()).hexdigest()}
            (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps(result,indent=2))
        _check()
    finally:
        context.make_robot_from_config=original
        fake.disconnect()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--exercise-short-loop',action='store_true')
    p.add_argument('--inject-rejection',action='store_true')
    p.add_argument('--observation',type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    try:main(args.output,args.exercise_short_loop,args.inject_rejection,args.observation)
    except BaseException:
        import traceback
        (args.output/'failure.txt').write_text(traceback.format_exc())
        raise
