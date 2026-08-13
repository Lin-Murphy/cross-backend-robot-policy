"""Bounded PushT environment and saved-processor check; never loads a policy model.
Uses the existing LeRobot venv plus a project-local optional dependency directory.
All Hugging Face access is disabled during this check. Output must be new.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.sim-v1-deps'))
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['SDL_VIDEODRIVER']='dummy'
os.environ['SDL_AUDIODRIVER']='dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'

import gymnasium as gym
import gym_pusht  # noqa: F401 registers the actual environment
import numpy as np
from PIL import Image
import torch
from safetensors.numpy import load_file


def encode(x):
    if isinstance(x,np.ndarray):return x.tolist()
    if isinstance(x,np.generic):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);ap.add_argument('--metadata',type=Path,default=ROOT/'artifacts/sim-v1-preflight-20260921');args=ap.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    result={'started_utc':datetime.now(timezone.utc).isoformat(),'purpose':'environment and processor compatibility only; scripted commands, no policy model','hardware_access':False,'model_weights_loaded':False,'training':False,'packages':{n:importlib.metadata.version(n) for n in ['gym-pusht','gymnasium','pymunk','pygame','shapely','scikit-image','numpy','torch','lerobot']},'episodes':[],'processors':[]}
    traces=[];first_obs=None
    try:
        for repeat in range(2):
            env=gym.make('gym_pusht/PushT-v0',obs_type='pixels_agent_pos',render_mode='rgb_array',observation_width=96,observation_height=96,visualization_width=512,visualization_height=512,max_episode_steps=30)
            trace=[]
            try:
                obs,info=env.reset(seed=1000)
                if first_obs is None:first_obs={k:v.copy() for k,v in obs.items()};Image.fromarray(env.render()).save(out/'environment_start.png')
                assert obs['pixels'].shape==(96,96,3) and obs['pixels'].dtype==np.uint8
                assert obs['agent_pos'].shape==(2,) and env.action_space.shape==(2,)
                assert int(obs['pixels'].max())>10
                for step in range(30):
                    action=np.array([256+100*np.cos(step/5),256+100*np.sin(step/5)],dtype=np.float32)
                    assert env.action_space.contains(action)
                    t=time.perf_counter_ns();new,reward,terminated,truncated,info=env.step(action);duration=(time.perf_counter_ns()-t)/1e6
                    trace.append({'step':step,'sim_time_before_s':step/10,'sim_time_after_s':(step+1)/10,'state_before':obs['agent_pos'].copy(),'state_after':new['agent_pos'].copy(),'action':action,'reward':float(reward),'coverage':float(info['coverage']),'is_success':bool(info['is_success']),'terminated':bool(terminated),'truncated':bool(truncated),'step_wall_ms':duration,'rgb_sha256':hashlib.sha256(new['pixels'].tobytes()).hexdigest()})
                    obs=new
                    if terminated or truncated:break
                if repeat==0:Image.fromarray(env.render()).save(out/'environment_end.png')
                result['episodes'].append({'repeat':repeat,'seed':1000,'steps':len(trace),'action_source':'fixed circular position commands (not learned)','last_coverage':trace[-1]['coverage'],'terminated':trace[-1]['terminated'],'truncated':trace[-1]['truncated']})
                traces.append(trace)
            finally:env.close()
        equivalent=all(all(np.array_equal(v,b[k]) if isinstance(v,np.ndarray) else v==b[k] for k,v in a.items() if k!='step_wall_ms') for a,b in zip(*traces)) and len(traces[0])==len(traces[1])
        # Exact arrays/scalars and RGB hash equality; wall-clock timing naturally differs.
        assert equivalent,'seeded environment sequence differs'
        result['seeded_replay_exact']=equivalent
        from lerobot.configs import PreTrainedConfig
        from lerobot.policies import make_pre_post_processors
        from lerobot.policies.act.configuration_act import ACTConfig  # registers config
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        for name in ['aadarshram__act_pusht','naonaon__smolvla_pusht']:
            md=args.metadata/'model_metadata'/name;entry={'model':name,'weights_loaded':False}
            try:
                cfg=PreTrainedConfig.from_pretrained(str(md),local_files_only=True);cfg.device='cpu'
                entry['saved_input_features']={k:list(v.shape) for k,v in cfg.input_features.items()};entry['saved_action_shape']=list(cfg.action_feature.shape)
                entry['direct_feature_match']=list(cfg.action_feature.shape)==[2] and list(cfg.robot_state_feature.shape)==[2]
                entry['statistics']={f.name:{k:list(v.shape) for k,v in load_file(str(f)).items()} for f in md.glob('policy_*.safetensors')}
                pre,post=make_pre_post_processors(cfg,pretrained_path=str(md),preprocessor_overrides={'device_processor':{'device':'cpu'}},postprocessor_overrides={'device_processor':{'device':'cpu'}})
                action=torch.tensor([170.,280.],dtype=torch.float32)
                batch={'observation.image':torch.from_numpy(first_obs['pixels'].copy()).permute(2,0,1).float()/255,'observation.state':torch.from_numpy(first_obs['agent_pos'].copy()).float(),'action':action.clone(),'task':'Push the T-shaped block onto the T-shaped target.'}
                processed=pre(batch);roundtrip=post(processed['action'])
                error=float(torch.max(torch.abs(roundtrip.cpu()-action)))
                assert error<1e-3 and all(torch.isfinite(v).all() for v in processed.values() if isinstance(v,torch.Tensor))
                entry.update(processor_pass=True,action_roundtrip_max_error=error,processed_shapes={k:list(v.shape) for k,v in processed.items() if isinstance(v,torch.Tensor)},model_compatibility='not tested without model weights')
                pre.reset();post.reset()
            except Exception as ex:
                entry.update(processor_pass=False,error=repr(ex))
            result['processors'].append(entry)
        result['environment_pass']=True
    except Exception as ex:
        result['fatal_error']=repr(ex);raise
    finally:
        result['finished_utc']=datetime.now(timezone.utc).isoformat()
        (out/'result.json').write_text(json.dumps(result,indent=2,default=encode))
        (out/'scripted_environment_trace.json').write_text(json.dumps(traces,indent=2,default=encode))
    print(json.dumps(result,indent=2,default=encode))

if __name__=='__main__':main()
