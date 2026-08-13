"""Bounded local ACT-PushT trials (two by default, or an explicit trial plan).
No training or hardware APIs. Uses native saved processors and a verified chunk queue.
"""
import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'.sim-v1-deps'), str(ROOT/'src')]
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', SDL_VIDEODRIVER='dummy',
                  SDL_AUDIODRIVER='dummy', PYGAME_HIDE_SUPPORT_PROMPT='1')
import av
import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np
from PIL import Image
import torch
from cross_backend.chunk_policy import ACTChunkAdapter, PolicyAdapter, PolicyObservation


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item() if isinstance(x, np.generic) else str(x)))


class Video:
    def __init__(self, path):
        self.container = av.open(str(path), 'w')
        self.stream = self.container.add_stream('libx264', rate=10)
        self.stream.width = self.stream.height = 512
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {'crf': '20', 'preset': 'fast'}
        self.frames = 0

    def write(self, rgb):
        for packet in self.stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
            self.container.mux(packet)
        self.frames += 1

    def close(self):
        for packet in self.stream.encode(): self.container.mux(packet)
        self.container.close()


def observation(raw, step, captured_ns):
    return PolicyObservation(raw['pixels'].copy(), raw['agent_pos'].copy(), step, step/10, captured_ns)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, default=ROOT/'models/act-pusht-6d403b1')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--trial-plan', type=Path, help='Frozen JSON protocol with explicit seed/horizon pairs')
    args = ap.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    ckpt = args.checkpoint.resolve()
    proposal = json.loads((ROOT/'configs/sim-v1-pusht-proposal.json').read_text())
    expected_hash = proposal['act']['file']['lfs']['sha256']
    protocol = json.loads(args.trial_plan.read_text()) if args.trial_plan else None
    trials = protocol['trials'] if protocol else [{'seed':1000,'execute_steps':100},{'seed':1001,'execute_steps':100}]
    if protocol:
        expected_pairs = {(seed, horizon, delay) for seed in protocol['seeds']
                          for horizon in protocol['execution_conditions'] for delay in protocol.get('delay_conditions', [0])}
        actual_pairs = [(t['seed'], t['execute_steps'], t.get('delay_steps', 0)) for t in trials]
        if len(actual_pairs) != len(set(actual_pairs)) or set(actual_pairs) != expected_pairs:
            raise ValueError('Trial plan must contain every planned seed/horizon pair exactly once')
        if protocol['budget']['total_trials'] != len(trials) or protocol['budget']['max_episode_steps'] != 300:
            raise ValueError('Trial count/step budget does not match this bounded runner')
        if protocol['weight_sha256'] != expected_hash or protocol['predicted_steps'] != 100:
            raise ValueError('Protocol checkpoint/prediction length mismatch')
        if any(type(t['seed']) is not int or type(t['execute_steps']) is not int or not 1<=t['execute_steps']<=100 for t in trials):
            raise ValueError('Invalid seed or executed horizon')
        if any(type(t.get('delay_steps', 0)) is not int or not 0 <= t.get('delay_steps', 0) <= 300 for t in trials):
            raise ValueError('Invalid observation delay')
        if len(trials)*300 > protocol['budget']['max_total_environment_steps']:
            raise ValueError('Maximum total step budget exceeded')
        if protocol['fixed']['device'] != args.device:
            raise ValueError('Device differs from frozen protocol')
        save(out/'protocol.json', protocol)
    history_capacity = 1 + max(t.get('delay_steps', 0) for t in trials)
    result = {'schema_version': 2, 'history_capacity': history_capacity, 'started_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'episode_seeds': [t['seed'] for t in trials], 'trial_plan': trials,
              'protocol_sha256': sha(args.trial_plan) if args.trial_plan else None, 'max_steps_per_episode': 300,
              'checkpoint': str(ckpt), 'revision': proposal['act']['revision'], 'device': args.device,
              'no_hardware': True, 'no_training': True, 'network_offline': True,
              'mode': 'synchronous simulation; physics pauses for inference and recording',
              'nominal_control_hz': 10, 'episodes': [], 'timing_limit': 'simulation age and wall time are distinct; no real-time deadline claim'}
    save(out/'result.json', result)
    try:
        actual_hash = sha(ckpt/'model.safetensors')
        if actual_hash != expected_hash: raise ValueError('Checkpoint SHA256 mismatch')
        result['checkpoint_sha256'] = actual_hash
        result['checkpoint_files_before'] = {f.name: sha(f) for f in ckpt.iterdir() if f.is_file()}
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable; no silent device switch')
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        seed_all(1000)
        result['packages'] = {n: importlib.metadata.version(n) for n in ['torch','torchvision','lerobot','gymnasium','gym-pusht','pymunk','av','numpy']}
        result['source_sha256'] = {str(f): sha(f) for f in [Path(__file__), ROOT/'src/cross_backend/chunk_policy.py', ROOT/'.sim-v1-deps/gym_pusht/envs/pusht.py']}
        if protocol:
            result['source_sha256'][str(args.trial_plan.resolve())] = sha(args.trial_plan)
        if args.device.startswith('cuda'):
            result['gpu'] = torch.cuda.get_device_name(0)
            result['gpu_total_bytes'] = torch.cuda.get_device_properties(0).total_memory
        loaded = time.perf_counter_ns()
        adapter: PolicyAdapter = ACTChunkAdapter(ckpt, args.device)
        result['strict_load_passed'] = True
        result['load_wall_ms'] = (time.perf_counter_ns()-loaded)/1e6
        result['prediction_steps'] = adapter.predicted_steps
        result['execute_steps'] = trials[0]['execute_steps'] if len({t['execute_steps'] for t in trials})==1 else None
        result['checkpoint_execute_steps'] = adapter.execute_steps
        result['model_overrides'] = {'pretrained_backbone_weights': None, 'use_amp': False, 'device': args.device}
        result['native_queue_verifications'] = {}
        for episode, trial in enumerate(trials):
            seed, execute_steps = trial['seed'], trial['execute_steps']
            delay_steps = trial.get('delay_steps', 0)
            epdir = out/f'episode_{episode:03d}'; epdir.mkdir()
            summary = {'episode': episode, 'seed': seed, 'execute_steps':execute_steps, 'delay_steps':delay_steps, 'started': True, 'steps': 0, 'success': False,
                       'max_coverage': 0., 'final_coverage': None, 'termination': None, 'chunks': [],
                       'action_bounds': [0,512], 'clipping': 'none; refuse out-of-range command'}
            result['episodes'].append(summary)
            env = video = None
            try:
                seed_all(seed); adapter.set_execution_steps(execute_steps); queue = deque()
                env = gym.make('gym_pusht/PushT-v0', obs_type='pixels_agent_pos', render_mode='rgb_array',
                               observation_width=96, observation_height=96,
                               visualization_width=512, visualization_height=512, max_episode_steps=300)
                raw, _ = env.reset(seed=seed)
                obs = observation(raw, 0, time.perf_counter_ns())
                summary['initial_state'] = obs.state.tolist()
                summary['initial_rgb_sha256'] = hashlib.sha256(obs.image_rgb.tobytes()).hexdigest()
                if str(execute_steps) not in result['native_queue_verifications']:
                    verification = adapter.verify_native_queue(obs)
                    result['native_queue_verifications'][str(execute_steps)] = verification
                    if episode == 0:
                        result['native_queue_verification'] = verification
                    # Warm-up uses no env steps. Refresh wall timestamp on the unchanged observation.
                    obs = observation(raw, 0, time.perf_counter_ns())
                adapter.reset(); queue.clear()
                history = deque([obs], maxlen=history_capacity)
                summary['reset'] = {'policy': True, 'preprocessor': True, 'postprocessor': True, 'queue_empty': len(queue)==0}
                if args.device.startswith('cuda'): torch.cuda.reset_peak_memory_stats()
                video = Video(epdir/'rollout.mp4')
                video.write(env.render()); Image.fromarray(obs.image_rgb).save(epdir/'input_start.png')
                start = time.perf_counter_ns()
                with (epdir/'trace.jsonl').open('w') as log:
                    for step in range(300):
                        if not queue:
                            replan_step = step
                            input_obs = history[-1-min(delay_steps, step)]
                            assert input_obs.step == max(0, step-delay_steps)
                            chunk = adapter.predict_chunk(input_obs)
                            cid = len(summary['chunks'])
                            np.savez_compressed(epdir/f'chunk_{cid:03d}.npz', actions=chunk.actions,
                                                input_rgb=input_obs.image_rgb, input_state=input_obs.state,
                                                source_step=input_obs.step, replan_step=replan_step)
                            cm = {k:v for k,v in asdict(chunk).items() if k!='actions'}
                            cm.update(chunk_id=cid, predicted_shape=list(chunk.actions.shape), replan_step=replan_step,
                                      actual_input_delay_steps=replan_step-input_obs.step)
                            summary['chunks'].append(cm)
                            queue.extend((a.copy(), cid, i, chunk) for i,a in enumerate(chunk.actions[:adapter.execute_steps]))
                        pop_start = time.perf_counter_ns()
                        action, cid, offset, origin = queue.popleft()
                        pop_ms = (time.perf_counter_ns()-pop_start)/1e6
                        if action.shape != (2,) or not env.action_space.contains(action):
                            save(epdir/'rejected_action.json', {'step':step,'action':action,'chunk_id':cid,'offset':offset})
                            raise ValueError(f'Action rejected at step {step}: {action}')
                        dispatch_ns = time.perf_counter_ns()
                        next_raw,reward,terminated,truncated,info = env.step(action)
                        captured_ns = time.perf_counter_ns()
                        next_obs = observation(next_raw, step+1, captured_ns)
                        row = {'step':step,'state_before':obs.state.tolist(),'state_after':next_obs.state.tolist(),
                               'action':action.tolist(),'chunk_id':cid,'chunk_offset':offset,'queue_remaining':len(queue),
                               'action_source_step':origin.source_step,'latest_observation_step':obs.step,
                               'replan_step':replan_step,'configured_input_delay_steps':delay_steps,
                               'actual_input_delay_steps':replan_step-origin.source_step,
                               'observation_capture_monotonic_ns':obs.captured_monotonic_ns,
                               'source_capture_monotonic_ns':origin.source_captured_monotonic_ns,
                               'history_size':len(history),
                               'action_source_age_sim_s':step/10-origin.source_simulation_time_s,
                               'latest_state_age_sim_s':0.,'action_source_age_wall_ms':(dispatch_ns-origin.source_captured_monotonic_ns)/1e6,
                               'latest_observation_age_wall_ms':(dispatch_ns-obs.captured_monotonic_ns)/1e6,
                               'dispatch_monotonic_ns':dispatch_ns,'next_capture_monotonic_ns':captured_ns,
                               'queue_pop_ms':pop_ms,'env_step_wall_ms':(captured_ns-dispatch_ns)/1e6,
                               'reward':float(reward),'coverage':float(info['coverage']),
                               'success':bool(info['is_success']),'terminated':bool(terminated),'truncated':bool(truncated),
                               'observation_rgb_sha256':hashlib.sha256(obs.image_rgb.tobytes()).hexdigest()}
                        log.write(json.dumps(row)+'\n'); log.flush()
                        summary['steps'] += 1
                        summary['max_coverage'] = max(summary['max_coverage'], row['coverage'])
                        summary['final_coverage'] = row['coverage']
                        summary['success'] |= row['success']
                        video.write(env.render()); obs = next_obs; history.append(obs)
                        if terminated or truncated:
                            summary['termination'] = 'success' if row['success'] else 'terminated' if terminated else 'time_limit'
                            break
                summary['wall_ms_including_recording'] = (time.perf_counter_ns()-start)/1e6
                summary['simulation_duration_s'] = summary['steps']/10
                summary['queue_discarded_at_end'] = len(queue)
                Image.fromarray(obs.image_rgb).save(epdir/'input_end.png')
                print(f"episode {episode} seed {seed} execute {execute_steps} delay {delay_steps}: steps={summary['steps']} success={summary['success']} max_coverage={summary['max_coverage']:.4f}", flush=True)
            except Exception as ex:
                summary['termination'] = 'error'; summary['error'] = repr(ex)
                (epdir/'error.txt').write_text(traceback.format_exc())
            finally:
                if video is not None:
                    video.close(); summary['video_frames'] = video.frames
                if env is not None: env.close()
                adapter.reset()
                if args.device.startswith('cuda'):
                    summary['gpu_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
                    summary['gpu_peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
                save(epdir/'summary.json', summary); save(out/'result.json', result)
        result['checkpoint_files_after'] = {f.name: sha(f) for f in ckpt.iterdir() if f.is_file()}
        assert result['checkpoint_files_before'] == result['checkpoint_files_after']
        result['status'] = 'completed' if all(e['termination']!='error' for e in result['episodes']) else 'completed_with_errors'
        result['successes'] = sum(e['success'] for e in result['episodes'])
    except Exception as ex:
        result['status']='failed'; result['error']=repr(ex); (out/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        result['finished_utc']=datetime.now(timezone.utc).isoformat(); save(out/'result.json', result)

if __name__=='__main__': main()
