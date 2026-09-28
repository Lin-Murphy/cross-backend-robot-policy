"""Read-only preflight: inspect local resources and probe selected interpreters.

Never imports hardware runners or loads policy weights. Passing means the checks
listed here passed, not that inference, rendering or physical motion is validated.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
PROBE = """
import importlib.util,json,sys
names=json.loads(sys.argv[1])
missing=[name for name in names if importlib.util.find_spec(name) is None]
result={'missing':missing}
device=sys.argv[2]
if device.startswith('cuda') and 'torch' not in missing:
 import torch
 index=int(device.split(':',1)[1]) if ':' in device else 0
 result['device_available']=torch.cuda.is_available() and 0<=index<torch.cuda.device_count()
elif device not in ('cpu',''):
 result['device_available']=False
print(json.dumps(result))
"""


def probe_environment(python, modules, device='', env=None):
    completed = subprocess.run([python, '-B', '-c', PROBE, json.dumps(modules), device],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
    if completed.returncode:
        raise ValueError(f'{python}: dependency probe failed: {completed.stderr.strip()[-1500:]}')
    return json.loads(completed.stdout)


def check_config(config, output=None):
    checks = []
    def add(name, status, detail):
        checks.append({'check': name, 'status': status, 'detail': str(detail)})
    def inspect(name, operation):
        try:
            detail = operation()
            add(name, 'passed', detail)
            return detail
        except (OSError, ValueError, TypeError, KeyError, AttributeError, ET.ParseError,
                subprocess.SubprocessError) as exc:
            add(name, 'error', str(exc))
            return None
    def file(path):
        p = Path(path)
        if not p.is_file() or p.stat().st_size == 0:
            raise ValueError(f'Missing or empty file: {p}')
        return str(p)
    def document(path):
        value = json.loads(Path(file(path)).read_text())
        if not isinstance(value, dict):
            raise ValueError(f'Expected JSON object: {path}')
        return value
    def environment(name, python, modules, device='', env=None):
        def run():
            data = probe_environment(python, modules, device, env)
            if data['missing']:
                raise ValueError(f"{python}: install missing modules: {', '.join(data['missing'])}")
            if data.get('device_available') is False:
                raise ValueError(f'{python}: requested device unavailable or unsupported: {device}')
            return f'{python}: required modules present; device={device or "not checked"}'
        inspect(name, run)

    if output is not None:
        def destination():
            p = Path(output).absolute()
            if p.exists():
                raise ValueError(f'Output already exists; choose a new directory: {p}')
            parent = p.parent
            while not parent.exists():
                parent = parent.parent
            if not parent.is_dir() or not os.access(parent, os.W_OK):
                raise ValueError(f'Output parent is not writable: {parent}')
            return str(p)
        inspect('output', destination)

    backend = config['backend']
    python = config.get('python', os.environ.get('LEROBOT_PYTHON', sys.executable))
    if backend == 'so101':
        # The audited validator is the authority for hardware-specific resources.
        environment('validation environment', python, ['numpy'])
        add('hardware boundary', 'warning', 'Run --validate-only for audited profiles/checkpoints. '
            'Device access, calibration validity and motion approval are not checked here.')
    else:
        sim_python = config.get('sim_python', str(ROOT / ('.venv-aloha/bin/python' if backend == 'aloha' else '.venv-sim/bin/python')))
        environment('simulation environment', sim_python,
                    ['numpy', 'gym_aloha', 'gymnasium'] if backend == 'aloha' else ['numpy', 'mujoco'])
        base_env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
        site = None
        if backend == 'aloha':
            def simulation_site():
                return subprocess.run([sim_python, '-c', 'import site; print(site.getsitepackages()[0])'],
                    capture_output=True, text=True, timeout=30, check=True).stdout.strip()
            site = inspect('simulation import path', simulation_site)
        for policy in config['policies']:
            name = policy['name']
            env = dict(base_env)
            if backend == 'aloha' and site:
                env['PYTHONPATH'] = (str(ROOT / 'third_party/lerobot-dot-legacy') + os.pathsep if name == 'dot' else '') + site
            environment(f'{name} model environment', python,
                ['numpy', 'torch', 'lerobot', 'safetensors', 'gymnasium', 'gym_aloha'] if backend == 'aloha'
                else ['numpy', 'torch', 'lerobot', 'safetensors'], config.get('device', 'cpu'), env)
            checkpoint = Path(policy['checkpoint'])
            inspect(f'{name} weights', lambda: file(checkpoint / 'model.safetensors'))
            def metadata():
                cfg = document(checkpoint / 'config.json')
                if cfg.get('type') != name:
                    raise ValueError(f'{checkpoint}: expected policy type {name}, got {cfg.get("type")}')
                dimension = 14 if backend == 'aloha' else 6
                if cfg.get('input_features', {}).get('observation.state', {}).get('shape') != [dimension]:
                    raise ValueError(f'{name}: expected observation.state shape [{dimension}]')
                if cfg.get('output_features', {}).get('action', {}).get('shape') != [dimension]:
                    raise ValueError(f'{name}: expected action shape [{dimension}]')
                visual = {k: v['shape'] for k, v in cfg['input_features'].items() if k.startswith('observation.images.')}
                expected = {'observation.images.top': [3,480,640]} if backend == 'aloha' else (
                    {'observation.images.follower':[3,480,640], 'observation.images.camera2':[3,480,640]} if name == 'act'
                    else {**{f'observation.images.camera{i}':[3,256,256] for i in (1,2,3)},
                          **{f'observation.images.empty_camera_{i}':[3,480,640] for i in (0,1)}})
                if visual != expected:
                    raise ValueError(f'{name}: camera features/shapes do not match this example adapter; expected {expected}')
                if name != 'dot':
                    chunk, steps = cfg.get('chunk_size'), cfg.get('n_action_steps')
                    if type(chunk) is not int or type(steps) is not int or not 1 <= steps <= chunk:
                        raise ValueError(f'{name}: invalid chunk_size/n_action_steps')
                    if backend == 'mujoco' and (chunk != 50 or cfg.get('n_obs_steps') != 1):
                        raise ValueError(f'{name}: move-pot adapter requires chunk_size=50 and n_obs_steps=1')
                    if backend == 'mujoco' and name == 'act' and cfg.get('temporal_ensemble_coeff') is not None:
                        raise ValueError('ACT move-pot adapter does not support temporal ensembling')
                    for processor in ('policy_preprocessor.json', 'policy_postprocessor.json'):
                        data = document(checkpoint / processor)
                        if not isinstance(data.get('steps'), list):
                            raise ValueError(f'{processor}: missing processor steps')
                        for step in data['steps']:
                            if step.get('state_file'):
                                file(checkpoint / step['state_file'])
                    if name == 'smolvla':
                        pre = document(checkpoint / 'policy_preprocessor.json')
                        renames = [s['config'].get('rename_map', {}) for s in pre['steps'] if s.get('registry_name') == 'rename_observations_processor']
                        if renames != [{'observation.images.follower':'observation.images.camera1'}]:
                            raise ValueError('SmolVLA: expected saved follower-to-camera1 rename mapping')
                        if cfg.get('adapt_to_pi_aloha') or cfg.get('use_delta_joint_actions_aloha'):
                            raise ValueError('SmolVLA: incompatible action conversion for degree/calibrated move-pot adapter')
                return f'{checkpoint}: feature dimensions and processor references checked'
            inspect(f'{name} checkpoint metadata', metadata)
            if backend == 'aloha' and name == 'act':
                executable = config.get('act_evaluator', os.environ.get('LEROBOT_EVAL', str(Path(python).parent / 'lerobot-eval') if '/' in python else 'lerobot-eval'))
                inspect('ACT evaluator', lambda: file(shutil.which(executable) or executable))
        inspect('video encoder', lambda: file(shutil.which('ffmpeg') or 'ffmpeg'))
        add('runtime scope', 'warning', 'Checks do not load weights, render EGL frames, validate cached VLM/tokenizer assets, or prove inference compatibility.')

    if backend == 'mujoco':
        def scene():
            p = Path(file(config['scene']))
            root = ET.parse(p).getroot()
            compiler = root.find('compiler')
            meshdir = p.parent / (compiler.get('meshdir', '') if compiler is not None else '')
            for mesh in root.findall('./asset/mesh'):
                if mesh.get('file'): file(meshdir / mesh.get('file'))
            return str(p)
        inspect('scene and meshes', scene)
        def calibration():
            cal = document(config['calibration'])
            for joint in ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper'):
                low, high = cal[joint]['range_min'], cal[joint]['range_max']
                if type(low) is not int or type(high) is not int or not 0 <= low < high <= 4095:
                    raise ValueError(f'Invalid calibration range: {joint}')
            return 'Nominal mapping ranges valid; backend radians are explicitly converted to policy degrees/calibrated gripper units'
        inspect('calibration and units', calibration)
        for index, path in enumerate(config.get('initial_conditions', [])):
            inspect(f'initial condition {index}', lambda path=path: document(path) and str(path))
        if config.get('alignment_spec'):
            inspect('optional alignment specification', lambda: document(config['alignment_spec']) and config['alignment_spec'])
    add('action units', 'warning', 'Checkpoints do not generally encode physical joint units. Adapter unit contracts are enforced at execution; training-unit correctness is not inferred from tensor shape.')
    return {'schema_version': 1, 'status': 'failed' if any(c['status']=='error' for c in checks) else 'passed',
            'backend': backend, 'checks': checks, 'hardware_access': False, 'weights_loaded': False}
