"""One evaluation lifecycle over native policy/backend runners (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
SUPPORTED = {"aloha": ("transfer_cube", {"act", "dot"}),
             "mujoco": ("move_pot", {"act", "smolvla"}),
             "so101": ("move_pot", {"act", "smolvla"})}
FIELDS = {"schema_version", "backend", "task", "policies", "episodes", "first_seed",
          "python", "sim_python", "act_evaluator", "device", "scene", "calibration",
          "initial_conditions", "observation_delay_frames", "alignment_spec", "max_control_ticks"}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def load_config(path):
    path = Path(path).resolve()
    c = json.loads(path.read_text())
    if not isinstance(c, dict) or set(c) - FIELDS:
        raise ValueError('Unknown evaluation configuration fields')
    if type(c.get('schema_version')) is not int or c['schema_version'] != 1:
        raise ValueError('schema_version must be 1')
    backend = c.get('backend')
    if backend not in SUPPORTED or c.get('task') != SUPPORTED[backend][0]:
        raise ValueError('Unsupported backend/task combination')
    policies = c.get('policies')
    if not isinstance(policies, list) or not policies:
        raise ValueError('policies must be a nonempty list')
    names = []
    for p in policies:
        if not isinstance(p, dict) or set(p) - {'name', 'checkpoint'}:
            raise ValueError('Each policy declares name and optional checkpoint')
        if p.get('name') not in SUPPORTED[backend][1]:
            raise ValueError('Unsupported policy/backend combination')
        names.append(p['name'])
        if backend != 'so101' and not p.get('checkpoint'):
            raise ValueError('Simulation policies require an explicit local checkpoint')
        if backend == 'so101' and 'checkpoint' in p:
            raise ValueError('SO101 checkpoint is fixed by the existing audited profile')
    if len(names) != len(set(names)):
        raise ValueError('Duplicate policies')
    explicit_seed = 'first_seed' in c
    c.setdefault('episodes', 1)
    c.setdefault('first_seed', 1000)
    for key, minimum in [('episodes', 1), ('first_seed', 0)]:
        if type(c[key]) is not int or c[key] < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    specific = {'aloha': {'sim_python', 'act_evaluator'},
                'mujoco': {'sim_python', 'scene', 'calibration', 'initial_conditions', 'observation_delay_frames', 'alignment_spec', 'max_control_ticks'},
                'so101': set()}
    all_specific = set.union(*specific.values())
    if (set(c) & all_specific) - specific[backend]:
        raise ValueError('Configuration contains options for another backend')
    if backend == 'so101' and (len(policies) != 1 or c['episodes'] != 1):
        raise ValueError('SO101 supports one separately approved policy episode per invocation')
    if backend == 'so101' and ('device' in c or explicit_seed):
        raise ValueError('SO101 device and seed are controlled by the audited runner')
    if 'device' in c and (not isinstance(c['device'], str) or not c['device']):
        raise ValueError('device must be a nonempty string')
    def resolve(value):
        if not isinstance(value, str) or not value:
            raise ValueError('Paths must be nonempty strings')
        p = Path(value).expanduser()
        return str((path.parent / p).resolve())
    for p in policies:
        if 'checkpoint' in p:
            p['checkpoint'] = resolve(p['checkpoint'])
    for key in ('scene', 'calibration', 'alignment_spec'):
        if key in c:
            c[key] = resolve(c[key])
    for key in ('python', 'sim_python', 'act_evaluator'):
        if key in c:
            value = c[key]
            if not isinstance(value, str) or not value:
                raise ValueError(f'{key} must be a nonempty executable name or path')
            c[key] = os.path.abspath(path.parent / Path(value).expanduser()) if '/' in value else value
    if backend == 'mujoco':
        if not c.get('calibration'):
            raise ValueError('MuJoCo move_pot requires an explicit calibration file')
        c.setdefault('scene', str(ROOT / 'assets/so101/tape-base-offset.xml'))
        delay = c.setdefault('observation_delay_frames', 0)
        if type(delay) is not int or delay not in (0, 1):
            raise ValueError('MuJoCo observation_delay_frames must be 0 or 1')
        ticks = c.get('max_control_ticks', 600)
        if type(ticks) is not int or not 1 <= ticks <= 600:
            raise ValueError('max_control_ticks must be an integer in 1..600')
        if c.get('alignment_spec') and not c.get('initial_conditions'):
            raise ValueError('Measured alignment requires explicit initial conditions')
        if 'initial_conditions' in c:
            initial = c['initial_conditions']
            if not isinstance(initial, list) or len(initial) != c['episodes']:
                raise ValueError('initial_conditions must contain one C initial file per episode')
            c['initial_conditions'] = [resolve(p) for p in initial]
    return c


def make_plan(config, output, *, approved=False, validate_only=False):
    c = config
    output = Path(output).resolve()
    backend = c['backend']
    python = c.get('python', os.environ.get('LEROBOT_PYTHON', sys.executable))
    if backend == 'so101' and not (approved or validate_only):
        raise ValueError('SO101 requires --validate-only or --execute-approved-once with an on-site approved plan')
    if backend != 'so101' and (approved or validate_only):
        raise ValueError('Hardware execution/validation options only apply to SO101')
    jobs = []
    for policy in c['policies']:
        name = policy['name']
        for index in range(1 if backend == 'aloha' else c['episodes']):
            job_id = name if backend == 'aloha' else f'{name}-{index:03d}'
            raw = output / 'raw' / job_id
            seed = c['first_seed'] + index
            env = {'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
                   'TRANSFORMERS_OFFLINE': '1', 'WANDB_MODE': 'disabled',
                   'PYTHONDONTWRITEBYTECODE': '1', 'MUJOCO_GL': 'egl'}
            if backend == 'aloha':
                device = c.get('device', 'cpu')
                common = [f"--policy.path={policy['checkpoint']}", '--env.type=aloha',
                          '--env.task=AlohaTransferCube-v0', '--env.episode_length=400',
                          f"--eval.n_episodes={c['episodes']}", '--eval.batch_size=1',
                          '--eval.use_async_envs=false', f'--output_dir={raw}', f'--seed={seed}']
                executable = c.get('act_evaluator', os.environ.get('LEROBOT_EVAL',
                                  str(Path(python).parent / 'lerobot-eval') if '/' in python else 'lerobot-eval'))
                argv = ([executable, f'--policy.device={device}'] if name == 'act' else
                        [python, '-B', str(ROOT / 'third_party/lerobot-dot-legacy/lerobot/scripts/eval.py'),
                         f'--device={device}', '--use_amp=false']) + common
            elif backend == 'mujoco':
                argv = [c.get('sim_python', str(ROOT / '.venv-sim/bin/python')), '-B',
                        str(ROOT / 'scripts/run_s1_policy_development.py'),
                        '--output', str(raw), '--policy', name, '--scene', c['scene'],
                        '--calibration', c['calibration'], '--policy-python', python,
                        '--checkpoint', policy['checkpoint'], '--device', c.get('device', 'cpu'),
                        '--seed', str(seed), '--trial-prefix', f'eval-{job_id}',
                        '--observation-delay-frames', str(c['observation_delay_frames'])]
                if 'max_control_ticks' in c:
                    argv += ['--max-control-ticks', str(c['max_control_ticks'])]
                if 'alignment_spec' in c:
                    argv += ['--alignment-spec', c['alignment_spec']]
                if 'initial_conditions' in c:
                    argv += ['--initial-condition', c['initial_conditions'][index]]
            else:
                argv = [python, '-B', str(ROOT / 'scripts/run_so101_legacy30_audited_pilot.py'),
                        '--output', str(raw), '--policy', name, '--shared-boundary',
                        '--validate-only' if validate_only else '--execute-approved-once']
            jobs.append({'id': job_id, 'policy': name, 'episode_index': index,
                         'seed': seed if backend != 'so101' else None,
                         'argv': argv, 'environment': env, 'raw_directory': str(raw)})
    return {'schema_version': 1, 'backend': backend, 'task': c['task'],
            'mode': 'validate' if validate_only else 'evaluate', 'config': c, 'jobs': jobs}


def normalize(job, config):
    """Keep task outcomes independent of evaluator/process completion."""
    raw = Path(job['raw_directory'])
    backend, policy = config['backend'], job['policy']
    records = []
    def record(index, status, outcome, source, **metrics):
        return {'policy': policy, 'episode_index': index, 'task_status': status,
                'outcome': outcome, 'outcome_source': source if outcome is not None else None,
                'metrics': metrics}
    if backend == 'aloha':
        path = raw / 'eval_info.json'
        data = json.loads(path.read_text())
        if policy == 'act':
            if len(data['per_task']) != 1:
                raise ValueError('Expected a single ALOHA task')
            metrics = data['per_task'][0]['metrics']
            rewards = metrics['max_rewards']
            native = metrics['successes']
        else:
            episodes = data['per_episode']
            if [e['seed'] for e in episodes] != list(range(config['first_seed'], config['first_seed'] + config['episodes'])):
                raise ValueError('DOT episode seeds mismatch')
            rewards = [e['max_reward'] for e in episodes]
            native = [e['success'] for e in episodes]
        if len(rewards) != config['episodes'] or len(native) != len(rewards):
            raise ValueError('Evaluator episode count mismatch')
        for index, (reward, success) in enumerate(zip(rewards, native)):
            reward = float(reward)
            if not math.isfinite(reward):
                raise ValueError('Nonfinite task reward')
            records.append(record(index, 'success' if reward >= 4 else 'failed', reward >= 4,
                                  'episode_max_reward>=4', max_reward=reward, native_success=success))
    elif backend == 'mujoco':
        path = raw / policy / 'summary.json'
        data = json.loads(path.read_text())
        status = data['status']
        known_failure = {'timeout', 'collision', 'drop', 'placement_outside', 'incomplete'}
        known_unknown = {'uncertain', 'not_started', 'stopped_error_or_gate', 'input_error',
                         'simulation_error', 'invalid_initial_state'}
        if status not in known_failure | known_unknown | {'success'}:
            raise ValueError(f'Unknown MuJoCo status: {status}')
        outcome = True if status == 'success' else False if status in known_failure else None
        records.append(record(job['episode_index'], status, outcome, 'TapeTaskEvaluator',
                              selected_actions=data.get('selected_actions'),
                              control_ticks=data.get('control_ticks'), error=data.get('error'),
                              initial_file_sha256=data.get('initial_file_sha256'),
                              execution_status=data.get('execution_status')))
    else:
        path = raw / 'summary.json'
        data = json.loads(path.read_text())
        if data.get('policy') != policy or data.get('backend') != 'so101':
            raise ValueError('SO101 result identity mismatch')
        records.append(record(0, data['status'], None, None,
                              physical_dispatches=data.get('raw_goal_packets_transmitted'),
                              rejection_reasons=data.get('rejection_reasons')))
    for row in records:
        row.update(job_id=job.get('id'), backend=backend, task=config['task'], evidence={'path': str(path), 'sha256': digest(path)},
                   seed=config['first_seed'] + row['episode_index'] if backend != 'so101' else None,
                   seed_role='environment' if backend == 'aloha' else 'policy_sampling' if backend == 'mujoco' else None)
    return records


def aggregate(records, policies, expected_episodes=None):
    rows = []
    for policy in policies:
        selected = [r for r in records if r['policy'] == policy['name']]
        known = [r for r in selected if r['outcome'] is not None]
        successes = sum(r['outcome'] is True for r in known)
        rows.append({'policy': policy['name'], 'recorded_episodes': len(selected),
                     'expected_episodes': expected_episodes,
                     'missing_episodes': max(0, expected_episodes - len(selected)) if expected_episodes is not None else None,
                     'known_outcomes': len(known), 'unknown_outcomes': len(selected) - len(known),
                     'successes': successes, 'success_rate_known': successes / len(known) if known else None})
    return rows


def run_evaluation(plan, output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'raw').mkdir()
    (output / 'logs').mkdir()
    write_json(output / 'plan.json', plan)
    c = plan['config']
    result = {'schema_version': 1, 'backend': c['backend'], 'task': c['task'],
              'mode': plan['mode'], 'execution_status': 'running', 'jobs': [], 'episodes': [],
              'comparison': [], 'scope': 'development_evaluation',
              'notes': ['Task failure is separate from execution failure.',
                        'ALOHA uses native evaluators with common reward scoring; evaluator implementations differ.',
                        'MuJoCo seeds control policy sampling, not randomized initial states.',
                        'SO101 task outcome remains unknown pending operator evidence.']}
    write_json(output / 'results.json', result)
    interrupted = False
    for job in plan['jobs']:
        started = time.monotonic()
        entry = {'id': job['id'], 'policy': job['policy'], 'execution_status': 'error',
                 'returncode': None, 'log': str(output / 'logs' / (job['id'] + '.log'))}
        try:
            env = dict(os.environ, **job['environment'])
            policy = next(p for p in c['policies'] if p['name'] == job['policy'])
            if 'checkpoint' in policy:
                checkpoint = Path(policy['checkpoint'])
                entry['checkpoint_sha256'] = digest(checkpoint / 'model.safetensors')
                entry['checkpoint_path'] = str(checkpoint)
            entry['input_sha256'] = {key: digest(c[key]) for key in ('scene', 'calibration', 'alignment_spec') if key in c}
            if 'initial_conditions' in c:
                entry['input_sha256']['initial_condition'] = digest(c['initial_conditions'][job['episode_index']])
            if c['backend'] == 'aloha':
                sim_python = c.get('sim_python', str(ROOT / '.venv-aloha/bin/python'))
                site = subprocess.run([sim_python, '-c', 'import site; print(site.getsitepackages()[0])'],
                                      check=True, capture_output=True, text=True, timeout=30).stdout.strip()
                env['PYTHONPATH'] = (str(ROOT / 'third_party/lerobot-dot-legacy') + os.pathsep
                                     if job['policy'] == 'dot' else '') + site
            with open(entry['log'], 'w') as log:
                process = subprocess.Popen(job['argv'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    entry['returncode'] = process.wait()
                except KeyboardInterrupt:
                    # Forward Ctrl-C so the hardware runner can perform its existing stop/hold cleanup.
                    import signal
                    if process.poll() is None:
                        process.send_signal(signal.SIGINT)
                    entry['returncode'] = process.wait()
                    interrupted = True
            if plan['mode'] == 'validate':
                validation_path = Path(job['raw_directory']) / 'validation.json'
                entry['validation'] = json.loads(validation_path.read_text())
                if entry['validation'].get('status') != 'validated_no_hardware':
                    raise ValueError('Unexpected hardware validation status')
            else:
                result['episodes'].extend(normalize(job, c))
            if interrupted:
                entry['execution_status'] = 'interrupted'
            elif entry['returncode'] == 0:
                entry['execution_status'] = 'completed'
            elif c['backend'] == 'so101' and result['episodes'][-1]['task_status'] in {'gate_rejected', 'aborted'}:
                entry['execution_status'] = 'stopped'
            else:
                entry['error'] = 'Native runner returned a nonzero exit status'
            if c['backend'] == 'mujoco' and result['episodes'] and not interrupted:
                row = result['episodes'][-1]
                native_execution = row['metrics'].get('execution_status')
                if native_execution in {'error', 'stopped'}:
                    entry['execution_status'] = native_execution
                if row['task_status'] in {'input_error', 'simulation_error', 'invalid_initial_state', 'not_started'}:
                    entry['execution_status'] = 'error'
                if entry['execution_status'] == 'error':
                    entry['error'] = 'Native runner reported an execution/setup error; see raw summary'
        except KeyboardInterrupt:
            interrupted = True
            entry.update(execution_status='interrupted', error='Interrupted by user')
        except Exception as exc:
            entry['error'] = f'{type(exc).__name__}: {exc}'
            if interrupted:
                entry['execution_status'] = 'interrupted'
        entry['wall_seconds'] = time.monotonic() - started
        result['jobs'].append(entry)
        result['comparison'] = aggregate(result['episodes'], c['policies'], 0 if plan['mode'] == 'validate' else c['episodes'])
        write_json(output / 'results.json', result)
        if interrupted:
            break
    states = [j['execution_status'] for j in result['jobs']]
    result['execution_status'] = ('interrupted' if interrupted else 'error' if 'error' in states
                                  else 'stopped' if 'stopped' in states else 'completed')
    write_json(output / 'results.json', result)
    return result
