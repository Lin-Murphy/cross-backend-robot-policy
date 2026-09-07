"""Evidence-checked descriptive comparisons of unified evaluation results."""
from collections import Counter
import json
import math
from pathlib import Path
import random

from .evaluation import digest, normalize, write_json


def wilson95(successes, count):
    if not count:
        return None
    z = 1.959963984540054
    p = successes / count
    den = 1 + z * z / count
    centre = (p + z * z / (2 * count)) / den
    half = z * math.sqrt(p * (1 - p) / count + z * z / (4 * count * count)) / den
    return [max(0., centre - half), min(1., centre + half)]


def paired_statistics(outcomes, *, bootstrap_seed=20260928):
    """Pairs contain only two observed boolean outcomes; no unknown imputation."""
    for a, b in outcomes:
        if type(a) is not bool or type(b) is not bool:
            raise ValueError('Paired statistics require known boolean outcomes')
    n = len(outcomes)
    if not n:
        return {'pairs': 0, 'difference_right_minus_left': None, 'bootstrap95': None,
                'left_only': 0, 'right_only': 0, 'both_success': 0, 'both_failure': 0,
                'mcnemar_exact_two_sided': None}
    left = sum(a and not b for a, b in outcomes)
    right = sum(b and not a for a, b in outcomes)
    discordant = left + right
    p = min(1., 2 * sum(math.comb(discordant, k) for k in range(min(left, right) + 1)) / 2**discordant) if discordant else 1.
    differences = [int(b) - int(a) for a, b in outcomes]
    rng = random.Random(bootstrap_seed)
    draws = sorted(sum(differences[rng.randrange(n)] for _ in range(n)) / n for _ in range(10000))
    return {'pairs': n, 'difference_right_minus_left': sum(differences) / n,
            'bootstrap95': [draws[249], draws[9749]], 'bootstrap_seed': bootstrap_seed,
            'left_only': left, 'right_only': right,
            'both_success': sum(a and b for a, b in outcomes),
            'both_failure': sum(not a and not b for a, b in outcomes),
            'mcnemar_exact_two_sided': p}


def read_verified_run(run_directory, protocol_path):
    root = Path(run_directory).resolve()
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    plan = json.loads((root / 'plan.json').read_text())
    result = json.loads((root / 'results.json').read_text())
    c = plan['config']
    if result['execution_status'] == 'running' or plan['mode'] != 'evaluate':
        raise ValueError('Study requires a finished evaluation, not validation or a running job')
    if (c['backend'], c['task']) != (protocol['backend'], protocol['task']):
        raise ValueError('Protocol and result task/backend differ')
    if (c['episodes'], c['first_seed']) != (protocol['episodes_per_policy'], protocol['first_seed']):
        raise ValueError('Protocol episode or seed mismatch')
    # Verify the frozen configuration, not just a matching filename or title.
    config_path = protocol_path.parent / (c['backend'] + '-config.json')
    if digest(config_path) != protocol['config_sha256']:
        raise ValueError('Frozen configuration has changed')
    from .evaluation import load_config
    if load_config(config_path) != c:
        raise ValueError('Actual plan differs from frozen configuration')
    if len({j['id'] for j in result['jobs']}) != len(result['jobs']):
        raise ValueError('Duplicate result job IDs')
    jobs = {j['id']: j for j in result['jobs']}
    if set(jobs) != {j['id'] for j in plan['jobs']}:
        raise ValueError('Planned jobs missing or unexpected')
    model_hashes = {}
    for p in c['policies']:
        frozen = protocol['model_manifests'][p['name']]
        cp = Path(p['checkpoint'])
        if any(digest(cp / name) != sha for name, sha in frozen.items()):
            raise ValueError('Checkpoint or processor changed since protocol freeze')
        model_hashes[p['name']] = frozen['model.safetensors']
    for path, sha in protocol['code_sha256'].items():
        archived = protocol_path.parent / 'executed-code' / sha
        source = archived if archived.is_file() else Path(path)
        if digest(source) != sha:
            raise ValueError(f'Execution code snapshot missing or changed: {path}')
    rebuilt = []
    for job in plan['jobs']:
        saved = jobs[job['id']]
        if saved.get('checkpoint_sha256') != model_hashes[job['policy']]:
            raise ValueError('Executed checkpoint differs from frozen checkpoint')
        if c['backend'] == 'mujoco':
            idx = job['episode_index']
            initial = Path(c['initial_conditions'][idx])
            expected = protocol['initials'][idx]['file_sha256']
            if digest(initial) != expected or saved.get('input_sha256', {}).get('initial_condition') != expected:
                raise ValueError('Initial condition changed between protocol and execution')
            for key in ('scene', 'calibration'):
                if digest(c[key]) != saved['input_sha256'][key]:
                    raise ValueError(f'{key} input changed')
        try:
            rows = normalize(job, c)
        except (OSError, KeyError, ValueError):
            if saved['execution_status'] not in {'error', 'interrupted'}:
                raise
            rows = []
        for row in rows:
            if c['backend'] == 'mujoco' and row['metrics']['initial_file_sha256'] != protocol['initials'][row['episode_index']]['file_sha256']:
                raise ValueError('Native initial identity differs from frozen input')
            rebuilt.append(row)
    if rebuilt != result['episodes']:
        raise ValueError('Unified episode results do not reproduce from native evidence')
    return protocol, plan, result


def analyze_model_comparison(run_directory, protocol_path):
    protocol, plan, result = read_verified_run(run_directory, protocol_path)
    c = plan['config']
    names = [p['name'] for p in c['policies']]
    if len(names) != 2:
        raise ValueError('Model comparison requires exactly two policies')
    rows = result['episodes']
    by_key = {(r['policy'], r['episode_index']): r for r in rows}
    if len(by_key) != len(rows):
        raise ValueError('Duplicate episode identity')
    groups = []
    for name in names:
        selected = [r for r in rows if r['policy'] == name]
        known = [r for r in selected if r['outcome'] is not None]
        successes = sum(r['outcome'] is True for r in known)
        unknown = c['episodes'] - len(known)
        groups.append({'policy': name, 'planned': c['episodes'], 'recorded': len(selected),
                       'known': len(known), 'unknown_or_missing': unknown, 'successes': successes,
                       'success_rate_known': successes / len(known) if known else None,
                       'wilson95_known': wilson95(successes, len(known)),
                       'success_rate_bounds_all_planned': [successes / c['episodes'], (successes + unknown) / c['episodes']],
                       'task_status_counts': dict(Counter(r['task_status'] for r in selected)),
                       'execution_status_counts': dict(Counter(j['execution_status'] for j in result['jobs'] if j['policy'] == name))})
    pairs, excluded, outcomes = [], [], []
    for index in range(c['episodes']):
        a, b = [by_key.get((name, index)) for name in names]
        identity = (f'seed-{c["first_seed"] + index}' if c['backend'] == 'aloha'
                    else protocol['initials'][index]['id'])
        row = {'pair_id': identity, 'left': a, 'right': b}
        pairs.append(row)
        if a is None or b is None:
            excluded.append({'pair_id': identity, 'reason': 'missing_evidence'})
        elif a['outcome'] is None or b['outcome'] is None:
            excluded.append({'pair_id': identity, 'reason': 'unknown_task_outcome'})
        else:
            if a['seed'] != b['seed']:
                raise ValueError('Pair seed mismatch')
            outcomes.append((a['outcome'], b['outcome']))
    root = Path(run_directory).resolve()
    return {'schema_version': 1, 'study_id': protocol['study_id'], 'scope': protocol['scope'],
            'backend': c['backend'], 'task': c['task'], 'left_policy': names[0], 'right_policy': names[1],
            'execution_status': result['execution_status'], 'evidence_verified': True,
            'sources': {str(p): digest(p) for p in (Path(protocol_path).resolve(), root / 'plan.json', root / 'results.json')},
            'groups': groups, 'paired': paired_statistics(outcomes), 'excluded_pairs': excluded, 'pairs': pairs,
            'interpretation': 'Descriptive finite sample of deployment systems. Unknown and missing outcomes are excluded from binary pair statistics and shown separately. Bootstrap intervals may degenerate with uniform outcomes; they do not establish equivalence.'}


def render_model_report(report):
    lines = [f"# Model comparison: {report['study_id']}", '', report['scope'], '',
             f"Execution: **{report['execution_status']}**. Native evidence and frozen identities verified.", '',
             '| Policy | Recorded / planned | Known outcomes | Successes | Unknown / missing | Task statuses |',
             '| --- | --- | --- | --- | --- | --- |']
    for g in report['groups']:
        lines.append(f"| {g['policy']} | {g['recorded']}/{g['planned']} | {g['known']} | {g['successes']} | {g['unknown_or_missing']} | {g['task_status_counts']} |")
    paired = report['paired']
    lines += ['', f"Known-outcome pairs: **{paired['pairs']}**; excluded pairs: **{len(report['excluded_pairs'])}**.",
              f"Difference ({report['right_policy']} minus {report['left_policy']}): {paired['difference_right_minus_left']}.",
              f"Paired bootstrap 95% interval: {paired['bootstrap95']}; exact two-sided McNemar p: {paired['mcnemar_exact_two_sided']}.",
              '', report['interpretation'], '', '| Initial condition | Left task status | Right task status |', '| --- | --- | --- |']
    for pair in report['pairs']:
        lines.append(f"| {pair['pair_id']} | {(pair['left'] or {}).get('task_status', 'missing')} | {(pair['right'] or {}).get('task_status', 'missing')} |")
    return '\n'.join(lines) + '\n'
