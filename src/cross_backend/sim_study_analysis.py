"""Strict V-stage simulation analysis; never upgrades development trials to formal data."""
from collections import Counter
from math import sqrt
from pathlib import Path
import hashlib,json,random
from .sim_trial_initial import initial_geometry_sha256

POLICIES = ('ACT', 'SmolVLA')
CONDITIONS = ('baseline', 'added_observation_age')
TERMINAL = {'success', 'timeout', 'failed_grasp', 'dropped', 'collision',
            'placement_failed', 'gate_rejected', 'numerical_error', 'aborted', 'uncertain'}
STATUSES = TERMINAL | {'not_run', 'running'}


def wilson_95(successes, started):
    if type(successes) is not int or type(started) is not int or not 0 <= successes <= started:
        raise ValueError('Invalid Wilson counts')
    if started == 0:
        return None
    z = 1.959963984540054
    p = successes / started
    den = 1 + z*z/started
    centre = (p + z*z/(2*started))/den
    half = z*sqrt((p*(1-p)+z*z/(4*started))/started)/den
    return [max(0.0, centre-half), min(1.0, centre+half)]


def _percentile(values, q):
    values.sort()
    pos = (len(values)-1)*q
    lo = int(pos)
    hi = min(lo+1, len(values)-1)
    return values[lo] + (values[hi]-values[lo])*(pos-lo)


def _file_sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def _verify_evidence(row, root):
    evidence=row.get('evidence')
    if not isinstance(evidence,dict):return False,'evidence_manifest_missing'
    paths={}
    for kind in ('summary','events','video','initial'):
        item=evidence.get(kind)
        if not isinstance(item,dict) or not isinstance(item.get('path'),str) or not isinstance(item.get('sha256'),str):
            return False,f'{kind}_identity_missing'
        path=(root/item['path']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size==0:
            return False,f'{kind}_file_missing_or_outside_root'
        if _file_sha256(path)!=item['sha256']:
            return False,f'{kind}_hash_mismatch'
        paths[kind]=path
    try:
        summary=json.loads(paths['summary'].read_text())
        with paths['events'].open() as events:first_event=json.loads(events.readline())
        with paths['video'].open('rb') as video:header=video.read(8)
    except (OSError,UnicodeError,json.JSONDecodeError):
        return False,'evidence_unreadable'
    fields=('trial_id','initial_condition_id','policy','condition','backend','split','status',
            'initial_sha256','scene_sha256','protocol_sha256','checkpoint_sha256')
    if any(summary.get(field)!=row.get(field) for field in fields) or first_event.get('trial_id')!=row['trial_id']:
        return False,'evidence_identity_mismatch'
    if summary.get('formal_trial') is not True:
        return False,'trial_not_formal'
    if summary.get('physical_alignment_verified') is not True:
        return False,'physical_alignment_not_verified'
    if len(header)<8 or header[4:8]!=b'ftyp':
        return False,'video_container_invalid'
    try:
        initial=json.loads(paths['initial'].read_text())
        if initial.get('split')!='V' or initial.get('initial_condition_id')!=row['initial_condition_id'] or initial.get('scene_sha256')!=row['scene_sha256']:
            return False,'initial_record_identity_mismatch'
        if initial_geometry_sha256(initial)!=row['initial_sha256']:
            return False,'initial_geometry_mismatch'
    except (OSError,UnicodeError,json.JSONDecodeError,ValueError,TypeError):
        return False,'initial_record_invalid'
    return True,None


def analyze_v_stage(record, *, evidence_root=None, bootstrap_samples=2000, seed=0):
    """Analyze exactly V01..V10 × two policies × two conditions.

    Non-synthetic rates need all 40 terminal trials and file/identity checks.
    Stage acceptance also needs separate S1/protocol/video review and is not inferred here.
    """
    if record.get('stage') != 'V' or type(record.get('synthetic_fixture')) is not bool:
        raise ValueError('Explicit V stage and synthetic_fixture boolean required')
    if type(bootstrap_samples) is not int or not 1 <= bootstrap_samples <= 100000:
        raise ValueError('Invalid bootstrap sample count')
    if type(seed) is not int:
        raise ValueError('Integer bootstrap seed required')
    rows = record.get('trials')
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError('V requires all 40 planned slots, including unstarted trials')
    expected = {(f'V{i:02d}', p, c) for i in range(1, 11)
                for p in POLICIES for c in CONDITIONS}
    keys = [(r.get('initial_condition_id'), r.get('policy'), r.get('condition')) for r in rows]
    ids = [r.get('trial_id') for r in rows]
    if set(keys) != expected or len(set(keys)) != 40 or len(set(ids)) != 40 or any(not x for x in ids):
        raise ValueError('Duplicate/missing V slots or trial IDs')
    root=Path(evidence_root).resolve() if evidence_root is not None else None
    verified=set();evidence_failures={}
    if not record['synthetic_fixture']:
        for row in rows:
            if row['status']=='not_run':continue
            if root is None:
                evidence_failures[row['trial_id']]='evidence_root_missing';continue
            ok,reason=_verify_evidence(row,root)
            if ok:verified.add(row['trial_id'])
            else:evidence_failures[row['trial_id']]=reason
    by_key = dict(zip(keys, rows, strict=True))
    scene_ids = set()
    protocol_ids = set()
    policy_ids = {p:set() for p in POLICIES}
    for row in rows:
        status = row.get('status')
        if row.get('backend') != 'mujoco' or row.get('split') != 'V' or status not in STATUSES:
            raise ValueError('Wrong backend/split/status in V slot')
        if status == 'not_run' and row['trial_id'] in verified:
            raise ValueError('Unstarted trial cannot be verified')
        for field in ('initial_sha256', 'scene_sha256', 'protocol_sha256', 'checkpoint_sha256'):
            value = row.get(field)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'Missing/invalid {field}')
        scene_ids.add(row['scene_sha256'])
        protocol_ids.add(row['protocol_sha256'])
        policy_ids[row['policy']].add(row['checkpoint_sha256'])
    block_initials=[]
    for i in range(1, 11):
        block = [by_key[(f'V{i:02d}', p, c)] for p in POLICIES for c in CONDITIONS]
        if len({r['initial_sha256'] for r in block}) != 1:
            raise ValueError(f'Paired initial identity mismatch at V{i:02d}')
        block_initials.append(block[0]['initial_sha256'])
    if len(set(block_initials))!=10:
        raise ValueError('Repeated V initial geometry identity across blocks')
    if len(scene_ids) != 1 or len(protocol_ids) != 1 or any(len(x) != 1 for x in policy_ids.values()):
        raise ValueError('Scene, protocol or policy checkpoint changed inside V')
    file_complete = (not record['synthetic_fixture'] and all(r['status'] in TERMINAL for r in rows)
                     and verified == set(ids))
    stats_allowed = file_complete or record['synthetic_fixture']
    groups = {}
    for p in POLICIES:
        for c in CONDITIONS:
            block = [r for r in rows if r['policy'] == p and r['condition'] == c]
            counts = Counter(r['status'] for r in block)
            started = len(block) - counts['not_run']
            successes = counts['success']
            groups[f'{p}/{c}'] = {
                'planned': 10, 'started': started, 'verified': sum(r['trial_id'] in verified for r in block),
                'status_counts': dict(sorted(counts.items())),
                'successes': successes if (stats_allowed) else None,
                'success_rate': successes/started if started and (stats_allowed) else None,
                'wilson_95': wilson_95(successes, started) if started and (stats_allowed) else None,
            }
    complete_blocks = []
    excluded = []
    for i in range(1, 11):
        block = [by_key[(f'V{i:02d}', p, c)] for p in POLICIES for c in CONDITIONS]
        if any(r['status'] not in TERMINAL for r in block):
            excluded.append({'initial_condition_id':f'V{i:02d}', 'reason':'not_all_terminal'})
        elif not record['synthetic_fixture'] and any(r['trial_id'] not in verified for r in block):
            excluded.append({'initial_condition_id':f'V{i:02d}', 'reason':'evidence_not_verified'})
        else:
            complete_blocks.append({(r['policy'], r['condition']):int(r['status']=='success') for r in block})
    contrast = None
    if (stats_allowed) and complete_blocks:
        def estimate(blocks):
            n = len(blocks)
            return {
                'ACT_delay_minus_baseline':sum(b[('ACT','added_observation_age')]-b[('ACT','baseline')] for b in blocks)/n,
                'SmolVLA_delay_minus_baseline':sum(b[('SmolVLA','added_observation_age')]-b[('SmolVLA','baseline')] for b in blocks)/n,
                'SmolVLA_minus_ACT_at_baseline':sum(b[('SmolVLA','baseline')]-b[('ACT','baseline')] for b in blocks)/n,
                'difference_of_delay_effects':sum((b[('SmolVLA','added_observation_age')]-b[('SmolVLA','baseline')])-(b[('ACT','added_observation_age')]-b[('ACT','baseline')]) for b in blocks)/n,
            }
        point = estimate(complete_blocks)
        rng = random.Random(seed)
        samples = {k:[] for k in point}
        for _ in range(bootstrap_samples):
            draw=[complete_blocks[rng.randrange(len(complete_blocks))] for _ in complete_blocks]
            for k,v in estimate(draw).items():samples[k].append(v)
        contrast = {k:{'estimate':v,'block_bootstrap_95':[_percentile(samples[k],.025),_percentile(samples[k],.975)]}
                    for k,v in point.items()}
    return {
        'stage':'V', 'synthetic_fixture':record['synthetic_fixture'], 'full_matrix_file_verified':file_complete,
        'formal_stage_accepted':False, 'evidence_failures':evidence_failures,
        'evidence_label':'SYNTHETIC TEST DATA — NOT EXPERIMENTAL RESULTS' if record['synthetic_fixture'] else
                         ('File-integrity-verified V records; S2 stage gate still separate' if file_complete else 'INCOMPLETE OR UNVERIFIED V — rates withheld'),
        'planned':40, 'started':sum(r['status']!='not_run' for r in rows),
        'verified':len(verified), 'complete_blocks':len(complete_blocks), 'excluded_blocks':excluded,
        'groups':groups, 'paired_contrasts':contrast,
        'bootstrap_samples':bootstrap_samples, 'bootstrap_seed':seed,
        'denominator_rule':'all started slots, including rejection, numerical error and abort; no missing-trial imputation',
    }
