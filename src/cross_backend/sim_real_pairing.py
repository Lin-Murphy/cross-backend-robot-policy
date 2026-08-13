"""Identity gate for the future T simulation/SO101 pairing; no outcome statistics."""

from collections import Counter

POLICIES = ('ACT', 'SmolVLA')
CONDITIONS = ('baseline', 'added_observation_age')
def _sha256(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

SIM_STATUSES = {'not_run', 'running', 'success', 'timeout', 'failed_grasp',
                'dropped', 'collision', 'placement_failed', 'gate_rejected',
                'numerical_error', 'aborted', 'uncertain'}
REAL_STATUSES = {'not_run', 'running', 'success', 'failed', 'uncertain', 'aborted', 'rejected'}


def _slots(rows, *, backend):
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError(f'{backend} must have exactly 40 planned slots')
    expected = {(f'P{i:02d}', p, c) for i in range(1, 11)
                for p in POLICIES for c in CONDITIONS}
    keys = [(r.get('initial_condition_id'), r.get('policy'), r.get('condition')) for r in rows]
    ids = [r.get('trial_id') for r in rows]
    if set(keys) != expected or len(set(keys)) != 40 or len(set(ids)) != 40:
        raise ValueError(f'{backend} has missing or duplicate P slots/IDs')
    if any(r.get('status') not in (SIM_STATUSES if backend == 'mujoco' else REAL_STATUSES)
           for r in rows):
        raise ValueError(f'{backend} has an invalid status')
    return dict(zip(keys, rows, strict=True))


def make_t_template(real_record):
    """Mirror the frozen SO101 slot identities before any real trial starts."""
    real=_slots(real_record.get('trials'),backend='so101')
    if not _sha256(real_record.get('order_source_sha256')):
        raise ValueError('Real frozen order identity missing')
    if any(r['status']!='not_run' for r in real.values()):
        raise ValueError('T template must be created before real trials start')
    rows=[]
    for r in real_record['trials']:
        if type(r['trial_id']) is not int or r['trial_id']<1:
            raise ValueError('Real trial ID must be a positive integer')
        rows.append({'trial_id':f"T-sim-{r['trial_id']:02d}",
                     'paired_real_trial_id':r['trial_id'],
                     'initial_condition_id':r['initial_condition_id'],
                     'policy':r['policy'],'condition':r['condition'],
                     'backend':'mujoco','split':'T','status':'not_run',
                     'initial_sha256':None,'evidence':None})
    return {'stage':'T','synthetic_fixture':False,
            'purpose':'STRUCTURAL_PLACEHOLDER_ONLY_not_frozen_T_initial_states',
            'real_order_source_sha256':real_record['order_source_sha256'],
            'trials':rows}


def inspect_t_pairing(sim_record, real_record, *, real_validation):
    """Inspect identities only; evidence/outcomes require separate future acceptance."""
    if sim_record.get('stage') != 'T' or sim_record.get('synthetic_fixture') is not False:
        raise ValueError('Only non-synthetic T simulation records may pair')
    sim = _slots(sim_record.get('trials'), backend='mujoco')
    real = _slots(real_record.get('trials'), backend='so101')
    if any(r.get('backend') != 'mujoco' or r.get('split') != 'T' for r in sim.values()):
        raise ValueError('Simulation rows must declare MuJoCo/T')
    if real_validation.get('record_structure') != 'passed' or real_validation.get('slots') != 40:
        raise ValueError('Real ledger has not passed its structural validator')
    if not _sha256(real_record.get('order_source_sha256')):
        raise ValueError('Real frozen order identity missing')
    if sim_record.get('real_order_source_sha256') != real_record['order_source_sha256']:
        raise ValueError('T simulation is not bound to the real frozen order')
    rows = []
    for key in sorted(sim):
        s, r = sim[key], real[key]
        if s.get('paired_real_trial_id') != r['trial_id']:
            raise ValueError(f'T simulation paired real trial ID mismatch at {key}')
        s_started = s['status'] != 'not_run'
        r_started = r['status'] != 'not_run'
        s_identity = s.get('initial_sha256')
        r_identity = r.get('initial_condition_sha256')
        if s_started and not _sha256(s_identity):
            raise ValueError(f'Started T simulation missing initial identity at {key}')
        if r_started and not _sha256(r_identity):
            raise ValueError(f'Started SO101 trial missing initial identity at {key}')
        identity_match = s_identity == r_identity if s_started and r_started else None
        rows.append({'initial_condition_id': key[0], 'policy': key[1], 'condition': key[2],
                     'sim_trial_id': s['trial_id'], 'real_trial_id': r['trial_id'],
                     'sim_status': s['status'], 'real_status': r['status'],
                     'both_started': s_started and r_started,
                     'initial_identity_match': identity_match,
                     'pair_state': ('initial_identity_mismatch' if identity_match is False else
                                    'both_started_evidence_pending' if identity_match is True else
                                    'sim_only' if s_started else 'real_only' if r_started else 'not_started')})
    counts = Counter(row['pair_state'] for row in rows)
    return {'stage': 'T', 'planned_pairs': 40,
            'real_order_source_sha256': real_record['order_source_sha256'],
            'order_and_trial_id_binding_verified': True,
            'real_field_ready': real_validation.get('field_ready') is True,
            'sim_started': sum(row['sim_status'] != 'not_run' for row in rows),
            'real_started': sum(row['real_status'] != 'not_run' for row in rows),
            'pair_state_counts': dict(sorted(counts.items())), 'pairs': rows,
            'outcome_statistics': None, 'formal_stage_accepted': False,
            'evidence_label': 'PAIRING READINESS ONLY — no outcome or predictive-validity claim',
            'remaining_gates': ['frozen_T_initial_conditions', 'S1_physical_alignment',
                                'S2_simulation_screening', 'SO101_execution_authorization',
                                'both_backend_evidence_review', 'paired_outcome_analysis']}
