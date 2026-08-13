"""Descriptive paired analysis; unstarted/running slots never become fabricated outcomes."""
from collections import Counter
from datetime import datetime
import statistics

POLICIES=('ACT','SmolVLA')
CONDITIONS=('baseline','added_observation_age')
TERMINAL={'success','failed','uncertain','aborted','rejected'}


def elapsed_s(row):
    if row['status'] not in TERMINAL:return None
    start=datetime.fromisoformat(row['started_utc'].replace('Z','+00:00'))
    end=datetime.fromisoformat(row['ended_utc'].replace('Z','+00:00'))
    if start.tzinfo is None or end.tzinfo is None:raise ValueError('Use timezone-aware actual start/end timestamps')
    delta=(end-start).total_seconds()
    if delta<0:raise ValueError('Trial ends before it starts')
    return delta


def duration_summary(values):
    return {'n':len(values),'mean_s':statistics.mean(values) if values else None,
            'median_s':statistics.median(values) if values else None,
            'min_s':min(values) if values else None,'max_s':max(values) if values else None}


def group_summary(rows):
    counts=Counter(r['status'] for r in rows)
    started=len(rows)-counts['not_run']
    completed=sum(counts[s] for s in TERMINAL)
    success=counts['success']
    durations=[elapsed_s(r) for r in rows if r['status'] in TERMINAL]
    successful=[elapsed_s(r) for r in rows if r['status']=='success']
    return {'planned':len(rows),'started':started,'not_run':counts['not_run'],
            'running':counts['running'],'terminal':completed,'confirmed_successes':success,
            'success_rate_among_started':success/started if started else None,
            'rate_is_provisional':bool(counts['running']),
            'status_counts':dict(sorted(counts.items())),
            'failure_classes':dict(Counter(r.get('failure_class') or r['status'] for r in rows
                                           if r['status'] in TERMINAL-{'success'})),
            'terminal_trial_duration':duration_summary(durations),
            'successful_completion_duration':duration_summary(successful)}


def analyze_trials(record, *, verified_trial_ids=()):
    verified_trial_ids=set(verified_trial_ids)
    rows=record['trials']
    expected={(f'P{i:02d}',p,c) for i in range(1,11) for p in POLICIES for c in CONDITIONS}
    actual=[(r['initial_condition_id'],r['policy'],r['condition']) for r in rows]
    if len(rows)!=40 or set(actual)!=expected or len({r['trial_id'] for r in rows})!=40:
        raise ValueError('Require all 40 unique planned slots; do not drop failures or missing trials')
    for row in rows:
        status=row['status']
        if status not in TERMINAL|{'not_run','running'}:raise ValueError('Unknown status')
        if status=='not_run':
            if row['started_utc'] is not None or row['independent_success'] is not None:
                raise ValueError('Unstarted trial cannot have an outcome')
        elif not row['started_utc']:raise ValueError('Started trials require start time')
        if status=='running' and row['independent_success'] is not None:
            raise ValueError('Running trial cannot have final success label')
        if status in TERMINAL:
            if type(row['human_intervention']) is not bool:raise ValueError('Intervention must be explicitly boolean')
            if row['independent_success'] is not (status=='success'):raise ValueError('Outcome mismatch')
            if status=='success' and row['human_intervention']:raise ValueError('Assisted trial cannot be independent success')
            elapsed_s(row)
    by_key={key:row for key,row in zip(actual,rows,strict=True)}
    identities={p:sorted({r['checkpoint_sha256'] for r in rows if r['policy']==p
                         and r['status'] in TERMINAL and r.get('checkpoint_sha256')}) for p in POLICIES}

    def paired(name,left_key,right_key,direction):
        pairs=[];excluded=[]
        for i in range(1,11):
            initial=f'P{i:02d}';left=by_key[(initial,*left_key)];right=by_key[(initial,*right_key)]
            reason=None
            if left['status'] not in TERMINAL or right['status'] not in TERMINAL:
                reason='not_both_terminal'
            elif not left.get('initial_condition_sha256') or not right.get('initial_condition_sha256'):
                reason='initial_identity_missing'
            elif left['initial_condition_sha256']!=right['initial_condition_sha256']:
                reason='initial_identity_mismatch'
            elif not left.get('checkpoint_sha256') or not right.get('checkpoint_sha256'):
                reason='checkpoint_identity_missing'
            elif any(len(identities[p])!=1 for p in {left_key[0],right_key[0]}):
                reason='checkpoint_changed_within_policy'
            elif not record.get('synthetic_fixture') and not {left['trial_id'],right['trial_id']} <= verified_trial_ids:
                reason='trial_evidence_not_verified'
            if reason:
                excluded.append({'initial_condition_id':initial,'reason':reason});continue
            a=int(left['status']=='success');b=int(right['status']=='success')
            pairs.append({'initial_condition_id':initial,'left_trial_id':left['trial_id'],
                          'right_trial_id':right['trial_id'],'left_success':a,'right_success':b,
                          'success_difference_right_minus_left':b-a,
                          'success_duration_difference_s':elapsed_s(right)-elapsed_s(left) if a and b else None})
        differences=[p['success_difference_right_minus_left'] for p in pairs]
        times=[p['success_duration_difference_s'] for p in pairs if p['success_duration_difference_s'] is not None]
        return {'comparison':name,'direction':direction,'complete_recorded_pairs':len(pairs),
                'complete_verified_pairs':sum({p['left_trial_id'],p['right_trial_id']} <= verified_trial_ids for p in pairs) if not record.get('synthetic_fixture') else 0,
                'right_better':sum(d>0 for d in differences),'left_better':sum(d<0 for d in differences),
                'ties':sum(d==0 for d in differences),
                'paired_success_difference':statistics.mean(differences) if differences else None,
                'duration_difference_both_successful':duration_summary(times),'pairs':pairs,'excluded_pairs':excluded}

    groups={f'{p}/{c}':group_summary([r for r in rows if r['policy']==p and r['condition']==c])
            for p in POLICIES for c in CONDITIONS}
    comparisons=[paired(p,(p,'baseline'),(p,'added_observation_age'),'added age minus baseline') for p in POLICIES]
    comparisons += [paired(c,('ACT',c),('SmolVLA',c),'SmolVLA minus ACT') for c in CONDITIONS]
    return {'synthetic_fixture':bool(record.get('synthetic_fixture',False)),
            'evidence_label':'SYNTHETIC TEST DATA — NOT EXPERIMENTAL RESULTS' if record.get('synthetic_fixture') else 'Recorded trial slots',
            'overall':group_summary(rows),'groups':groups,'paired_comparisons':comparisons,
            'outcome_scope':'reported labels; evidence verification is separate',
            'verified_trial_ids':sorted(verified_trial_ids) if not record.get('synthetic_fixture') else [],
            'checkpoint_identities':identities,'inference_scope':'descriptive only; small paired sample, no causal or significance claim',
            'denominator_rule':'all started trials including failures, uncertainty, rejection and abort; running rates provisional',
            'duration_rule':'success completion times separate from all-terminal durations; no imputation for missing/running outcomes'}
