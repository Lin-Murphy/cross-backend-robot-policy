"""Create/validate the frozen 40-slot evaluation records. Never controls hardware."""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ORDER=ROOT/'configs/move-pot-trial-order.json'
JOINTS=['shoulder_pan.pos','shoulder_lift.pos','elbow_flex.pos','wrist_flex.pos','wrist_roll.pos','gripper.pos']
UNITS=['deg']*5+['calibrated_0_100']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def save(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')


def initial_condition_sha256(condition):
    """Canonical UTF-8 JSON identity of the complete frozen initial-condition row."""
    payload=json.dumps(condition,sort_keys=True,separators=(',', ':'),ensure_ascii=False,allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def verify_trial_evidence(out, trials, initial, lock, field_ready):
    verified=[];issues={}
    conditions={row['id']:row for row in initial}
    def valid_hash(value):
        return isinstance(value,str) and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    for row in trials:
        if row['status'] in ('not_run','running'):continue
        errors=[]
        if not field_ready:errors.append('execution_or_initial_conditions_not_ready')
        for key in ('checkpoint_sha256','initial_condition_sha256'):
            if not valid_hash(row.get(key)):errors.append(key+'_invalid')
        expected=lock.get(row['policy'].lower()+'_checkpoint_sha256')
        if not expected or row.get('checkpoint_sha256')!=expected:
            errors.append('checkpoint_does_not_match_execution_lock')
        if row.get('initial_condition_sha256')!=initial_condition_sha256(conditions[row['initial_condition_id']]):
            errors.append('initial_condition_does_not_match_frozen_record')
        for key in ('event_log','follower_video','camera2_video'):
            value=row.get(key)
            # Missing evidence for a failed trial remains an explicit issue, never a dropped failure.
            if not isinstance(value,str) or not value:
                errors.append(key+'_missing');continue
            path=Path(value);path=path if path.is_absolute() else out/path
            if not path.is_file() or path.stat().st_size==0:
                errors.append(key+'_missing_or_empty');continue
            digest=row.get(key+'_sha256')
            if not valid_hash(digest) or sha(path)!=digest:
                errors.append(key+'_hash_missing_or_mismatch')
        if errors:issues[row['trial_id']]=errors
        else:verified.append(row['trial_id'])
    return {'verified_trial_ids':verified,'trial_evidence_issues':issues,
            'evidence_scope':'file integrity and frozen identities only; human outcome review remains required'}


def initialize(out):
    out.mkdir(parents=True,exist_ok=False)
    order=json.loads(ORDER.read_text())
    trials=[]
    for row in order['trials']:
        trials.append({**row,'started_utc':None,'ended_utc':None,'not_started_reason':None,
                       'independent_success':None,'human_intervention':None,'failure_class':None,
                       'checkpoint_sha256':None,'initial_condition_sha256':None,
                       'event_log':None,'follower_video':None,'camera2_video':None,
                       'event_log_sha256':None,'follower_video_sha256':None,'camera2_video_sha256':None,
                       'reviewer':None,'notes':None})
    save(out/'trials.json',{'order_source_sha256':sha(ORDER),'seed':order['seed'],'trials':trials})
    save(out/'initial_conditions.json',{'conditions':[
        {'id':f'P{i:02d}','pot_position_and_orientation':None,'robot_start_state':None,
         'outside_green_mat_confirmed':None,'reachable_region_confirmed':None,
         'follower_photo':{'path':None,'sha256':None},'camera2_photo':{'path':None,'sha256':None},
         'recorded_by':None,'frozen_utc':None}
        for i in range(1,11)]})
    save(out/'execution_lock.json',{
        'status':'draft_not_authorized','task':'move pot','success_rule':'grasp pot outside green mat and place inside without help',
        'training_data_role':'all 30 demonstrations are train/development; no unseen offline test',
        'historical_rollouts_role':'pilot only; do not reuse as final test',
        'candidate_parameters':{'rate_hz':30,'predicted_steps':50,'execute_steps':50,'timeout_s':20,'added_age_s':1/30},
        'locked_parameters':{'rate_hz':None,'predicted_steps':None,'execute_steps':None,'timeout_s':None,'added_age_s':None},
        'joint_names':JOINTS,'joint_units':UNITS,
        'common_limits':{'lower':None,'upper':None,'max_speed_per_s':None,'max_state_age_s':None,'max_camera_age_s':None,'max_action_source_age_s':None},
        'act_checkpoint_sha256':None,'smolvla_checkpoint_sha256':None,
        'code_manifest_sha256':None,'follower_calibration_sha256':None,'leader_calibration_sha256':None,
        'camera_mount_reference':None,'independent_pilot_evidence':None,
        'degree_unit_gate_validation_evidence':None,'operator_authorization_reference':None,'locked_utc':None})
    print('Created 40 not_run slots:',out)


def validate(out):
    order=json.loads(ORDER.read_text());record=json.loads((out/'trials.json').read_text())
    if record['order_source_sha256']!=sha(ORDER) or record['seed']!=order['seed']:
        raise ValueError('Frozen order identity changed')
    trials=record['trials']
    if len(trials)!=40:raise ValueError('Exactly 40 original slots required')
    for ref,row in zip(order['trials'],trials,strict=True):
        if any(row[k]!=ref[k] for k in ('trial_id','initial_condition_id','policy','condition')):
            raise ValueError('Do not reorder/replace trial identities')
        status=row['status']
        if status not in ('not_run','running','success','failed','uncertain','aborted','rejected'):
            raise ValueError('Unknown trial status')
        if status=='not_run':
            if row['started_utc'] is not None or row['independent_success'] is not None:
                raise ValueError('Unstarted slot cannot contain started time or a result')
        else:
            if not row['started_utc']:raise ValueError('Started trials require actual start time')
            if status!='running':
                if not row['ended_utc']:raise ValueError('Terminal trials require actual end time')
                if row['human_intervention'] is None:raise ValueError('Explicitly record intervention')
                if row['independent_success'] is not (status=='success'):
                    raise ValueError('Only confirmed success may count as success')
                if status=='success' and row['human_intervention'] is not False:
                    raise ValueError('Assisted completion cannot be independent success')
                if status in ('failed','aborted','rejected') and not row['failure_class']:
                    raise ValueError('Retain failure category')
                for field in ('event_log','reviewer','checkpoint_sha256','initial_condition_sha256'):
                    if not row[field]:raise ValueError(f'Missing terminal trial evidence: {field}')
                if status=='success' and (not row['follower_video'] or not row['camera2_video']):
                    raise ValueError('Confirmed success requires both recorded views')
    initial=json.loads((out/'initial_conditions.json').read_text())['conditions']
    if [r['id'] for r in initial]!=[f'P{i:02d}' for i in range(1,11)]:raise ValueError('Initial-condition identities changed')
    lock=json.loads((out/'execution_lock.json').read_text())
    if lock['joint_names']!=JOINTS or lock['joint_units']!=UNITS:raise ValueError('Wrong joint units/order')
    missing=[]
    for row in initial:
        for k in ('pot_position_and_orientation','robot_start_state','outside_green_mat_confirmed',
                  'reachable_region_confirmed','recorded_by','frozen_utc'):
            if not row[k]:missing.append(f"{row['id']}.{k}")
        state=row['robot_start_state']
        if state is not None and (len(state)!=6 or not all(math.isfinite(v) for v in state)):
            raise ValueError('Initial robot state must contain six finite values')
        for k in ('follower_photo','camera2_photo'):
            photo=row[k]
            if not photo['path'] or not photo['sha256']:missing.append(f"{row['id']}.{k}");continue
            path=Path(photo['path']);path=path if path.is_absolute() else out/path
            if not path.is_file() or sha(path)!=photo['sha256']:raise ValueError('Photo evidence missing or changed')
    if lock['status']!='locked':missing.append('execution_lock.status')
    for k in ('act_checkpoint_sha256','smolvla_checkpoint_sha256','code_manifest_sha256',
              'follower_calibration_sha256','leader_calibration_sha256','camera_mount_reference',
              'independent_pilot_evidence','degree_unit_gate_validation_evidence','operator_authorization_reference','locked_utc'):
        if not lock[k]:missing.append(k)
        elif k.endswith('_sha256') and (len(lock[k])!=64 or any(c not in '0123456789abcdef' for c in lock[k])):
            raise ValueError('Identity hashes must be lowercase SHA256')
    for group in ('locked_parameters','common_limits'):
        for k,v in lock[group].items():
            if v is None:missing.append(f'{group}.{k}')
            elif k in ('lower','upper','max_speed_per_s') and not isinstance(v,list):
                raise ValueError('Per-joint limits require six-element lists')
            elif isinstance(v,list):
                if len(v)!=6 or not all(isinstance(x,(float,int)) and math.isfinite(x) for x in v):
                    raise ValueError('Limits must be six finite numbers')
                if k=='max_speed_per_s' and min(v)<=0:raise ValueError('Speeds must be positive')
            elif not isinstance(v,(float,int)) or not math.isfinite(v) or v<=0:
                raise ValueError('Locked timing/horizon values must be finite and positive')
    limits=lock['common_limits'];params=lock['locked_parameters']
    if limits['lower'] is not None and limits['upper'] is not None:
        if any(a>=b for a,b in zip(limits['lower'],limits['upper'])):raise ValueError('Lower limits must be below upper limits')
        if limits['lower'][-1]<0 or limits['upper'][-1]>100:raise ValueError('Gripper limits outside calibrated range')
    if params['predicted_steps'] is not None and params['execute_steps'] is not None:
        if type(params['predicted_steps']) is not int or type(params['execute_steps']) is not int or params['execute_steps']>params['predicted_steps']:
            raise ValueError('Invalid predicted/executed horizon')
    started=sum(r['status']!='not_run' for r in trials)
    success=sum(r['status']=='success' for r in trials)
    return {'record_structure':'passed','field_ready':not missing,'missing_field_evidence':missing,
            'slots':40,'started':started,'confirmed_successes':success,
            'success_fraction_among_started':success/started if started else None,
            'result_is_provisional':any(r['status']=='running' for r in trials),
            'hardware_authorization_granted_by_this_tool':False,
            **verify_trial_evidence(out,trials,initial,lock,not missing)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('init','validate'))
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--require-ready',action='store_true')
    args=parser.parse_args()
    if args.command=='init':initialize(args.directory)
    else:
        result=validate(args.directory);print(json.dumps(result,indent=2,ensure_ascii=False))
        if args.require_ready and not result['field_ready']:sys_exit=2
        else:sys_exit=0
        raise SystemExit(sys_exit)
