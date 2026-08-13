"""Map one audited legacy SO101 development rollout to the shared RunRecord."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.run_record import RunRecord
from cross_backend.legacy_receipt_bridge import bridge_legacy_receipts


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build_record(run,events,verdict,source,evidence):
    if run['scope']!='single_legacy30_development_pilot_not_formal' or run['formal_trial_count']!=0:
        raise ValueError('not an authorized single development rollout')
    transmitted=sum(e['event']=='legacy_goal_packet_transmitted' for e in events)
    candidates=sum(e['event']=='legacy_goal_candidate' for e in events)
    if transmitted!=run['raw_goal_packets_transmitted'] or candidates<transmitted:
        raise ValueError('inconsistent raw target evidence')
    if run['status']=='ended_unreviewed':
        if verdict not in ('success','failed','uncertain') or not source:
            raise ValueError('operator verdict and source required')
        status=verdict;outcome=True if verdict=='success' else False if verdict=='failed' else None
        outcome_source=source if outcome is not None else None
        feedback=True
    elif run['status'] in ('gate_rejected','aborted'):
        status='rejected' if transmitted==0 else 'aborted'
        if transmitted==0 and verdict is not None:
            raise ValueError('zero-dispatch rejection cannot have a task verdict')
        if verdict is not None and (verdict not in ('success','failed','uncertain') or not source):
            raise ValueError('operator verdict and source required')
        # Preserve control interruption even if an on-site operator observed
        # the physical task complete before the guard stopped the rollout.
        outcome=True if verdict=='success' else False if verdict=='failed' else None
        outcome_source=source if outcome is not None else None
        feedback=verdict is not None
    else:raise ValueError('unrecognized run status')
    policy={'smolvla':'SmolVLA','SmolVLA':'SmolVLA','act':'ACT'}.get(run.get('policy','SmolVLA'))
    if policy is None:raise ValueError('unsupported or missing policy identity')
    record=RunRecord(1,'so101','move pot',policy,'development_trial',status,outcome,outcome_source,
        candidates,transmitted,transmitted,feedback,False,'host_monotonic',evidence)
    return record.to_dict()


def main(run_dir,out,verdict,source,operator_observation=None):
    out.mkdir(parents=True,exist_ok=False)
    summary_path=run_dir/'summary.json';event_path=run_dir/'hardware-events.jsonl'
    summary=json.loads(summary_path.read_text())
    events=[json.loads(line) for line in event_path.read_text().splitlines()]
    capabilities,receipts=bridge_legacy_receipts(events,summary['raw_goal_packets_transmitted'])
    if summary.get('shared_boundary_enabled'):
        shared=[event for event in events if event['event']=='shared_backend_action_receipt']
        observations=[event for event in events if event['event']=='shared_backend_observation']
        if len(shared)!=summary['raw_goal_packets_transmitted'] or len(observations)<len(shared) or any(
            not event['receipt']['accepted'] or event['receipt']['physical_dispatches']!=1 or
            event['receipt']['ack_scope']!='sync_transport_return' for event in shared):
            raise ValueError('shared live boundary events do not match audited transports')
    evidence={'summary':sha(summary_path),'hardware_events':sha(event_path),'profile':sha(run_dir/'profile.json')}
    info=run_dir/'dataset'/'meta'/'info.json'
    if info.exists():evidence['dataset_info']=sha(info)
    if verdict is not None:
        if operator_observation is None or not operator_observation.is_file():
            raise ValueError('operator observation file required for verdict')
        evidence['operator_observation']=sha(operator_observation)
    record=build_record(summary,events,verdict,source,evidence)
    result={'record':record,'raw_run_dir':str(run_dir),'verdict':verdict,'verdict_source':source,
            'formal_trial_count':0,'per_motor_goal_acknowledgements_verified':False,
            'physical_dispatches_meaning':'successful sync packet transport return, not per-servo acknowledgement',
            'raw_evidence_unchanged':True,
            'shared_receipt_contract_validated':True,
            'shared_receipt_ack_scope':capabilities['dispatch_ack_scope'],
            'shared_receipt_accepted':sum(r['accepted'] for r in receipts),
            'shared_receipt_rejected':sum(not r['accepted'] for r in receipts)}
    (out/'evaluation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':'passed','trial_status':record['status'],'outcome':record['outcome']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verdict',choices=['success','failed','uncertain']);p.add_argument('--source')
    p.add_argument('--operator-observation',type=Path)
    a=p.parse_args();main(a.run,a.output,a.verdict,a.source,a.operator_observation)
