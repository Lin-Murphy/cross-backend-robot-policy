"""Attach an operator verdict to one completed SO101 pilot without modifying raw evidence."""
import argparse
import hashlib
import json
from pathlib import Path
from cross_backend.run_record import RunRecord


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def build_record(run,verdict,source):
    if run['scope']!='single_bounded_development_pilot_not_formal' or run['formal_trial_count']!=0:
        raise ValueError('not a single development pilot')
    if run['status']=='ended_unreviewed':
        if verdict not in ('success','failed','uncertain') or not source:
            raise ValueError('operator verdict and source required')
        status=verdict;outcome=True if verdict=='success' else False if verdict=='failed' else None
        outcome_source=source if outcome is not None else None
        feedback=True
    elif run['status'] in ('gate_rejected','aborted'):
        if verdict is not None:raise ValueError('aborted/rejected run cannot be relabeled as task outcome')
        status='rejected' if run['status']=='gate_rejected' and run['applied_actions']==0 else 'aborted'
        outcome=None;outcome_source=None;feedback=False
    else:raise ValueError('unrecognized raw run status')
    return status,outcome,outcome_source,feedback


def main(run_dir,out,verdict,source):
    out.mkdir(parents=True,exist_ok=False)
    path=run_dir/'summary.json';run=json.loads(path.read_text())
    status,outcome,outcome_source,feedback=build_record(run,verdict,source)
    evidence={'summary':sha(path),'hardware_events':sha(run_dir/'hardware-events.jsonl'),
              'queue_events':sha(run_dir/'queue-events.jsonl')}
    record=RunRecord(1,'so101','move pot','SmolVLA','development_trial',status,outcome,outcome_source,
        run['predictions']*50,run['applied_actions'],run['applied_actions'],feedback,False,
        'host_monotonic',evidence).to_dict()
    payload={'record':record,'raw_run_dir':str(run_dir),'verdict':verdict,
             'verdict_source':source,'formal_trial_count':0,'raw_evidence_unchanged':True}
    (out/'evaluation.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'passed','trial_status':status,'outcome':outcome},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--verdict',choices=['success','failed','uncertain'])
    p.add_argument('--source')
    a=p.parse_args();main(a.run,a.output,a.verdict,a.source)
