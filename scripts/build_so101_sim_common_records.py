"""Map existing sim-v1 results and SO101 real-candidate rejection to one checked envelope."""
import argparse
import hashlib
import json
from pathlib import Path
from cross_backend.run_record import RunRecord


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main(sim,real,out):
    out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for policy in ('ACT','SmolVLA'):
        name=policy.lower();path=sim/name/'summary.json';data=json.loads(path.read_text())
        if data['status']!='timeout' or data['hardware_access'] or data['formal_trial']:
            raise ValueError('unexpected sim-v1 source')
        count=data['selected_actions']
        predicted=data['predictions']*50  # Existing ACT/SmolVLA move-pot adapters declare 50-action chunks.
        if count>predicted:raise ValueError('selected count exceeds chunk predictions')
        rows.append(RunRecord(1,'mujoco','move pot',policy,'development_trial','timeout',False,
            'sim_task_events',predicted,count,0,True,False,'simulation',{'summary':sha(path)}).to_dict())
    path=real/'summary.json';data=json.loads(path.read_text())
    if data['status']!='passed' or data['hardware_dispatches']!=0 or data['formal_trial_count']!=0:
        raise ValueError('unexpected SO101 source')
    rows.append(RunRecord(1,'so101','move pot','SmolVLA','saved_candidate_rejection','rejected',None,None,
        data['recorded_actions'],0,0,False,False,None,{'summary':sha(path),'events':sha(real/'events.jsonl')}).to_dict())
    payload={'schema_version':1,'scope':'common_result_envelope_validation_not_paired_performance',
             'rows':rows,'source_paths':{'sim':str(sim),'so101':str(real)},
             'interpretation':'Common identity/outcome/action provenance fields validated; simulation and real execution remain separate'}
    (out/'records.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps({'status':'passed','rows':len(rows),'backends':sorted({r['backend'] for r in rows}),
                      'physical_dispatches':sum(r['physical_dispatches'] for r in rows)},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sim',type=Path,required=True)
    p.add_argument('--real',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.sim,a.real,a.output)
