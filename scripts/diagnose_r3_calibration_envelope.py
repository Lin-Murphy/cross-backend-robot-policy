"""Static calibration-envelope rejection evidence and training/development diagnosis."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.move_pot_policy import MovePotChunk, ReplayChunkQueue
from cross_backend.degree_execution import DegreeGateRejected


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main(out,recovery):
    recovered=json.loads(recovery.read_text());bounds=recovered['bounds']
    lo=np.array([x['calibration_lower'] for x in bounds]);hi=np.array([x['calibration_upper'] for x in bounds])
    result={'scope':'calibration-envelope-only diagnostic; all predictions still blocked for execution',
            'calibration_sha256':recovered['calibration_sha256'],'recovery_sha256':sha(recovery),
            'models':{},'hardware_dispatches':0}
    source_dir=ROOT/'artifacts/r2-captured-dual-policy-20260924-attempt01'
    for model in ('act','smolvla'):
        predictions=[json.loads(s) for s in (source_dir/f'{model}-events.jsonl').read_text().splitlines()]
        predictions=[x for x in predictions if x['event']=='prediction']
        q=ReplayChunkQueue(out/f'{model}-calibration-rejections.jsonl',model+'-calibration-static-audit')
        counts=[]
        try:
            for row in predictions:
                actions=np.array(row['actions'])
                outside=[{'offset':int(i),'joint':bounds[j]['joint'],'value':float(actions[i,j]),
                          'lower':float(lo[j]),'upper':float(hi[j])} for i,j in np.argwhere((actions<lo)|(actions>hi))]
                counts.append(len(outside))
                q.enqueue(MovePotChunk(actions,row['source'],row['prediction_start_ns'],row['prediction_end_ns'],row['forward_ms'],row['processing_and_forward_ms']),50)
                def reject(action,event):
                    event['calibration_envelope_audit']={'calibration_sha256':recovered['calibration_sha256'],
                        'scope':'necessary recorded-range check, not workspace-safe limits','whole_chunk_violations':outside}
                    reasons=['execution_limits_not_validated','capture_times_unknown']
                    if outside: reasons.insert(0,'predicted_chunk_outside_recorded_calibration_envelope')
                    raise DegreeGateRejected(reasons)
                try:q.pop(context_validator=reject)
                except DegreeGateRejected:pass
                else:raise AssertionError('Audit cannot authorize dispatch')
                assert q.failed
                q.reset()
        finally:q.close()
        result['models'][model]={'per_sample_outside_scalar_counts':counts,'total_outside_scalars':sum(counts)}
    dataset=Path('/home/murphy/.cache/huggingface/lerobot/local/move_pot_20260923_142914/data/chunk-000/file-000.parquet')
    leader=Path('/home/murphy/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/so101_leader_arm.json')
    before={str(p):sha(p) for p in (dataset,leader)}
    import pyarrow.parquet as pq
    table=pq.read_table(dataset,columns=['action','observation.state','episode_index','frame_index'])
    episodes=np.asarray(table['episode_index']);frames=np.asarray(table['frame_index'])
    detail=[];summaries={}
    for key in ('action','observation.state'):
        values=np.array(table[key].to_pylist());assert values.shape==(10769,6) and np.isfinite(values).all()
        outside=(values<lo)|(values>hi)
        summaries[key]={'per_joint_outside_counts':outside.sum(axis=0).tolist(),
                        'frames_with_any_outside':int(outside.any(axis=1).sum()),
                        'min':values.min(axis=0).tolist(),'max':values.max(axis=0).tolist()}
        for i,j in np.argwhere(outside):
            detail.append({'field':key,'episode':int(episodes[i]),'frame':int(frames[i]),
                           'joint':bounds[j]['joint'],'value':float(values[i,j]),'lower':float(lo[j]),'upper':float(hi[j])})
    leader_cal=json.loads(leader.read_text());w=leader_cal['wrist_flex']
    result['dataset_diagnostic']={'role':'all training/development, not independent test','frame_count':10769,
        'joint_order':[x['joint'] for x in bounds],'summary':summaries,
        'leader_wrist_calibration_bounds_deg':[-(w['range_max']-w['range_min'])*180/4095,(w['range_max']-w['range_min'])*180/4095],
        'follower_wrist_calibration_bounds_deg':[float(lo[3]),float(hi[3])],
        'interpretation':'Recorded leader targets and follower readback are different fields; exceedances are not proof of training failure, collisions or a safe wider range.'}
    (out/'dataset-envelope-exceedances.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in detail))
    assert all(sha(Path(p))==h for p,h in before.items())
    result.update(status='passed',protected_input_sha256=before,protected_inputs_unchanged=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--recovery',type=Path,required=True)
    a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False);main(a.output,a.recovery)
