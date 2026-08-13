"""Validate paired fixed-queue delay outcomes from saved artifacts only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args()
    p=args.run;r=json.loads((p/'result.json').read_text());protocol=json.loads((p/'protocol.json').read_text())
    assert r['status'] in {'completed','completed_with_errors'} and r['schema_version']==2
    assert len(r['episodes'])==len(protocol['trials'])==10
    assert r['trial_plan']==protocol['trials']
    assert json.loads((p/'verification.json').read_text())['all_checks_passed']
    assert r['checkpoint_sha256']==protocol['weight_sha256']
    assert r['checkpoint_files_before']==r['checkpoint_files_after']
    for filename,digest in r['source_sha256'].items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest
    assert r['history_capacity']==6 and r['execute_steps']==10 and r['prediction_steps']==100
    assert set(r['native_queue_verifications'])=={'10'}
    assert r['native_queue_verifications']['10']['max_abs_error']<=1e-4
    traces={}
    rejected_checks=[]
    for e,t in zip(r['episodes'],protocol['trials']):
        assert all(e[k]==t[k] for k in ['seed','execute_steps','delay_steps'])
        d=p/f"episode_{e['episode']:03d}"
        rr=[json.loads(s) for s in (d/'trace.jsonl').read_text().splitlines()]
        traces[e['episode']]=rr
        assert len(e['chunks'])==(len(rr)+9)//10 + int(e['termination']=='error' and len(rr)%10==0)
        assert e['reset']=={'policy':True,'preprocessor':True,'postprocessor':True,'queue_empty':True}
        for i,row in enumerate(rr):
            assert row['latest_observation_step']==i and row['chunk_id']==i//10
            assert row['queue_remaining']==9-i%10
            if i:
                assert row['state_before']==rr[i-1]['state_after']
                assert row['observation_capture_monotonic_ns']==rr[i-1]['next_capture_monotonic_ns']
            assert row['observation_capture_monotonic_ns']<=row['dispatch_monotonic_ns']<=row['next_capture_monotonic_ns']
        for cid,c in enumerate(e['chunks']):
            replan=cid*10;source=max(0,replan-e['delay_steps'])
            assert c['chunk_id']==cid and c['replan_step']==replan and c['source_step']==source
            assert c['actual_input_delay_steps']==replan-source
            expected_capture=rr[source]['observation_capture_monotonic_ns'] if source<len(rr) else rr[-1]['next_capture_monotonic_ns']
            assert c['source_captured_monotonic_ns']==expected_capture
            assert c['source_simulation_time_s']==source/10 and c['predicted_shape']==[100,2]
        assert not any(x['terminated'] or x['truncated'] for x in rr[:-1])
        if e['termination']=='error':
            rejected=json.loads((d/'rejected_action.json').read_text())
            assert rejected['step']==len(rr) and not any(x['terminated'] or x['truncated'] for x in rr)
            assert not e['success']
            with np.load(d/f"chunk_{rejected['chunk_id']:03d}.npz") as c:
                assert np.array_equal(c['actions'][rejected['offset']],np.array(rejected['action'],dtype=np.float32))
                assert np.isfinite(rejected['action']).all() and any(v<0 or v>512 for v in rejected['action'])
                source=int(c['source_step']);replan=int(c['replan_step'])
                assert source==max(0,replan-e['delay_steps']) and replan==rejected['step']
                expected_state=rr[source]['state_before'] if source<len(rr) else rr[-1]['state_after']
                assert np.array_equal(c['input_state'],np.array(expected_state))
                if source<len(rr):
                    assert hashlib.sha256(c['input_rgb'].tobytes()).hexdigest()==rr[source]['observation_rgb_sha256']
                rejected_checks.append({'episode':e['episode'],'seed':e['seed'],'delay_steps':e['delay_steps'],
                    'rejected_step':rejected['step'],'rejected_action':rejected['action'],'not_executed':True,
                    'saved_prediction_matches':True,'state_source_verified':True,
                    'rgb_source_independently_verified':source<len(rr),
                    'limit':None if source<len(rr) else 'Final current RGB exists in chunk NPZ but has no independent trace hash because command was rejected before that row was written.'})
        elif e['success']:
            assert e['termination']=='success' and rr[-1]['success'] and rr[-1]['terminated']
        else:
            assert e['steps']==300 and e['termination']=='time_limit' and rr[-1]['truncated']
    pairs=[]
    for seed in protocol['seeds']:
        es={e['delay_steps']:e for e in r['episodes'] if e['seed']==seed}
        x,y=es[0],es[5]
        assert x['initial_rgb_sha256']==y['initial_rgb_sha256'] and x['initial_state']==y['initial_state']
        ds={delay:p/f"episode_{e['episode']:03d}" for delay,e in es.items()}
        with np.load(ds[0]/'chunk_000.npz') as a,np.load(ds[5]/'chunk_000.npz') as b:
            assert np.array_equal(a['actions'],b['actions'])
        a,b=traces[x['episode']],traces[y['episode']]
        for u,v in zip(a[:10],b[:10]):
            for key in ['action','state_before','state_after','coverage','observation_rgb_sha256']:
                assert u[key]==v[key]
        first=next((i for i in range(min(len(a),len(b))) if a[i]['action']!=b[i]['action']),None)
        assert first is None or first>=10
        pairs.append({'seed':seed,'initial_state_rgb_equal':True,'first_chunk_equal':True,'first_10_steps_equal':True,
           'first_action_difference_step':first,
           'conditions':{str(delay):{k:e[k] for k in ['episode','success','steps','termination','max_coverage','final_coverage']} for delay,e in es.items()},
           'success_delta_delayed_minus_zero':int(y['success'])-int(x['success']),
           'max_coverage_delta_delayed_minus_zero':y['max_coverage']-x['max_coverage'],
           'final_coverage_delta_delayed_minus_zero':y['final_coverage']-x['final_coverage']})
    conditions={}
    for delay in protocol['delay_conditions']:
        es=[e for e in r['episodes'] if e['delay_steps']==delay]
        chunks=[c for e in es for c in e['chunks']];rows=[row for e in es for row in traces[e['episode']]]
        steps=len(rows)
        conditions[str(delay)]={'successes':sum(e['success'] for e in es),'trials':len(es),'steps':steps,
            'rejected_trials':sum(e['termination']=='error' for e in es),
            'timeouts':sum(e['termination']=='time_limit' for e in es),
            'mean_max_coverage':float(np.mean([e['max_coverage'] for e in es])),
            'mean_final_coverage':float(np.mean([e['final_coverage'] for e in es])),
            'successful_completion_sim_s':[e['steps']/10 for e in es if e['success']],
            'full_chunk_calls':len(chunks),'calls_per_100_steps':100*len(chunks)/steps,
            'forward_median_ms':float(np.median([c['forward_ms'] for c in chunks])),
            'forward_ms_per_100_steps':100*sum(c['forward_ms'] for c in chunks)/steps,
            'processed_ms_per_100_steps':100*sum(c['processing_and_forward_ms'] for c in chunks)/steps,
            'max_action_source_age_sim_s':max(row['action_source_age_sim_s'] for row in rows),
            'mean_action_source_age_sim_s':float(np.mean([row['action_source_age_sim_s'] for row in rows])),
            'realized_replan_input_delays_steps':sorted(set(c['actual_input_delay_steps'] for c in chunks)),
            'peak_allocated_MiB':max(e['gpu_peak_allocated_bytes'] for e in es)/2**20}
    summary={'all_available_checks_passed':True,'rejected_action_checks':rejected_checks,
        'coverage_limit':'Means include observed prefixes of the two rejected trials; their max/final coverage is not a full 300-step outcome.',
        'total_environment_steps':sum(e['steps'] for e in r['episodes']),
        'total_video_frames':sum(e['video_frames'] for e in r['episodes']),
        'pairs':pairs,'conditions':conditions,
        'success_rate_delta_delayed_minus_zero':float(np.mean([x['success_delta_delayed_minus_zero'] for x in pairs])),
        'mean_paired_max_coverage_delta':float(np.mean([x['max_coverage_delta_delayed_minus_zero'] for x in pairs])),
        'mean_paired_final_coverage_delta':float(np.mean([x['final_coverage_delta_delayed_minus_zero'] for x in pairs])),
        'no_additional_rollouts':True,'analysis_unit':'5 paired seeds; exploratory; no hardware timing or training-independence claim'}
    (p/'paired_analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
