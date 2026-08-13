"""Paired analysis of saved queue trials; never runs inference or simulation."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
p = ROOT/'artifacts/sim-v1-act-queue-20260921/run01'
r = json.loads((p/'result.json').read_text())
protocol = json.loads((p/'protocol.json').read_text())
assert r['status'] == 'completed'
assert r['trial_plan'] == protocol['trials']
assert len(r['episodes']) == 10
assert all(v['max_abs_error'] <= 1e-4 for v in r['native_queue_verifications'].values())
assert set(r['native_queue_verifications']) == {'100', '10'}
assert json.loads((p/'verification.json').read_text())['all_checks_passed']
assert hashlib.sha256((ROOT/'configs/sim-v1-act-queue-ablation.json').read_bytes()).hexdigest() == r['protocol_sha256']
pairs=[]
for seed in protocol['seeds']:
    es={e['execute_steps']:e for e in r['episodes'] if e['seed']==seed}
    a,b=es[100],es[10]
    assert a['initial_state']==b['initial_state'] and a['initial_rgb_sha256']==b['initial_rgb_sha256']
    dirs={h:p/f"episode_{e['episode']:03d}" for h,e in es.items()}
    with np.load(dirs[100]/'chunk_000.npz') as x, np.load(dirs[10]/'chunk_000.npz') as y:
        error=float(np.max(np.abs(x['actions']-y['actions'])))
        assert error==0
    traces={h:[json.loads(s) for s in (d/'trace.jsonl').read_text().splitlines()] for h,d in dirs.items()}
    for x,y in zip(traces[100][:10],traces[10][:10]):
        for k in ['action','state_before','state_after','coverage','observation_rgb_sha256']:
            assert x[k]==y[k], (seed,k)
    pairs.append({'seed':seed,'initial_conditions_equal':True,'first_prediction_max_abs_error':error,
                  'first_10_steps_equal':True,'conditions':{str(h):{k:e[k] for k in ['episode','success','steps','termination','max_coverage','final_coverage']} for h,e in es.items()},
                  'success_delta_10_minus_100':int(b['success'])-int(a['success']),
                  'max_coverage_delta_10_minus_100':b['max_coverage']-a['max_coverage']})
conditions={}
for h in [100,10]:
    es=[e for e in r['episodes'] if e['execute_steps']==h]
    chunks=[c for e in es for c in e['chunks']]
    assert all(c['predicted_shape']==[100,2] for c in chunks)
    steps=sum(e['steps'] for e in es)
    ages=[json.loads(s)['action_source_age_sim_s'] for e in es for s in (p/f"episode_{e['episode']:03d}"/'trace.jsonl').read_text().splitlines()]
    conditions[str(h)]={'successes':sum(e['success'] for e in es),'trials':len(es),'steps':steps,
       'full_chunk_calls':len(chunks),'calls_per_100_steps':100*len(chunks)/steps,
       'forward_median_ms':float(np.median([c['forward_ms'] for c in chunks])),
       'forward_ms_per_100_steps':100*sum(c['forward_ms'] for c in chunks)/steps,
       'processed_chunk_ms_per_100_steps':100*sum(c['processing_and_forward_ms'] for c in chunks)/steps,
       'max_action_source_age_sim_s':max(ages),'mean_action_source_age_sim_s':float(np.mean(ages)),
       'mean_max_coverage':float(np.mean([e['max_coverage'] for e in es])),
       'mean_final_coverage':float(np.mean([e['final_coverage'] for e in es])),
       'successful_completion_sim_s':[e['steps']/10 for e in es if e['success']],
       'peak_allocated_MiB':max(e['gpu_peak_allocated_bytes'] for e in es)/2**20}
old=json.loads((ROOT/'artifacts/sim-v1-act-pilot-20260921/run01/result.json').read_text())
for e in old['episodes']:
    new=next(x for x in r['episodes'] if x['seed']==e['seed'] and x['execute_steps']==100)
    assert all(e[k]==new[k] for k in ['steps','success','max_coverage','final_coverage'])
summary={'all_paired_checks_passed':True,'pairs':pairs,'conditions':conditions,
         'success_rate_delta_10_minus_100':(conditions['10']['successes']-conditions['100']['successes'])/5,
         'mean_paired_max_coverage_delta':float(np.mean([x['max_coverage_delta_10_minus_100'] for x in pairs])),
         'original_pilot_100_step_outcomes_reproduced':True,'analysis_unit':'5 paired seeds; exploratory, 2 reused pilot seeds',
         'total_environment_steps':sum(e['steps'] for e in r['episodes']),
         'no_additional_rollouts':True}
(p/'paired_analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
