"""MuJoCo + shared queue scheduling diagnostic; latency profiles, not model trials."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.move_pot_policy import ReplayChunkQueue
from cross_backend.sim_chunk_schedule import SimChunkSchedule
OUT=ROOT/'artifacts/s1-discrete-schedule-20260925'
summary=[]
for profile,latency in [('fast_profile',6_300_000),('slow_profile',113_400_000)]:
    for delay in [0,33_333_333]:
        name=f'{profile}-delay{delay}';out=OUT/name;out.mkdir(exist_ok=False)
        (out/'budget.json').write_text(json.dumps({'max_ticks':180,'selected_targets':150,'modeled_latency_ns':latency,'delay_ns':delay,'input':'synthetic identity/no images','predictor':'zero target stub','profile_source':'representative R2 historical forward medians; declared simulation cost, not current model execution','policy_used':False},indent=2)+'\n')
        b=TapeSimBackend(ROOT/'artifacts/s1-base-spacing-20260925/scene.xml',1/30);b.reset()
        q=ReplayChunkQueue(out/'queue.jsonl',name)
        s=SimChunkSchedule(q,lambda obs:np.zeros((50,6)),latency_ns=latency,observation_delay_ns=delay,clock_id=name)
        ticks=[];selected=0;held=b.data.ctrl.copy()
        try:
            for tick in range(180):
                now=round(b.data.time*1e9);s.capture(now,{'observation_id':f'{name}:{tick}'})
                action=s.tick(now,lambda a,e:b._validate_targets(a))
                if action is not None:held=action;selected+=1
                state=b.step_sim_targets(held)
                ticks.append({'tick':tick,'selection_time_ns':now,'advanced_to_ns':state['sim_ns'],'new_target':action is not None,'tape_position':state['tape_position']})
                if selected==150:break
            assert selected==150
            dispatch=[e for e in s.events if e['event']=='sim_target_selected']
            for i in range(3):assert [e['chunk_offset'] for e in dispatch[i*50:(i+1)*50]]==list(range(50))
            assert all(e['sim_ns']>=e['source']['available_sim_ns'] and e['source_age_ns']>=delay for e in dispatch)
            assert all(t['advanced_to_ns']>t['selection_time_ns'] for t in ticks)
            swaps=[dispatch[i*50]['sim_ns']-dispatch[i*50-1]['sim_ns'] for i in (1,2)]
            row={'case':name,'selected':selected,'control_ticks':len(ticks),'hold_ticks':sum(not t['new_target'] for t in ticks),'swap_intervals_ns':swaps,'maximum_source_age_ns':max(e['source_age_ns'] for e in dispatch),'simulation_end_ns':round(b.data.time*1e9),'policy_used':False}
            summary.append(row)
            (out/'ticks.json').write_text(json.dumps(ticks,indent=2)+'\n')
            (out/'summary.json').write_text(json.dumps(row,indent=2)+'\n')
        finally:
            s.reset();(out/'schedule-events.json').write_text(json.dumps(s.events,indent=2)+'\n');q.close();b.close()
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
