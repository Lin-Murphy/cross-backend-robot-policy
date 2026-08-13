"""Current-scene 0/1-frame scheduler preflight with a synthetic zero predictor."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.move_pot_policy import ReplayChunkQueue
from cross_backend.sim_chunk_schedule import SimChunkSchedule
SCENE=ROOT/'artifacts/sim-v1-scene-20260925/scene.xml'
OUT=ROOT/'artifacts/sim-v1-delay-preflight-20260925';OUT.mkdir(exist_ok=False)
rows=[]
for delay in (0,1):
    dest=OUT/f'delay{delay}';dest.mkdir()
    b=TapeSimBackend(SCENE,1/30);b.reset(robot_q=np.zeros(6))
    q=ReplayChunkQueue(dest/'queue.jsonl',f'sim-v1-synthetic-delay{delay}')
    schedule=SimChunkSchedule(q,lambda obs:np.zeros((50,6)),latency_ns=0,observation_delay_frames=delay,clock_id=f'sim-v1-synthetic-delay{delay}')
    try:
        initial=json.dumps(b.snapshot(),sort_keys=True).encode();held=b.data.ctrl.copy()
        for tick in range(55):
            now=round(b.data.time*1e9)
            schedule.capture(now,{'observation_id':f'delay{delay}:frame{tick}'})
            action=schedule.tick(now,lambda a,e:b._validate_targets(a))
            if action is not None:held=action
            b.step_sim_targets(held)
        jobs=[e for e in schedule.events if e['event']=='prediction_scheduled']
        selected=[e for e in schedule.events if e['event']=='sim_target_selected']
        assert jobs and selected and all(e['realized_delay_frames']==delay for e in jobs)
        assert all(e['source']['realized_delay_frames']==delay for e in selected)
        assert all(e['sim_ns']>=e['source']['available_sim_ns'] for e in selected)
        row={'delay_frames':delay,'initial_snapshot_sha256':hashlib.sha256(initial).hexdigest(),'predictions':len(jobs),'selected_targets':len(selected),'first_request_source_age_ns':jobs[0]['request_sim_ns']-jobs[0]['capture_sim_ns'],'first_selected_source_age_ns':selected[0]['source_age_ns'],'realized_delay_frames':sorted(set(e['realized_delay_frames'] for e in jobs)),'policy_used':False,'hardware_access':False,'formal_trial':False}
        rows.append(row)
        (dest/'summary.json').write_text(json.dumps(row,indent=2)+'\n')
        (dest/'schedule.json').write_text(json.dumps(schedule.events,indent=2)+'\n')
    finally:
        schedule.reset();q.close();b.close()
assert rows[0]['initial_snapshot_sha256']==rows[1]['initial_snapshot_sha256']
assert rows[0]['first_request_source_age_ns']==0
assert rows[1]['first_request_source_age_ns']>0
(OUT/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows,indent=2))
