"""Persist replay and rational-clock evidence without policies/hardware."""
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
out=ROOT/'artifacts/s1-state-clock-20260925'
b=TapeSimBackend(ROOT/'artifacts/s1-base-spacing-20260925/scene.xml',1/30)
try:
    b.reset(tape_xy=[-.17,-.14],mat_xy=[-.2,.09])
    for _ in range(20):b.step_sim_targets(np.zeros(6))
    snap=b.snapshot();(out/'snapshot.json').write_text(json.dumps(snap,indent=2)+'\n')
    first=[b.step_sim_targets([.02,0,0,0,0,.1]) for _ in range(20)]
    b.reset();b.restore(snap);replayed=[b.step_sim_targets([.02,0,0,0,0,.1]) for _ in range(20)]
    error=max(np.max(np.abs(np.array(a['qpos'])-c['qpos'])) for a,c in zip(first,replayed))
    assert error<1e-12
    (out/'replay.json').write_text(json.dumps({'original':first,'replayed':replayed},indent=2)+'\n')
    b.reset();timing=[]
    for _ in range(300):
        s=b.step_sim_targets(np.zeros(6));timing.append({k:s[k] for k in ['sim_ns','scheduled_control_ns','control_lateness_ns']})
    assert timing[-1]['sim_ns']==10_000_000_000
    (out/'timing.json').write_text(json.dumps(timing,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps({'replay_max_qpos_error':float(error),'contacts_equal':all(a['contacts']==c['contacts'] for a,c in zip(first,replayed)),'clock_final_ns':timing[-1]['sim_ns'],'maximum_control_lateness_ns':max(x['control_lateness_ns'] for x in timing),'hardware_access':False,'policy_used':False,'physical_alignment_verified':False},indent=2)+'\n')
finally:b.close()
