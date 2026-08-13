"""Current-scene static mat-boundary fixtures; no policy or hardware actions."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.tape_task_observer import observe_task
from cross_backend.tape_task import TapeTaskEvaluator
from cross_backend.png_write import write_png
SCENE=ROOT/'artifacts/sim-v1-scene-20260925/scene.xml'
OUT=ROOT/'artifacts/s1-current-scene-boundary-20260925';OUT.mkdir(exist_ok=False)
cases=[('outside',[-.18,-.15],'outside',False,'incomplete'),
       ('inside_initial',[-.18,.08],'inside',True,'invalid_initial_state'),
       ('edge_x',[-.12,.08],'boundary',True,'invalid_initial_state'),
       ('edge_y',[-.18,.15],'boundary',True,'invalid_initial_state')]
b=TapeSimBackend(SCENE);rows=[]
try:
    for name,xy,zone,on_mat,status in cases:
        b.reset(xy,robot_q=np.zeros(6))
        for _ in range(50):b.step_sim_targets(np.zeros(6))
        facts=observe_task(b);evaluator=TapeTaskEvaluator();evaluator.update(facts)
        actual=evaluator.finish()
        assert facts.zone==zone and facts.supported_on_mat==on_mat and actual==status,(name,facts,actual)
        image=b.render()['observation.images.follower'];write_png(OUT/f'{name}.png',image)
        rows.append({'case':name,'initial_tape_xy_m':xy,'facts':asdict(facts),'evaluator_status':actual,'policy_used':False,'hardware_access':False})
    (OUT/'results.json').write_text(json.dumps({'scene':str(SCENE),'cases':rows,'formal_trial':False,'physical_alignment_verified':False},indent=2)+'\n')
    print(json.dumps([{'case':r['case'],'zone':r['facts']['zone'],'supported_on_mat':r['facts']['supported_on_mat'],'status':r['evaluator_status']} for r in rows]))
finally:b.close()
