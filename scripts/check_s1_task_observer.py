"""Evaluator integration fixtures; no policy or successful grasp trial."""
import json,sys
from pathlib import Path
from dataclasses import asdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.tape_task import TapeTaskEvaluator,TaskLimits
from cross_backend.tape_task_observer import observe_task
OUT=ROOT/'artifacts/s1-task-evaluator-20260925'

def main():
    limits=TaskLimits();(OUT/'development-limits.json').write_text(json.dumps({'limits':asdict(limits),'status':'synthetic_development_not_frozen_or_real_safe','mat_boundary_margin_m':.001,'budget':'two static integration fixtures; each 50 settle + 10 observed control steps; no policy trials'},indent=2)+'\n')
    b=TapeSimBackend(ROOT/'artifacts/s1-base-spacing-20260925/scene.xml')
    results=[]
    try:
        for name,xy,expected_zone,expected_status in [('outside',[-.18,-.15],'outside','incomplete'),('already_on_mat',[-.18,.08],'inside','invalid_initial_state')]:
            b.reset(xy)
            for _ in range(50):b.step_sim_targets(np.zeros(6))
            evaluator=TapeTaskEvaluator(limits);records=[]
            for _ in range(10):
                b.step_sim_targets(np.zeros(6));f=observe_task(b);evaluator.update(f);records.append(asdict(f))
            status=evaluator.finish()
            results.append({'fixture':name,'status':status,'facts':records,'events':evaluator.events,'policy_trial':False})
            assert records[-1]['zone']==expected_zone,records[-1]
            assert status==expected_status,(status,records[-1])
        (OUT/'integration.json').write_text(json.dumps({'passed':True,'fixtures':results,'hardware_access':False,'physical_grasp_verified':False},indent=2)+'\n')
        print('PASS: static outside stays incomplete; already-on-mat initial state rejected; no task success claimed.')
    finally:
        b.close()
        (OUT/'integration-events.json').write_text(json.dumps(results,indent=2)+'\n')
if __name__=='__main__':main()
