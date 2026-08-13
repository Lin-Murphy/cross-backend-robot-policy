"""Replay saved fixed targets, then open the simulated gripper at two heights."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import hashlib,json,subprocess,sys
from dataclasses import asdict
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.tape_task_observer import observe_task
from cross_backend.tape_task import TapeTaskEvaluator
SCENE=ROOT/'artifacts/sim-v1-scene-20260925/scene.xml'
SOURCE=ROOT/'artifacts/sim-v1-demo-20260925T132612Z-al1eoV/run/events.jsonl'
OUT=ROOT/'artifacts/s1-current-scene-drop-20260925';OUT.mkdir(exist_ok=False)
commands=[np.array(json.loads(x)['target_sim_rad'],dtype=float) for x in SOURCE.read_text().splitlines()]
assert len(commands)==736
source_sha=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
results=[]
for name,release_after_step,expected in [('near_lift_threshold',347,'placement_outside'),('high_release',409,'drop')]:
    dest=OUT/name;dest.mkdir();b=TapeSimBackend(SCENE,.02);b.reset(robot_q=np.zeros(6))
    for _ in range(25):b.step_sim_targets(np.zeros(6))
    evaluator=TapeTaskEvaluator();evaluator.update(observe_task(b));events=[]
    video=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x480','-r','10','-i','-','-an','-c:v','libx264','-crf','23','-pix_fmt','yuv420p',str(dest/'dual-camera.mp4')],stdin=subprocess.PIPE)
    def frame():video.stdin.write(np.concatenate(list(b.render().values()),axis=1).tobytes())
    frame()
    try:
        for step,target in enumerate(commands[:release_after_step+1]):
            b.step_sim_targets(target);facts=observe_task(b);status=evaluator.update(facts)
            events.append({'step':step,'phase':'saved_fixed_replay','facts':asdict(facts),'status':status})
            if step%5==4:frame()
        assert evaluator.lifted and evaluator.status=='running'
        release_facts=asdict(observe_task(b))
        held=commands[release_after_step].copy();held[5]=.6
        for j in range(25):
            b.step_sim_targets(held);facts=observe_task(b);status=evaluator.update(facts)
            events.append({'step':release_after_step+1+j,'phase':'open_gripper_and_hold_arm','facts':asdict(facts),'status':status})
            if j%5==4 or status!='running':frame()
            if status!='running':break
        assert evaluator.status==expected,(name,evaluator.events)
        assert not any(e['facts']['forbidden_contact'] or e['facts']['numerical_error'] for e in events)
        (dest/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        row={'case':name,'source_events_sha256':source_sha,'release_after_step':release_after_step,'release_facts':release_facts,'status':evaluator.status,'task_events':evaluator.events,'control_steps':len(events),'policy_used':False,'hardware_access':False,'formal_trial':False}
        results.append(row)
    finally:
        video.stdin.close();exit_code=video.wait(timeout=15);b.close()
        assert exit_code==0
    (dest/'summary.json').write_text(json.dumps(row,indent=2)+'\n')
(OUT/'summary.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps([{'case':r['case'],'status':r['status'],'events':r['task_events'],'video_exit':0} for r in results]))
