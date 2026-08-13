"""Replay saved simulated targets into a C-only obstacle scene."""
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
SCENE=ROOT/'artifacts/s1-current-scene-collision-20260925/scene.xml'
SOURCE=ROOT/'artifacts/sim-v1-demo-20260925T132612Z-al1eoV/run/events.jsonl'
OUT=ROOT/'artifacts/s1-current-scene-collision-20260925/run01';OUT.mkdir(exist_ok=False)
commands=[np.array(json.loads(x)['target_sim_rad'],dtype=float) for x in SOURCE.read_text().splitlines()]
b=TapeSimBackend(SCENE,.02);b.reset(robot_q=np.zeros(6))
for _ in range(25):b.step_sim_targets(np.zeros(6))
evaluator=TapeTaskEvaluator();evaluator.update(observe_task(b));assert evaluator.status=='running'
video=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x480','-r','10','-i','-','-an','-c:v','libx264','-crf','23','-pix_fmt','yuv420p',str(OUT/'dual-camera.mp4')],stdin=subprocess.PIPE)
records=[]
def frame():video.stdin.write(np.concatenate(list(b.render().values()),axis=1).tobytes())
frame()
try:
    for step,target in enumerate(commands[:200]):
        state=b.step_sim_targets(target);facts=observe_task(b);status=evaluator.update(facts)
        records.append({'step':step,'target_sim_rad':target.tolist(),'facts':asdict(facts),'status':status,'contacts':state['contacts']})
        if step%5==4 or status!='running':frame()
        if status!='running':break
    assert evaluator.status=='collision' and records[-1]['step']==58
    assert records[-1]['facts']['forbidden_contact']
    assert any('c_obstacle' in (x['geom1'],x['geom2']) for x in records[-1]['contacts'])
    (OUT/'events.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    result={'status':evaluator.status,'first_collision_step':records[-1]['step'],'task_events':evaluator.events,'collision_contacts':[x for x in records[-1]['contacts'] if 'c_obstacle' in (x['geom1'],x['geom2'])],'source_events_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'scene_sha256':b.scene_sha256,'control_steps':len(records),'policy_used':False,'hardware_access':False,'formal_trial':False}
finally:
    video.stdin.close();video_exit=video.wait(timeout=15);b.close()
assert video_exit==0
result['video_exit']=video_exit
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'first_collision_step':result['first_collision_step'],'contacts':result['collision_contacts'],'video_exit':video_exit}))
