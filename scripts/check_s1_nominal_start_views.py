"""Render saved demo starts under a nominal, unverified coordinate mapping."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.png_write import write_png
source=json.loads((ROOT/'artifacts/s1-camera-approx-20260925-run06/training-start-states.json').read_text())
out=ROOT/'artifacts/s1-nominal-initial-state-20260925';out.mkdir(exist_ok=False)
records=[]
for label,scene in [('original',ROOT/'artifacts/s1-base-spacing-20260925/scene.xml'),('follower_approx',ROOT/'artifacts/s1-camera-approx-20260925-run02/scene.xml')]:
 b=TapeSimBackend(scene)
 try:
  gr=b.model.jnt_range[b.model.joint('gripper').id]
  for ep in [0,3]:
   p=np.asarray(source['episode_starts'][str(ep)],dtype=float)
   q=np.r_[np.deg2rad(p[:5]),gr[0]+p[5]/100*(gr[1]-gr[0])]
   b.reset(robot_q=q)
   images=b.render();folder=out/f'{label}-ep{ep:02d}';folder.mkdir()
   for cam,img in images.items():write_png(folder/(cam.rsplit('.',1)[-1]+'.png'),img)
   cid=b.model.camera('camera2').id
   pos=b.data.cam_xpos[cid].copy();forward=-b.data.cam_xmat[cid].reshape(3,3)[:,2]
   hit=(pos-pos[2]/forward[2]*forward).tolist() if forward[2]<0 else None
   records.append({'scene':label,'episode':ep,'input_policy_start':p.tolist(),'nominal_sim_q':q.tolist(),'camera2_forward':forward.tolist(),'camera2_table_hit':hit,'camera_alignment_verified':False})
 finally:b.close()
(out/'manifest.json').write_text(json.dumps({'purpose':'static start-view diagnostic, no policy rollout','training_start_source_sha256':hashlib.sha256((ROOT/'artifacts/s1-camera-approx-20260925-run06/training-start-states.json').read_bytes()).hexdigest(),'records':records},indent=2)+'\n')
print(json.dumps([{'scene':r['scene'],'episode':r['episode'],'forward_z':r['camera2_forward'][2],'table_hit':r['camera2_table_hit']} for r in records]))
