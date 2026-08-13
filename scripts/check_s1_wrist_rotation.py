"""Static +/-90 degree wrist-roll view check; no control trajectory."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.png_write import write_png
scene=ROOT/'artifacts/s1-base-spacing-20260925/scene.xml'
out=ROOT/'artifacts/s1-wrist-roll-90-20260925';out.mkdir(exist_ok=False)
starts=json.loads((ROOT/'artifacts/s1-camera-approx-20260925-run06/training-start-states.json').read_text())['episode_starts']
b=TapeSimBackend(scene);records=[]
try:
 gr=b.model.jnt_range[b.model.joint('gripper').id]
 for label,base in [('zero',np.zeros(6)),('training_ep0_nominal',None)]:
  if base is None:
   p=np.asarray(starts['0']);base=np.r_[np.deg2rad(p[:5]),gr[0]+p[5]/100*(gr[1]-gr[0])]
  for sign in (-1,1):
   q=base.copy();q[4]+=sign*np.pi/2
   name=f'{label}_{"plus" if sign>0 else "minus"}90'
   try:state=b.reset(robot_q=q)
   except ValueError as exc:records.append({'case':name,'rejected':str(exc)});continue
   images=b.render();folder=out/name;folder.mkdir()
   for camera,image in images.items():write_png(folder/(camera.rsplit('.',1)[-1]+'.png'),image)
   cid=b.model.camera('camera2').id;pos=b.data.cam_xpos[cid].copy();forward=-b.data.cam_xmat[cid].reshape(3,3)[:,2]
   hit=(pos-pos[2]/forward[2]*forward).tolist() if forward[2]<0 else None
   records.append({'case':name,'sim_joint_q':q.tolist(),'camera2_forward':forward.tolist(),'camera2_table_hit':hit,'contacts':state['contacts'],'physical_alignment_verified':False})
finally:b.close()
(out/'manifest.json').write_text(json.dumps({'scope':'static nominal wrist-roll diagnostics only','body_pose_mapping_verified':False,'camera_location_unchanged':True,'records':records},indent=2)+'\n')
print(json.dumps([{'case':r['case'],'rejected':r.get('rejected'),'forward_z':r.get('camera2_forward',[None,None,None])[2],'contact_count':len(r.get('contacts',[]))} for r in records]))
