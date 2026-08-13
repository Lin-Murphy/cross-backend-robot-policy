"""Static relocated-base view with the original wrist-mounted camera intact."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.png_write import write_png
source=ROOT/'artifacts/s1-overhead-camera-candidate-20260925/scene.xml';out=ROOT/'artifacts/s1-relocated-wrist-camera-20260925';out.mkdir(exist_ok=False)
root=ET.parse(source)
root.find(".//camera[@name='wrist_camera_reference']").set('name','camera2')
base=root.find(".//body[@name='base']");base.remove(base.find("./camera[@name='camera2']"))
scene=out/'scene.xml';root.write(scene,encoding='unicode')
starts=json.loads((ROOT/'artifacts/s1-camera-approx-20260925-run06/training-start-states.json').read_text())['episode_starts']
b=TapeSimBackend(scene);records=[]
try:
 gr=b.model.jnt_range[b.model.joint('gripper').id]
 p=np.asarray(starts['0']);q=np.r_[np.deg2rad(p[:5]),gr[0]+p[5]/100*(gr[1]-gr[0])]
 for name,roll in [('training_ep0_nominal',q[4]),('training_ep0_minus90',-np.pi/2)]:
  trial=q.copy();trial[4]=roll;state=b.reset(robot_q=trial)
  folder=out/name;folder.mkdir();images=b.render()
  for camera,image in images.items():write_png(folder/(camera.rsplit('.',1)[-1]+'.png'),image)
  cid=b.model.camera('camera2').id;pos=b.data.cam_xpos[cid].copy();direction=-b.data.cam_xmat[cid].reshape(3,3)[:,2]
  records.append({'case':name,'roll_deg':float(np.rad2deg(roll)),'camera_world_pos':pos.tolist(),'camera_forward':direction.tolist(),'contacts':state['contacts']})
finally:b.close()
(out/'manifest.json').write_text(json.dumps({'purpose':'static wrist-camera preservation check','base_scene_source':str(source),'camera2_on_wrist':True,'policy_trials':0,'physical_alignment_verified':False,'records':records},indent=2)+'\n')
print(json.dumps([{'case':r['case'],'forward_z':r['camera_forward'][2],'contacts':len(r['contacts'])} for r in records]))
