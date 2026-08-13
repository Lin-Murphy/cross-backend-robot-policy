"""Static candidate: robot base centered between tape and mat, wrist roll -90 degrees."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.png_write import write_png
source=ROOT/'artifacts/s1-base-spacing-20260925/scene.xml';out=ROOT/'artifacts/s1-between-objects-20260925';out.mkdir(exist_ok=False)
root=ET.parse(source)
base=root.find(".//body[@name='base']");tape=root.find(".//body[@name='tape']");mat=root.find(".//geom[@name='green_mat']")
tape_xy=np.fromstring(tape.get('pos'),sep=' ')[:2];mat_xy=np.fromstring(mat.get('pos'),sep=' ')[:2];midpoint=(tape_xy+mat_xy)/2
base.set('pos',f'{midpoint[0]:.12g} {midpoint[1]:.12g} 0')
scene=out/'scene.xml';root.write(scene,encoding='unicode')
b=TapeSimBackend(scene)
try:
 q=np.zeros(6);q[4]=-np.pi/2
 state=b.reset(robot_q=q);images=b.render()
 for camera,image in images.items():write_png(out/(camera.rsplit('.',1)[-1]+'.png'),image)
 cid=b.model.camera('camera2').id;pos=b.data.cam_xpos[cid].copy();forward=-b.data.cam_xmat[cid].reshape(3,3)[:,2]
 result={'classification':'static_development_candidate','scene_source':str(source),'base_xy':midpoint.tolist(),'tape_xy':tape_xy.tolist(),'mat_xy':mat_xy.tolist(),'wrist_roll_rad':q[4],'camera2_world_pos':pos.tolist(),'camera2_forward':forward.tolist(),'contacts':state['contacts'],'policy_trial_count':0,'hardware_access':False,'physical_alignment_verified':False}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'base_xy':result['base_xy'],'contacts':len(result['contacts']),'camera2_forward':result['camera2_forward']}))
finally:b.close()
