"""Static user-directed scene candidate with a base-top second camera."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.png_write import write_png
source=ROOT/'artifacts/s1-base-spacing-20260925/scene.xml';out=ROOT/'artifacts/s1-overhead-camera-candidate-20260925';out.mkdir(exist_ok=False)
root=ET.parse(source)
base=root.find(".//body[@name='base']");base.set('pos','-0.04 -0.035 0')
wrist_camera=root.find(".//camera[@name='camera2']");assert wrist_camera is not None;wrist_camera.set('name','wrist_camera_reference')
ET.SubElement(base,'camera',name='camera2',pos='0 0 0.55',xyaxes='-1 0 0 0 -1 0',fovy='61')
scene=out/'scene.xml';root.write(scene,encoding='unicode')
b=TapeSimBackend(scene)
try:
 q=np.zeros(6);q[4]=-np.pi/2;state=b.reset(robot_q=q);images=b.render()
 for camera,image in images.items():write_png(out/(camera.rsplit('.',1)[-1]+'.png'),image)
 pos=b.data.cam_xpos[b.model.camera('camera2').id].copy();direction=-b.data.cam_xmat[b.model.camera('camera2').id].reshape(3,3)[:,2]
 result={'classification':'static_camera_above_base_candidate','source_scene':str(source),'base_xy':[-.04,-.035],'base_equally_distant_along_tape_mat_axis':True,'base_lateral_clearance_m':.14,'wrist_roll_rad':q[4],'camera2_parent':'base','camera2_world_pos':pos.tolist(),'camera2_forward':direction.tolist(),'wrist_camera_preserved_as':'wrist_camera_reference','contacts':state['contacts'],'training_camera_semantics_changed':True,'model_trial_count':0,'hardware_access':False}
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'camera2_pos':pos.tolist(),'camera2_forward':direction.tolist(),'contacts':len(state['contacts'])}))
finally:b.close()
