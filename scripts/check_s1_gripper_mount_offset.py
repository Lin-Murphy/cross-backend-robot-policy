"""Development-only 90-degree gripper mounting offset with nominal wrist zero."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.png_write import write_png
source=ROOT/'artifacts/s1-relocated-wrist-camera-20260925/scene.xml';out=ROOT/'artifacts/s1-gripper-mount-offset-20260925';out.mkdir(exist_ok=False)
root=ET.parse(source);grip=root.find(".//body[@name='gripper']");assert grip is not None
original=np.fromstring(grip.get('quat'),sep=' ');axis=np.array([np.cos(-np.pi/4),0,0,np.sin(-np.pi/4)]);new=np.zeros(4);mujoco.mju_mulQuat(new,original,axis)
grip.set('quat',' '.join(f'{v:.16g}' for v in new));scene=out/'scene.xml';root.write(scene,encoding='unicode')
old=TapeSimBackend(source);candidate=TapeSimBackend(scene)
try:
 q_old=np.zeros(6);q_old[4]=-np.pi/2
 old.reset(robot_q=q_old);candidate.reset(robot_q=np.zeros(6))
 old_images=old.render();new_images=candidate.render()
 comparison={name:{'max_abs_pixel_difference':int(np.max(np.abs(old_images[name].astype(int)-new_images[name].astype(int)))),'equal_pixels':bool(np.array_equal(old_images[name],new_images[name]))} for name in old_images}
 for name,img in new_images.items():write_png(out/(name.rsplit('.',1)[-1]+'.png'),img)
 camera_pos_diff=float(np.max(np.abs(old.data.cam_xpos[old.model.camera('camera2').id]-candidate.data.cam_xpos[candidate.model.camera('camera2').id])))
 physical_tip_diff=float(np.max(np.abs(old.data.site_xpos[old.model.site('gripperframe').id]-candidate.data.site_xpos[candidate.model.site('gripperframe').id])))
 result={'class':'C_nominal_coordinate_hypothesis','source_scene':str(source),'mount_quat_original':original.tolist(),'mount_quat_candidate':new.tolist(),'mount_rotation_rad':-np.pi/2,'old_initial_wrist_q_rad':-np.pi/2,'candidate_initial_wrist_q_rad':0.,'images':comparison,'camera_pos_max_diff_m':camera_pos_diff,'gripperframe_pos_max_diff_m':physical_tip_diff,'physical_mapping_verified':False,'policy_trials':0}
 (out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({'images':comparison,'camera_pos_diff':camera_pos_diff,'tip_diff':physical_tip_diff}))
finally:old.close();candidate.close()
