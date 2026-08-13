"""Set a clearly downward looking wrist camera in a separate development scene."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.png_write import write_png
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source',type=Path,default=ROOT/'artifacts/s1-camera-approx-20260925-run02/scene.xml')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--forward-offset-m',type=float,required=True)
args=parser.parse_args()
source=args.source.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
m=mujoco.MjModel.from_xml_path(str(source));d=mujoco.MjData(m);mujoco.mj_forward(m,d)
c=m.camera('camera2').id;body=m.cam_bodyid[c]
right=np.array([-.8660254038,-.5,0.])
theta=np.arcsin(.9)
up=np.array([.5*np.sin(theta),-.8660254038*np.sin(theta),np.cos(theta)])
back=np.cross(right,up)
world=np.column_stack((right,up,back))
local=d.xmat[body].reshape(3,3).T@world
quat=np.zeros(4);mujoco.mju_mat2Quat(quat,local.reshape(9))
root=ET.parse(source);cam=root.find(".//camera[@name='camera2']");assert cam is not None
cam.set('quat',' '.join(f'{v:.12g}' for v in quat))
local_shift=d.xmat[body].reshape(3,3).T@(-back*args.forward_offset_m)
new_local_pos=m.cam_pos[c]+local_shift
cam.set('pos',' '.join(f'{v:.12g}' for v in new_local_pos))
scene=out/'scene.xml';root.write(scene,encoding='unicode')
b=TapeSimBackend(scene)
try:
 b.reset();images=b.render();write_png(out/'follower.png',images['observation.images.follower']);write_png(out/'camera2.png',images['observation.images.camera2'])
 pos=b.data.cam_xpos[b.model.camera('camera2').id];forward=-b.data.cam_xmat[b.model.camera('camera2').id].reshape(3,3)[:,2]
 assert forward[2]<-.8,(pos,forward)
 intersection=pos-pos[2]/forward[2]*forward
finally:b.close()
(out/'manifest.json').write_text(json.dumps({'classification':'C_camera2_approximation','physical_alignment_verified':False,'training_reference':'camera2 episodes 0-2 starts; desk visible; tape sometimes visible','source_scene':str(source),'camera_local_quat':quat.tolist(),'camera_local_pos':new_local_pos.tolist(),'forward_offset_m':args.forward_offset_m,'camera_world_forward_at_nominal_zero':forward.tolist(),'table_intersection_at_nominal_zero':intersection.tolist(),'model_trial_count':0},indent=2)+'\n')
print(json.dumps({'output':str(out),'forward':forward.tolist(),'table_intersection':intersection.tolist()}))
