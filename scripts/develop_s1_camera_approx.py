"""Render an isolated, approximate follower camera candidate from saved training frames."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,hashlib,json,sys,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.png_write import write_png
source=ROOT/'artifacts/s1-base-spacing-20260925/scene.xml'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--focus-x',type=float,required=True)
parser.add_argument('--focus-y',type=float,required=True)
parser.add_argument('--distance',type=float,required=True)
args=parser.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
root=ET.parse(source)
cam=root.find(".//camera[@name='follower']")
assert cam is not None
# Chosen from a rough comparison to training episode 0, not measured optics.
yaw=np.deg2rad(30);tilt=np.deg2rad(35)
right=np.array([-np.cos(yaw),-np.sin(yaw),0.])
up=np.array([np.sin(yaw)*np.sin(tilt),-np.cos(yaw)*np.sin(tilt),np.cos(tilt)])
back=np.cross(right,up)
focus=np.array([args.focus_x,args.focus_y,0.]);distance=args.distance
pos=focus+distance*back
cam.set('pos',' '.join(f'{v:.12g}' for v in pos))
cam.set('xyaxes',' '.join(f'{v:.12g}' for v in np.r_[right,up]))
cam.set('fovy','38')
scene=out/'scene.xml';root.write(scene,encoding='unicode')
b=TapeSimBackend(scene)
try:
 b.reset();images=b.render();write_png(out/'follower.png',images['observation.images.follower']);write_png(out/'camera2.png',images['observation.images.camera2'])
finally:b.close()
manifest={'classification':'C_camera_approximation','physical_alignment_verified':False,'model_trial_count':0,'source_scene_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'focus_world':focus.tolist(),'distance_m':distance,'camera_pos':pos.tolist(),'camera_xyaxes':np.r_[right,up].tolist(),'camera_fovy_deg':38,'assumptions':'training episode0 visual rough target; no measured K or extrinsics','scene_sha256':hashlib.sha256(scene.read_bytes()).hexdigest()}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'output':str(out),'focus_world':focus.tolist(),'distance_m':distance,'camera_pos':pos.tolist()}))
