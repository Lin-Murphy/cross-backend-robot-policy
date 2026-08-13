"""Current scene camera2 visibility under zero and saved nominal demo starts."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.png_write import write_png


def main():
    scene=ROOT/'artifacts/sim-v1-scene-20260925/scene.xml'
    starts=ROOT/'artifacts/s1-camera-approx-20260925-run06/training-start-states.json'
    out=ROOT/'artifacts/s1-current-camera2-start-views-20260925'
    out.mkdir(exist_ok=False)
    source=json.loads(starts.read_text())
    backend=TapeSimBackend(scene)
    records=[]
    try:
        gr=backend.model.jnt_range[backend.model.joint('gripper').id]
        cases=[('zero',np.zeros(6),None)]
        for ep in (0,3):
            policy=np.asarray(source['episode_starts'][str(ep)],dtype=float)
            nominal=np.r_[np.deg2rad(policy[:5]),gr[0]+policy[5]/100*(gr[1]-gr[0])]
            cases.append((f'ep{ep:02d}_nominal',nominal,policy.tolist()))
        for label,q,policy in cases:
            backend.reset(robot_q=q)
            images=backend.render()
            for name,im in images.items():write_png(out/f'{label}-{name.rsplit(".",1)[-1]}.png',im)
            option=mujoco.MjvOption();option.geomgroup[5]=1
            backend.renderer.enable_segmentation_rendering()
            backend.renderer.update_scene(backend.data,camera='camera2',scene_option=option)
            seg=backend.renderer.render().copy()
            backend.renderer.disable_segmentation_rendering()
            geom=seg[:,:,0];obj=seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM)
            tape_ids=np.flatnonzero(backend.model.geom_bodyid==backend.tape_id)
            mat=backend.model.geom('green_mat').id
            cid=backend.model.camera('camera2').id
            pos=backend.data.cam_xpos[cid].copy()
            forward=-backend.data.cam_xmat[cid].reshape(3,3)[:,2]
            hit=(pos-pos[2]/forward[2]*forward).tolist() if forward[2]<0 else None
            records.append({'case':label,'nominal_mapping_unverified':policy is not None,
                            'policy_start':policy,'sim_q_rad':q.tolist(),
                            'camera2_position_m':pos.tolist(),'camera2_forward_world':forward.tolist(),
                            'camera2_table_intersection_m':hit,
                            'camera2_tape_pixels':int(np.count_nonzero(np.isin(geom,tape_ids)&obj)),
                            'camera2_mat_pixels':int(np.count_nonzero((geom==mat)&obj)),
                            'images':{n:f'{label}-{n}.png' for n in ('follower','camera2')}})
    finally:backend.close()
    result={'stage':'C_static_visibility_diagnostic','scene_sha256':backend.scene_sha256,
            'training_start_file_sha256':hashlib.sha256(starts.read_bytes()).hexdigest(),
            'hardware_access':False,'model_inference':False,'physical_mapping_verified':False,
            'records':records}
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{'case':r['case'],'forward_z':r['camera2_forward_world'][2],
                       'tape_pixels':r['camera2_tape_pixels'],'mat_pixels':r['camera2_mat_pixels']}
                      for r in records]))


if __name__=='__main__':main()
