"""Bounded upstream-model readiness probe, not move-pot task evaluation."""
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import zlib
os.environ.setdefault('MUJOCO_GL','egl')
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/s0-inventory-20260925'

def png(path, a):
    def chunk(k,b):return struct.pack('!I',len(b))+k+b+struct.pack('!I',zlib.crc32(k+b)&0xffffffff)
    h,w,_=a.shape
    raw=b''.join(b'\0'+row.tobytes() for row in a)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('!IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))

def main():
    start=time.monotonic();result={'scope':'upstream compile/physics/render readiness only','task_evaluation':False,'hardware_access':False,'policy_used':False,'mujoco_version':mujoco.__version__}
    try:
        m=mujoco.MjModel.from_xml_path(str(OUT/'upstream/scene.xml'));d=mujoco.MjData(m)
        mujoco.mj_forward(m,d)
        names=[mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_JOINT,i) for i in range(m.njnt)]
        assert names==['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper']
        q0=d.qpos.copy();d.ctrl[:]=q0
        for _ in range(200):
            mujoco.mj_step(m,d)
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():raise ValueError('Nonfinite state')
        result.update(nq=m.nq,nv=m.nv,nu=m.nu,joints=names,physics_steps=200,sim_time_s=d.time,initial_qpos=q0.tolist(),final_qpos=d.qpos.tolist(),contacts_final=d.ncon,warnings=[int(w.number) for w in d.warning],images={})
        with mujoco.Renderer(m,height=480,width=640) as renderer:
            cam=mujoco.MjvCamera();mujoco.mjv_defaultFreeCamera(m,cam)
            cam.lookat[:]=[0,0,.15];cam.distance=.65;cam.azimuth=135;cam.elevation=-25
            for label,camera in [('upstream-overview',cam),('upstream-wrist','wrist_cam')]:
                renderer.update_scene(d,camera=camera);a=renderer.render().copy()
                assert a.shape==(480,640,3) and a.dtype==np.uint8
                png(OUT/(label+'.png'),a)
                result['images'][label]={'shape':list(a.shape),'std':float(a.std()),'sha256':hashlib.sha256(a.tobytes()).hexdigest(),'calibrated_to_real':False}
                if a.std()<1:raise ValueError('Near-constant render')
        result['status']='passed_upstream_readiness_only'
    except Exception as e:
        result.update(status='failed',error=repr(e));raise
    finally:
        result['wall_s']=time.monotonic()-start
        (OUT/'readiness.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))

if __name__=='__main__':main()
