"""Fixed C-development budget: 3 resets and 4 simulated target trials."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_scene import build_scene
from cross_backend.tape_sim_backend import TapeSimBackend
from check_s0_mujoco_readiness import png
OUT=ROOT/'artifacts/s1-tape-scene-20260925'

def main():
    out=OUT/'run02';out.mkdir(exist_ok=False)
    spec=json.loads((ROOT/'configs/s1-tape-scene-development.json').read_text())
    budget={'purpose':'C_development_not_policy_or_formal_trials','reset_cases':3,'fixed_target_trials':4,'steps_per_case':50,'control_period_s':.02,'policy_trials':0,'hardware_access':False}
    (out/'budget.json').write_text(json.dumps(budget,indent=2)+'\n')
    geom=build_scene(ROOT/'artifacts/s0-inventory-20260925/upstream',out/'scene.xml',spec)
    (out/'geometry.json').write_text(json.dumps(geom,indent=2)+'\n')
    backend=TapeSimBackend(out/'scene.xml');result={'status':'running','cases':[],'geometry':geom};t=time.monotonic()
    events=(out/'events.jsonl').open('x')
    try:
        for i,xy in enumerate([[-.18,-.15],[-.23,-.15],[-.13,-.15]]):
            state=backend.reset(xy)
            for k in range(50):
                state=backend.step_sim_targets(np.zeros(6));events.write(json.dumps({'case':f'reset_{i}','step':k,**state})+'\n')
            z=state['tape_position'][2]
            assert abs(z-.025)<.002,('Tape failed settle height',z)
            assert np.linalg.norm(np.array(state['tape_position'][:2])-xy)<.002
            assert any('tape_sector' in str(c) and 'table' in str(c) for c in state['contacts'])
            assert min(c['distance_m'] for c in state['contacts'])>-.002
            result['cases'].append({'name':f'reset_{i}','final':state,'status':'passed'})
        for label,joint,value in [('pan_positive',0,.03),('pan_negative',0,-.03),('gripper_open',5,.1),('gripper_hold',5,0.)]:
            backend.reset();target=np.zeros(6);target[joint]=value
            before=backend.data.qpos.copy()
            # Invalid actions and unknown policy mapping must not alter simulation state.
            for bad in [np.full(6,np.nan),np.full(6,10.),np.zeros(5)]:
                try:backend.step_sim_targets(bad)
                except ValueError:pass
                else:raise AssertionError('Invalid target accepted')
            try:backend.step_policy_action(np.zeros(6))
            except RuntimeError:pass
            else:raise AssertionError('Unaligned policy dispatch accepted')
            assert np.array_equal(before,backend.data.qpos) and backend.data.time==0
            for k in range(50):
                state=backend.step_sim_targets(target);events.write(json.dumps({'case':label,'step':k,'target_sim_rad':target.tolist(),**state})+'\n')
            assert abs(backend.data.time-1)<1e-9
            assert abs(state['joint_position'][joint]-value)<.01
            assert not any(w.number for w in backend.data.warning)
            result['cases'].append({'name':label,'status':'passed','final':state})
        # Isolate tape collision geometry: a vertical ray through its hole must miss,
        # while a ray through the annulus must hit. No floor/robot in this group.
        center=backend.data.xpos[backend.tape_id];groups=np.array([0,0,0,0,0,1],dtype=np.uint8)
        # Upstream also uses group4 for jaw meshes; restrict the ray-hit result by body.
        ray_results=[]
        for offset in [0.,.0415]:
            gid=np.array([-1],dtype=np.int32)
            distance=mujoco.mj_ray(backend.model,backend.data,np.array([center[0]+offset,center[1],.2]),np.array([0.,0.,-1.]),groups,1,-1,gid)
            ray_results.append({'offset_m':offset,'distance':float(distance),'geom_id':int(gid[0])})
        assert ray_results[0]['distance']==-1
        assert ray_results[1]['distance']>0
        assert backend.model.geom_bodyid[ray_results[1]['geom_id']]==backend.tape_id
        assert abs(backend.model.body_mass[backend.tape_id]-.02)<1e-12
        images=backend.render()
        for name,a in images.items():
            assert a.shape==(480,640,3) and a.dtype==np.uint8 and a.std()>1
            png(out/(name.split('.')[-1]+'.png'),a)
        # Semantic visibility check: tape pixels must appear in the overview.
        option=mujoco.MjvOption();option.geomgroup[5]=1
        backend.renderer.enable_segmentation_rendering()
        backend.renderer.update_scene(backend.data,camera='follower',scene_option=option)
        seg=backend.renderer.render().copy()
        backend.renderer.disable_segmentation_rendering()
        tape_geoms=np.flatnonzero(backend.model.geom_bodyid==backend.tape_id)
        tape_pixels=int(np.count_nonzero(np.isin(seg[:,:,0],tape_geoms) & (seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))))
        assert tape_pixels>50,('Tape not sufficiently visible',tape_pixels)
        result['overview_tape_pixels']=tape_pixels
        result.update(status='passed_development_geometry_checks',rays=ray_results,body_mass_kg=float(backend.model.body_mass[backend.tape_id]),physics_steps=1400,policy_dispatch_blocked=True,physical_alignment_validated=False)
    except BaseException as exc:
        result.update(status='failed',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        events.close();backend.close();result['wall_s']=time.monotonic()-t
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))

if __name__=='__main__':main()
