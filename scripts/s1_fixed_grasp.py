"""Single fixed-trajectory MuJoCo development attempt; no policy or hardware."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,json,sys,subprocess,time,traceback
from pathlib import Path
from dataclasses import asdict
import numpy as np
import mujoco
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.tape_task import TapeTaskEvaluator
from cross_backend.tape_task_observer import observe_task


def solve_ik(m,reference,position,rotation,start):
    d=mujoco.MjData(m);d.qpos[:]=reference;sid=m.site('gripperframe').id
    q=start.copy();best=None
    for _ in range(300):
        d.qpos[:6]=q;mujoco.mj_forward(m,d)
        actual=d.site_xmat[sid].reshape(3,3)
        pe=position-d.site_xpos[sid];re=.5*sum((np.cross(actual[:,i],rotation[:,i]) for i in range(3)),start=np.zeros(3))
        angle=float(np.arccos(np.clip((np.trace(rotation@actual.T)-1)/2,-1,1)))
        score=float(np.linalg.norm(pe)+.08*angle)
        if best is None or score<best[0]:best=(score,q.copy(),float(np.linalg.norm(pe)),angle)
        if np.linalg.norm(pe)<.002 and angle<.06:break
        jp=np.zeros((3,m.nv));jr=jp.copy();mujoco.mj_jacSite(m,d,jp,jr,sid)
        J=np.vstack([jp[:,:5],.08*jr[:,:5]]);err=np.r_[pe,.08*re]
        delta=np.linalg.solve(J.T@J+1e-4*np.eye(5),J.T@err)
        q[:5]+=np.clip(delta,-.12,.12)
        # Solver bounds constrain proposed waypoints; execution never clips policy actions.
        for i in range(5):q[i]=np.clip(q[i],max(m.jnt_range[i,0],m.actuator_ctrlrange[i,0]),min(m.jnt_range[i,1],m.actuator_ctrlrange[i,1]))
    return best

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scene',type=Path,default=ROOT/'artifacts/s1-base-spacing-20260925/scene.xml')
    parser.add_argument('--initial-wrist-roll-deg',type=float,default=None,help='override wrist roll; default is 0 without --initial-robot-q-rad')
    parser.add_argument('--initial-robot-q-rad',type=float,nargs=6,metavar=('PAN','LIFT','ELBOW','FLEX','ROLL','GRIP'),help='optional six-joint simulation start for C development')
    parser.add_argument('--mat-xy',type=float,nargs=2,metavar=('X_M','Y_M'),help='optional green-mat center in simulation meters; C development only')
    parser.add_argument('--grasp-radial-offset-m',type=float,default=.026,help='grasp reference offset toward base; C development only')
    parser.add_argument('--grasp-height-m',type=float,default=.045,help='grasp reference world z; C development only')
    parser.add_argument('--place-height-m',type=float,default=.046,help='place reference world z; C development only')
    parser.add_argument('--close-gripper-rad',type=float,default=-.14,help='closed gripper target in simulation radians; C development only')
    parser.add_argument('--side-approach-m',type=float,default=0.,help='open gripper lateral approach from robot side; C development only')
    args=parser.parse_args()
    if args.initial_wrist_roll_deg is not None and not np.isfinite(args.initial_wrist_roll_deg):parser.error('initial wrist roll must be finite')
    if args.initial_robot_q_rad is not None and not np.isfinite(args.initial_robot_q_rad).all():parser.error('initial robot q must be finite')
    if args.mat_xy is not None and not np.isfinite(args.mat_xy).all():parser.error('mat center must be finite')
    if not np.isfinite(args.grasp_radial_offset_m) or not -.03 <= args.grasp_radial_offset_m <= .05:parser.error('grasp radial offset must be -0.03..0.05 m')
    if not np.isfinite(args.grasp_height_m) or not .015 <= args.grasp_height_m <= .07:parser.error('grasp height must be 0.015..0.07 m')
    if not np.isfinite(args.place_height_m) or not .015 <= args.place_height_m <= .07:parser.error('place height must be 0.015..0.07 m')
    if not np.isfinite(args.close_gripper_rad) or not -.1745 <= args.close_gripper_rad <= 0:parser.error('closed gripper target outside allowed development range')
    if not np.isfinite(args.side_approach_m) or not 0 <= args.side_approach_m <= .12:parser.error('side approach must be 0..0.12 m')
    declared_wrist_deg=(args.initial_wrist_roll_deg if args.initial_wrist_roll_deg is not None else
                        float(np.rad2deg(args.initial_robot_q_rad[4])) if args.initial_robot_q_rad is not None else 0.)
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    (out/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    budget={'class':'C_development','attempts':1,'max_control_steps':950 if args.side_approach_m else 850,'control_period_s':.02,'video_fps':10,'hardware':False,'policy':False,'stop_on_collision':True,'wall_timeout_s':120,'scene':str(args.scene),'initial_wrist_roll_deg':declared_wrist_deg,'requested_mat_xy_m':args.mat_xy,'requested_initial_robot_q_rad':args.initial_robot_q_rad,'grasp_radial_offset_m':args.grasp_radial_offset_m,'grasp_height_m':args.grasp_height_m,'place_height_m':args.place_height_m,'close_gripper_rad':args.close_gripper_rad,'side_approach_m':args.side_approach_m}
    (out/'budget.json').write_text(json.dumps(budget,indent=2)+'\n')
    b=TapeSimBackend(args.scene);initial_q=np.asarray(args.initial_robot_q_rad if args.initial_robot_q_rad is not None else np.zeros(6),dtype=float);initial_q[4]=np.deg2rad(args.initial_wrist_roll_deg) if args.initial_wrist_roll_deg is not None else initial_q[4];b.reset(robot_q=initial_q,mat_xy=args.mat_xy);m=b.model
    result={'status':'planning','hardware_access':False,'policy_used':False,'initial_robot_q_rad':initial_q.tolist()};video=None;events=(out/'events.jsonl').open('x');started=time.monotonic()
    try:
        tape=b.data.xpos[b.tape_id].copy();base=b.data.xpos[m.body('base').id].copy();radial=base[:2]-tape[:2];radial/=np.linalg.norm(radial)
        target=tape.copy();target[:2]+=args.grasp_radial_offset_m*radial
        x=np.array([0.,0.,-1.])
        def waypoint_rotation(pos):
            radial_here=base[:2]-pos[:2];radial_here/=np.linalg.norm(radial_here)
            z=np.r_[-radial_here,0.];y=np.cross(z,x)
            return np.column_stack([x,y,z])
        mat=b.data.geom_xpos[m.geom('green_mat').id].copy();destination=mat.copy()
        mat_to_base=base[:2]-mat[:2];mat_to_base/=np.linalg.norm(mat_to_base)
        destination[:2]+=.031*mat_to_base
        result['initial_tape_xyz_m']=tape.tolist();result['mat_center_xyz_m']=mat.tolist();result['planned_destination_xyz_m']=destination.tolist()
        approach_xy=target[:2]+args.side_approach_m*radial
        approach_z=args.grasp_height_m if args.side_approach_m else .075
        targets=([('preapproach',np.r_[approach_xy,.075],.6,100)] if args.side_approach_m else []) + [('approach',np.r_[approach_xy,approach_z],.6,150),('descend',np.r_[target[:2],args.grasp_height_m],.6,100),('close',np.r_[target[:2],args.grasp_height_m],args.close_gripper_rad,60),('lift',np.r_[target[:2],.075],args.close_gripper_rad,100),('transfer',np.r_[destination[:2],.075],args.close_gripper_rad,150),('lower',np.r_[destination[:2],args.place_height_m],args.close_gripper_rad,100),('release',np.r_[destination[:2],args.place_height_m],.6,60),('retreat',np.r_[destination[:2],.075],.6,100)]
        plan=[];q=initial_q.copy()
        for name,pos,grip,steps in targets:
            rot=waypoint_rotation(pos)
            seeds=[q.copy(),np.array([.7,-.5,1.,.8,0.,grip]),np.array([.7,.5,-1.,-.8,0.,grip])]
            candidates=[]
            for seed in seeds:seed[5]=grip;candidates.append(solve_ik(m,b.data.qpos.copy(),pos,rot,seed))
            best=min(candidates,key=lambda a:a[0]);_,q,pe,angle=best
            plan.append({'phase':name,'position':pos.tolist(),'q':q.tolist(),'position_error_m':pe,'orientation_error_rad':angle,'steps':steps})
        (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        if any(p['position_error_m']>.004 or p['orientation_error_rad']>.15 for p in plan):
            result.update(status='rejected_ik_plan',reason='Waypoint error exceeds declared 4mm / 0.15rad');raise SystemExit(2)
        for _ in range(25):b.step_sim_targets(initial_q)
        evaluator=TapeTaskEvaluator();evaluator.update(observe_task(b));step=0
        video=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x480','-r','10','-i','-','-an','-c:v','libx264','-crf','23','-pix_fmt','yuv420p',str(out/'dual-camera.mp4')],stdin=subprocess.PIPE)
        for p in plan:
            q0=b.data.ctrl.copy();goal=np.array(p['q'])
            for k in range(p['steps']):
                u=(k+1)/p['steps'];u=u*u*(3-2*u);command=q0+(goal-q0)*u
                if step >= budget['max_control_steps'] or time.monotonic()-started > budget['wall_timeout_s']:
                    raise RuntimeError('demo budget exceeded')
                state=b.step_sim_targets(command);facts=observe_task(b);status=evaluator.update(facts)
                events.write(json.dumps({'step':step,'phase':p['phase'],'target_sim_rad':command.tolist(),'state':state,'facts':asdict(facts),'task_status':status})+'\n');step+=1
                if step%5==0 or status not in ('running','not_started'):
                    images=b.render();video.stdin.write(np.concatenate(list(images.values()),axis=1).tobytes())
                if status not in ('running','not_started'):break
            if evaluator.status not in ('running','not_started'):break
        result.update(status=evaluator.finish(),task_events=evaluator.events,control_steps=step,simulation_trial_started=True)
    except Exception as exc:result.update(status='execution_error',error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if video:
            video.stdin.close();result['video_exit']=video.wait(timeout=15)
        events.close();b.close();result['wall_s']=time.monotonic()-started
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if result['status'] != 'success' or result.get('video_exit') != 0:
        raise SystemExit(2)
if __name__=='__main__':main()
