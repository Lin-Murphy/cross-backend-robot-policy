"""C-only nominal-map physics pilot; does not unlock validated-policy or hardware paths."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import argparse,sys,json,subprocess,selectors,time,traceback,re
from pathlib import Path
from dataclasses import asdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.sim_trial_initial import apply_initial_condition
from cross_backend.sim_prediction_identity import verify_prediction_response
from cross_backend.sim_chunk_schedule import SimChunkSchedule
from cross_backend.move_pot_policy import ReplayChunkQueue,MovePotChunk,file_sha256
from cross_backend.tape_task import TapeTaskEvaluator
from cross_backend.tape_task_observer import observe_task
OUT=ROOT/'artifacts/s1-policy-development-pilot-20260925'
SCENE=ROOT/'assets/so101/tape-base-offset.xml'
INITIAL_WRIST_ROLL_DEG=0.
OBSERVATION_DELAY_FRAMES=0
INITIAL_CONDITION=None
TRIAL_PREFIX="C-development"
POLICY_PYTHON=os.environ.get('LEROBOT_PYTHON',sys.executable)
CHECKPOINT=None
DEVICE='cuda'
SEED=1000
MAX_CONTROL_TICKS=600
ALIGNMENT_SPEC=None
CALIBRATION=ROOT/'artifacts/r3-real-clock-timing-20260924/calibration-recovery/current-calibration.json'

class PolicyGateRejected(ValueError):
    """A recorded control stop, not a policy task verdict."""


def run(name):
    trial_id=f'{TRIAL_PREFIX}-{name}'
    out=OUT/name;out.mkdir(exist_ok=False)
    budget={'trial_id':trial_id,'purpose':'C nominal-mapping development pilot, not physical alignment or formal evaluation','max_control_ticks':MAX_CONTROL_TICKS,'horizon':50,'prefetch_remaining':5,'maximum_predictions':15,'control_hz':30,'source_age_limit_ns':2_500_000_000,'step_limit_body_deg':20,'step_limit_gripper_units':25,'limits_status':'development-only unvalidated','models_per_trial':1,'seed_smolvla':SEED,'hardware_access':False,'latency':'measured host callback includes IPC/file I/O; injected once in sim time','mapping':'nominal same-sign zero-offset deg/rad; gripper asset range linear 0..100','camera_alignment_verified':False,'scene':str(SCENE),'initial_wrist_roll_deg':INITIAL_WRIST_ROLL_DEG,'observation_delay_frames':OBSERVATION_DELAY_FRAMES}
    b=TapeSimBackend(SCENE,1/30)
    mapping=None;alignment_metadata={}
    if ALIGNMENT_SPEC is not None:
        from cross_backend.sim_alignment import load_sim_alignment
        mapping,alignment_metadata=load_sim_alignment(ALIGNMENT_SPEC,b)
        if file_sha256(CALIBRATION)!=alignment_metadata['robot_calibration_sha256']:
            b.close();raise ValueError('Alignment robot calibration does not match execution calibration')
        budget.update(alignment_metadata,mapping='measured affine coordinate mapping')
    if INITIAL_CONDITION is not None:
        initial_metadata=apply_initial_condition(INITIAL_CONDITION,b,expected_split='C')
        initial_q=b.data.qpos[b.qadr].copy()
        budget.update(initial_condition_id=initial_metadata['initial_condition_id'],
                      initial_condition_file=str(INITIAL_CONDITION),
                      initial_file_sha256=initial_metadata['initial_file_sha256'],
                      initial_geometry_sha256=initial_metadata['initial_geometry_sha256'],
                      initial_wrist_roll_deg=float(np.rad2deg(initial_q[4])))
    else:
        initial_metadata=None
        initial_q=np.zeros(6);initial_q[4]=np.deg2rad(INITIAL_WRIST_ROLL_DEG)
        b.reset(robot_q=initial_q)
    (out/'budget.json').write_text(json.dumps(budget,indent=2)+'\n')
    worker_log=(out/'worker.log').open('w');worker=subprocess.Popen([POLICY_PYTHON,'-B',str(ROOT/'scripts/s1_policy_worker.py'),name,str(out),trial_id,'--device',DEVICE,'--seed',str(SEED)]+(['--checkpoint',str(CHECKPOINT)] if CHECKPOINT else []),stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=worker_log,text=True,bufsize=1)
    def reply(timeout=90):
        selector=selectors.DefaultSelector();selector.register(worker.stdout,selectors.EVENT_READ)
        try:
            if not selector.select(timeout):raise TimeoutError('Policy worker response timed out')
            line=worker.stdout.readline()
            if not line:raise RuntimeError('Policy worker exited unexpectedly')
            return json.loads(line)
        finally:selector.close()
    q=ReplayChunkQueue(out/'queue.jsonl',trial_id);events=(out/'events.jsonl').open('x');video=None;schedule=None
    result={'trial_id':trial_id,'status':'initializing','hardware_access':False,'formal_trial':False,'physical_alignment_verified':False,'initial_condition_id':initial_metadata['initial_condition_id'] if initial_metadata else None,'initial_file_sha256':initial_metadata['initial_file_sha256'] if initial_metadata else None,'initial_geometry_sha256':initial_metadata['initial_geometry_sha256'] if initial_metadata else None};start=time.monotonic();pred_count=0;selected=0;tick=-1
    result.update(alignment_metadata)
    try:
        ready=reply()
        if ready.get('ready') is not True or ready.get('trial_id')!=trial_id:raise ValueError('Worker ready trial mismatch')
        checkpoint_manifest=json.loads((out/'checkpoint-manifest.json').read_text())
        model_sha256=checkpoint_manifest['model.safetensors']
        cal=json.loads(CALIBRATION.read_text())
        names=['shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll'];hi=np.array([(cal[n]['range_max']-cal[n]['range_min'])*180/4095 for n in names]+[100.]);lo=np.r_[-hi[:5],0.]
        grange=b.model.jnt_range[b.model.joint('gripper').id]
        def policy_state():
            if mapping is not None:return mapping.to_policy(b.data.qpos[b.qadr])
            v=b.data.qpos[b.qadr];return np.r_[np.rad2deg(v[:5]),100*(v[5]-grange[0])/(grange[1]-grange[0])]
        def sim_target(a):
            if mapping is not None:return mapping.to_sim(a)
            return np.r_[np.deg2rad(a[:5]),grange[0]+a[5]*(grange[1]-grange[0])/100]
        for _ in range(15):b.step_sim_targets(initial_q)
        initial=observe_task(b);evaluator=TapeTaskEvaluator();evaluator.update(initial)
        (out/'initial-snapshot.json').write_text(json.dumps(b.snapshot())+'\n')
        def predict(payload):
            nonlocal pred_count
            if pred_count>=15:raise RuntimeError('Prediction budget exhausted')
            index=pred_count
            path=out/f'input-{index:03d}.npz';np.savez_compressed(path,state=payload['state'],**payload['images'])
            input_hash=file_sha256(path)
            worker.stdin.write(json.dumps({'trial_id':trial_id,'input':str(path),'input_sha256':input_hash,
                                           'capture_sim_ns':payload['capture_sim_ns'],
                                           'observation_id':payload['observation_id'],'index':index})+'\n')
            worker.stdin.flush();answer=reply();pred_count+=1
            metadata=json.loads((out/f'prediction-{index:03d}.json').read_text())
            actions_path=verify_prediction_response(answer,metadata,trial_id=trial_id,
                observation_id=payload['observation_id'],capture_sim_ns=payload['capture_sim_ns'],
                prediction_index=index,input_sha256=input_hash,model_sha256=model_sha256,output_dir=out)
            return MovePotChunk(np.load(actions_path)['actions'].copy(),metadata['source'],metadata['prediction_host_start_ns'],metadata['prediction_host_end_ns'],metadata['forward_ms'],metadata['processing_ms'])
        schedule=SimChunkSchedule(q,predict,latency_ns='measured_host',observation_delay_frames=OBSERVATION_DELAY_FRAMES,clock_id=trial_id)
        last=policy_state();held=b.data.ctrl.copy();selected=0;now=0
        def validate(a,event):
            if np.any(a<lo) or np.any(a>hi):raise PolicyGateRejected('policy_target_outside_calibration_bounds')
            if np.any(np.abs(a-last)>np.r_[np.full(5,20.),25.]):raise PolicyGateRejected('development_step_limit')
            if now-event['source']['capture_sim_ns']>2_500_000_000:raise PolicyGateRejected('development_source_age_limit')
            b._validate_targets(sim_target(a))
        video=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1280x480','-r','10','-i','-','-an','-c:v','libx264','-crf','23','-pix_fmt','yuv420p',str(out/'dual-camera.mp4')],stdin=subprocess.PIPE)
        def record_frame(images):video.stdin.write(np.concatenate(list(images.values()),axis=1).tobytes())
        record_frame(b.render())
        for tick in range(MAX_CONTROL_TICKS):
            now=round(b.data.time*1e9);images=b.render();state=policy_state()
            if not np.isfinite(state).all() or np.any(state<lo) or np.any(state>hi):raise PolicyGateRejected('observed_state_outside_nominal_calibration')
            schedule.capture(now,{'observation_id':f'{trial_id}:frame:{tick}','capture_sim_ns':now,'state':state.copy(),'images':images})
            action=schedule.tick(now,validate)
            if action is not None:held=sim_target(action);last=action.copy();selected+=1
            physics=b.step_sim_targets(held);facts=observe_task(b);status=evaluator.update(facts)
            events.write(json.dumps({'trial_id':trial_id,'tick':tick,'new_target':action is not None,'policy_target':last.tolist(),'physics':physics,'facts':asdict(facts),'status':status})+'\n')
            if tick%3==2 or status!='running':record_frame(b.render())
            if status!='running':break
        result.update(execution_status='completed',status=evaluator.finish(),selected_actions=selected,control_ticks=tick+1,task_events=evaluator.events)
    except Exception as exc:
        result.update(execution_status='stopped' if isinstance(exc,PolicyGateRejected) else 'error',status='stopped_error_or_gate',error=repr(exc),traceback=traceback.format_exc(),selected_actions=selected,control_ticks=tick+1)
        if video:record_frame(b.render())
    finally:
        if schedule:schedule.reset();(out/'schedule.json').write_text(json.dumps(schedule.events,indent=2)+'\n')
        if video:video.stdin.close();result['video_exit']=video.wait(timeout=15)
        if worker.poll() is None:
            try:worker.stdin.write('{"stop":true}\n');worker.stdin.flush();result['worker_shutdown']=reply(30)
            except Exception as exc:result['shutdown_error']=repr(exc);worker.terminate()
        try:result['worker_exit']=worker.wait(timeout=15)
        except subprocess.TimeoutExpired:worker.kill();worker.wait();result['worker_killed']=True
        (out/'final-snapshot.json').write_text(json.dumps(b.snapshot())+'\n');b.close();q.close();events.close();worker_log.close()
        result.update(predictions=pred_count,wall_s=time.monotonic()-start)
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(name+': '+json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--scene',type=Path,default=SCENE)
    parser.add_argument('--initial-wrist-roll-deg',type=float,default=None)
    parser.add_argument('--initial-condition',type=Path,help='C-only versioned scene initial state; incompatible with wrist override')
    parser.add_argument('--preflight-only',action='store_true',help='validate C initial state and exit before model worker starts')
    parser.add_argument('--trial-prefix',default='C-development',help='C-only ID prefix for both model workers')
    parser.add_argument('--observation-delay-frames',type=int,choices=(0,1),default=0)
    parser.add_argument('--policy',choices=('act','smolvla'),help='Run a single policy; omitted keeps the two-policy development run')
    parser.add_argument('--policy-python',default=POLICY_PYTHON)
    parser.add_argument('--checkpoint',type=Path,help='Explicit local checkpoint for --policy')
    parser.add_argument('--calibration',type=Path,default=CALIBRATION)
    parser.add_argument('--device',default=DEVICE)
    parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--max-control-ticks',type=int,default=MAX_CONTROL_TICKS)
    parser.add_argument('--alignment-spec',type=Path,help='Measured and reviewed mapping/camera/scene specification')
    args=parser.parse_args()
    if not 1<=args.max_control_ticks<=600:parser.error('--max-control-ticks must be in 1..600')
    if args.alignment_spec and not args.initial_condition:parser.error('--alignment-spec requires --initial-condition')
    MAX_CONTROL_TICKS=args.max_control_ticks;ALIGNMENT_SPEC=args.alignment_spec.resolve() if args.alignment_spec else None
    if args.checkpoint and not args.policy:parser.error('--checkpoint requires --policy')
    if args.seed<0:parser.error('--seed must be nonnegative')
    POLICY_PYTHON=args.policy_python;CHECKPOINT=args.checkpoint.resolve() if args.checkpoint else None
    CALIBRATION=args.calibration.resolve();DEVICE=args.device;SEED=args.seed
    if args.initial_wrist_roll_deg is not None and not np.isfinite(args.initial_wrist_roll_deg):parser.error('initial wrist roll must be finite')
    if args.initial_condition is not None and args.initial_wrist_roll_deg is not None:parser.error('initial condition and wrist override are mutually exclusive')
    if args.preflight_only and args.initial_condition is None:parser.error('--preflight-only requires --initial-condition')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,64}',args.trial_prefix):parser.error('invalid trial prefix')
    OUT=args.output.resolve();SCENE=args.scene.resolve();INITIAL_WRIST_ROLL_DEG=args.initial_wrist_roll_deg or 0.;OBSERVATION_DELAY_FRAMES=args.observation_delay_frames
    INITIAL_CONDITION=args.initial_condition.resolve() if args.initial_condition else None
    TRIAL_PREFIX=args.trial_prefix
    if INITIAL_CONDITION is not None:
        probe=TapeSimBackend(SCENE,1/30)
        try:
            initial_metadata=apply_initial_condition(INITIAL_CONDITION,probe,expected_split='C')
            hold=probe.data.qpos[probe.qadr].copy()
            for _ in range(50):probe.step_sim_targets(hold)
            facts=observe_task(probe)
            judge=TapeTaskEvaluator()
            if judge.update(facts)!='running' or facts.forbidden_contact or any(c['distance_m']<-.002 for c in probe.contacts()):
                raise ValueError('C initial condition not task-ready after settling')
            preflight={'trial_prefix':TRIAL_PREFIX,'status':'passed_C_initial_preflight','initial_condition_id':initial_metadata['initial_condition_id'],
                       'initial_file_sha256':initial_metadata['initial_file_sha256'],
                       'initial_geometry_sha256':initial_metadata['initial_geometry_sha256'],
                       'scene_sha256':initial_metadata['scene_sha256'],
                       'settled_sim_ns':probe.state()['sim_ns'],'zone':facts.zone,
                       'supported_on_table':facts.supported_on_table,
                       'hardware_access':False,'model_worker_started':False,'formal_trial':False}
        finally:probe.close()
    if args.preflight_only:
        OUT.mkdir(parents=True,exist_ok=False)
        (OUT/'preflight.json').write_text(json.dumps(preflight,indent=2)+'\n')
        print(json.dumps(preflight))
        raise SystemExit(0)
    OUT.mkdir(parents=True,exist_ok=True)
    results=[run(name) for name in ([args.policy] if args.policy else ('act','smolvla'))]
    raise SystemExit(1 if any(r['execution_status']=='error' for r in results) else 0)
