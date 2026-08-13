"""One nominal development observation, never a calibrated physical observation."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
OUT=ROOT/'artifacts/s1-model-input-diagnostic-20260925'
b=TapeSimBackend(ROOT/'artifacts/s1-base-spacing-20260925/scene.xml');b.reset()
try:
    q=b.data.qpos[b.qadr].copy();grange=b.model.jnt_range[b.model.joint('gripper').id].copy()
    state=np.r_[np.rad2deg(q[:5]),100*(q[5]-grange[0])/(grange[1]-grange[0])]
    images=b.render();np.savez_compressed(OUT/'nominal-observation.npz',state=state,**images)
    metadata={'purpose':'one saved observation through each existing policy; no action selection/dispatch','body_mapping':'nominal identity sign/zero; q_rad -> deg; NOT physically verified','gripper_mapping':'nominal asset hinge endpoints mapped to 0..100; NOT measured opening calibration','gripper_endpoints_rad':grange.tolist(),'state':state.tolist(),'sim_q':q.tolist(),'clock_domain':'simulation','capture_sim_ns':0,'camera_alignment_verified':False,'physical_coordinate_alignment_verified':False,'action_execution_allowed':False,'inference_budget':{'act_chunks':1,'smolvla_chunks':1,'training':0,'downloads':0}}
    (OUT/'input-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
finally:b.close()
