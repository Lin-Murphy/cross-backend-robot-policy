"""Real renderer / synthetic coordinate mapping integration, with deployment refused."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import sys,json
import numpy as np
r=Path(__file__).resolve().parents[1];sys.path.insert(0,str(r/'src'))
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.sim_policy_mapping import CoordinateMapping,mapping_from_spec,simulation_observation
out=r/'artifacts/s1-coordinate-interface-20260925'
spec=json.loads((r/'configs/s1-tape-scene-development.json').read_text())
try:mapping_from_spec(spec)
except ValueError as e:refusal=str(e)
else:raise AssertionError('Unverified physical mapping accepted')
b=TapeSimBackend(r/'artifacts/s1-base-spacing-20260925/scene.xml');b.reset()
try:
    m=CoordinateMapping((np.pi/180,)*5+(.01,), (0,)*5+(-.2,),(-180,)*5+(0,),(180,)*5+(100,),(-3.2,)*5+(-.2,),(3.2,)*5+(.8,),'synthetic-only unit fixture, not robot calibration','synthetic_test')
    observation,metadata=simulation_observation(b,m,'s1-synthetic-render-0001')
    np.savez_compressed(out/'synthetic-observation.npz',state=observation.state,**observation.images_rgb)
    (out/'integration.json').write_text(json.dumps({'status':'passed_interface_only','metadata':metadata,'frame_ids':observation.frame_ids,'image_shapes':{k:list(v.shape) for k,v in observation.images_rgb.items()},'state':observation.state.tolist(),'actual_spec_refusal':refusal,'hardware_access':False,'policy_inference':False},indent=2)+'\n')
    print('PASS: real named RGB + synthetic state conversion; real spec correctly refuses unverified mapping.')
finally:b.close()
