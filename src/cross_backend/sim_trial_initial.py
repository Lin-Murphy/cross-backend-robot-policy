"""Versioned MuJoCo initial-condition input; identity checked before reset."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

IDS={'C':r'C\d{2}','V':r'V(?:0[1-9]|10)','T':r'P(?:0[1-9]|10)'}


def initial_geometry_sha256(record):
    """Identity of physical reset content, excluding split/ID/file formatting."""
    scene=record.get('scene_sha256')
    if not isinstance(scene,str) or len(scene)!=64 or any(c not in '0123456789abcdef' for c in scene):
        raise ValueError('Invalid initial scene SHA256')
    if record.get('joint_unit')!='sim_radian_unaligned':
        raise ValueError('Invalid initial joint unit')
    fields={'robot_q_rad':(6,),'tape_xy_m':(2,),'mat_xy_m':(2,),
            'tape_quaternion_wxyz':(4,)}
    values={}
    for name,shape in fields.items():
        value=np.asarray(record.get(name),dtype=float)
        if value.shape!=shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid initial geometry {name}')
        values[name]=value.copy()
    quat=values['tape_quaternion_wxyz']
    norm=float(np.linalg.norm(quat))
    if not np.isclose(norm,1.,atol=1e-8):
        raise ValueError('Initial geometry quaternion must be unit length')
    quat=quat/norm
    first=next((x for x in quat if abs(x)>1e-15),None)
    if first is None:raise ValueError('Invalid initial quaternion')
    if first<0:quat=-quat
    values['tape_quaternion_wxyz']=quat
    payload={'scene_sha256':scene,'joint_unit':'sim_radian_unaligned'}
    for name,value in values.items():
        value[np.abs(value)<1e-15]=0.
        payload[name]=value.tolist()
    data=json.dumps(payload,sort_keys=True,separators=(',', ':'),allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def load_initial_condition(path, backend, *, expected_split):
    source=Path(path)
    raw=source.read_bytes()
    record=json.loads(raw)
    if expected_split not in IDS or record.get('split')!=expected_split:
        raise ValueError('Initial condition split mismatch')
    if record.get('schema_version')!=1 or record.get('backend')!='mujoco':
        raise ValueError('Initial condition schema/backend mismatch')
    if not isinstance(record.get('initial_condition_id'),str) or not re.fullmatch(IDS[expected_split],record['initial_condition_id']):
        raise ValueError('Invalid initial condition ID for split')
    if record.get('scene_sha256')!=backend.scene_sha256:
        raise ValueError('Initial condition scene identity mismatch')
    if record.get('joint_unit')!='sim_radian_unaligned':
        raise ValueError('Initial condition joint unit mismatch')
    fields={'robot_q_rad':(6,),'tape_xy_m':(2,),'mat_xy_m':(2,),
            'tape_quaternion_wxyz':(4,)}
    arrays={}
    for name,shape in fields.items():
        value=np.asarray(record.get(name),dtype=float)
        if value.shape!=shape or not np.isfinite(value).all():
            raise ValueError(f'Invalid initial condition {name}')
        arrays[name]=value
    if not np.isclose(np.linalg.norm(arrays['tape_quaternion_wxyz']),1.,atol=1e-8):
        raise ValueError('Initial condition tape quaternion must be unit length')
    backend._validate_targets(arrays['robot_q_rad'])
    geometry_sha256=initial_geometry_sha256(record)
    return {'record':record,'file_sha256':hashlib.sha256(raw).hexdigest(),
            'initial_geometry_sha256':geometry_sha256,
            'reset_kwargs':{'robot_q':arrays['robot_q_rad'],'tape_xy':arrays['tape_xy_m'],
                            'mat_xy':arrays['mat_xy_m'],'tape_quaternion':arrays['tape_quaternion_wxyz']}}


def apply_initial_condition(path, backend, *, expected_split):
    prepared=load_initial_condition(path,backend,expected_split=expected_split)
    state=backend.reset(**prepared['reset_kwargs'])
    return {'initial_condition_id':prepared['record']['initial_condition_id'],
            'split':expected_split,'initial_file_sha256':prepared['file_sha256'],
            'initial_geometry_sha256':prepared['initial_geometry_sha256'],
            'scene_sha256':backend.scene_sha256,'reset_state':state}
