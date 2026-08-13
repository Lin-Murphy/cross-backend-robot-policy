"""Explicit affine policy/simulation coordinate mapping; no inferred calibration."""
from dataclasses import dataclass
import numpy as np
from .move_pot_policy import JOINTS, UNITS, CAMERAS, MovePotObservation

@dataclass(frozen=True)
class CoordinateMapping:
    scale: tuple[float, ...]
    offset: tuple[float, ...]
    policy_lower: tuple[float, ...]
    policy_upper: tuple[float, ...]
    sim_lower: tuple[float, ...]
    sim_upper: tuple[float, ...]
    provenance: str
    validation: str
    def __post_init__(self):
        for name in ('scale','offset','policy_lower','policy_upper','sim_lower','sim_upper'):
            a=np.asarray(getattr(self,name),dtype=float)
            if a.shape!=(6,) or not np.isfinite(a).all():raise ValueError('Missing finite six-joint mapping: '+name)
        if np.any(np.asarray(self.scale)==0):raise ValueError('Mapping is not invertible')
        if np.any(np.asarray(self.policy_lower)>=self.policy_upper) or np.any(np.asarray(self.sim_lower)>=self.sim_upper):raise ValueError('Invalid bounds')
        if not self.provenance or self.validation not in ('synthetic_test','physical_alignment_verified'):raise ValueError('Unverified mapping provenance')
    @staticmethod
    def _check(value,lower,upper):
        a=np.asarray(value,dtype=float)
        if a.shape!=(6,) or not np.isfinite(a).all():raise ValueError('Expected finite six-joint vector')
        if np.any(a<lower) or np.any(a>upper):raise ValueError('Outside mapping bounds; no clipping')
        return a
    def to_sim(self,action,*,joint_names=JOINTS,joint_units=UNITS):
        if joint_names!=JOINTS or joint_units!=UNITS:raise ValueError('Joint names/units mismatch')
        a=self._check(action,self.policy_lower,self.policy_upper)
        return self._check(a*np.asarray(self.scale)+self.offset,self.sim_lower,self.sim_upper)
    def to_policy(self,state):
        q=self._check(state,self.sim_lower,self.sim_upper)
        return self._check((q-np.asarray(self.offset))/self.scale,self.policy_lower,self.policy_upper)


def mapping_from_spec(spec):
    profile=spec.get('verified_coordinate_mapping')
    if profile is None:raise ValueError('No verified coordinate mapping; do not infer zero/sign/gripper endpoints')
    mapping=CoordinateMapping(**profile)
    if mapping.validation!='physical_alignment_verified':raise ValueError('Synthetic mapping cannot enable policy integration')
    return mapping


def simulation_observation(backend,mapping,episode_id):
    """Named RGB/state packet with a separate explicit simulation clock domain."""
    if not episode_id:raise ValueError('Episode identity required')
    stamp=round(backend.data.time*1e9)
    state=mapping.to_policy(backend.data.qpos[backend.qadr])
    images=backend.render()
    identity=f'{episode_id}:{backend.index}:{stamp}'
    obs=MovePotObservation(images_rgb=images,state=state,observation_id=identity,
        frame_ids={name:identity+':'+name for name in CAMERAS},
        camera_capture_ns={name:stamp for name in CAMERAS},state_capture_ns=stamp)
    obs.validate()
    return obs,{'clock_domain':'simulation','clock_id':episode_id,'sim_ns':stamp,
                'mapping_provenance':mapping.provenance,'mapping_validation':mapping.validation,
                'physical_alignment_verified':mapping.validation=='physical_alignment_verified',
                'camera_alignment_verified':False,'policy_dispatch_authorized':False}
