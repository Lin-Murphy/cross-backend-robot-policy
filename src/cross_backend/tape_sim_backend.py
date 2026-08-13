"""Simulation-radian engineering backend. Physical policy mapping intentionally unavailable."""
import time
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
from fractions import Fraction
import mujoco
import numpy as np
NAMES=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')

class TapeSimBackend:
    def __init__(self,scene,control_period_s=.02):
        self.model=mujoco.MjModel.from_xml_path(str(scene));self.data=mujoco.MjData(self.model)
        if not np.isfinite(control_period_s) or control_period_s<self.model.opt.timestep:
            raise ValueError('Control period must be finite and at least one physics step')
        self.period=Fraction(str(control_period_s)).limit_denominator(10**9)
        self.physics_period=Fraction(str(float(self.model.opt.timestep))).limit_denominator(10**9)
        scene_path=Path(scene)
        digest=hashlib.sha256(scene_path.read_bytes())
        xml=ET.parse(scene_path).getroot()
        compiler=xml.find('compiler')
        meshdir=scene_path.parent/((compiler.get('meshdir','')) if compiler is not None else '')
        for mesh in sorted(xml.findall('./asset/mesh'),key=lambda e:e.get('file','')):
            if mesh.get('file'):
                digest.update(mesh.get('file').encode());digest.update(hashlib.sha256((meshdir/mesh.get('file')).read_bytes()).digest())
        self.scene_sha256=digest.hexdigest()
        self.renderer=None;self.index=0
        self.default_mat_pos=self.model.geom_pos[self.model.geom('green_mat').id].copy()
        self.joint_ids=[self.model.joint(n).id for n in NAMES]
        self.qadr=[self.model.jnt_qposadr[i] for i in self.joint_ids]
        self.actuator_ids=[self.model.actuator(n).id for n in NAMES]
        self.tape_id=self.model.body('tape').id
        self.tape_qadr=self.model.jnt_qposadr[self.model.joint('tape_free').id]
    def reset(self,tape_xy=None,*,robot_q=None,tape_quaternion=None,mat_xy=None):
        # Validate the complete request before mutating a running simulator.
        def finite(value,shape):
            a=np.asarray(value,dtype=float)
            if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Invalid reset vector')
            return a
        xy=None if tape_xy is None else finite(tape_xy,(2,))
        mq=None if mat_xy is None else finite(mat_xy,(2,))
        q=None if robot_q is None else finite(robot_q,(6,))
        if q is not None:self._validate_targets(q)
        quat=None if tape_quaternion is None else finite(tape_quaternion,(4,))
        if quat is not None and not np.isclose(np.linalg.norm(quat),1.,atol=1e-8):raise ValueError('Quaternion must be unit length')
        mujoco.mj_resetData(self.model,self.data);self.index=0
        mat=self.model.geom('green_mat').id
        self.model.geom_pos[mat]=self.default_mat_pos
        if mq is not None:self.model.geom_pos[mat,:2]=mq
        if xy is not None:self.data.qpos[self.tape_qadr:self.tape_qadr+2]=xy
        if quat is not None:self.data.qpos[self.tape_qadr+3:self.tape_qadr+7]=quat
        if q is not None:self.data.qpos[self.qadr]=q
        self.data.ctrl[self.actuator_ids]=self.data.qpos[self.qadr]
        mujoco.mj_forward(self.model,self.data)
        return self.state()
    def snapshot(self):
        signature=mujoco.mjtState.mjSTATE_INTEGRATION
        vector=np.empty(mujoco.mj_stateSize(self.model,signature))
        mujoco.mj_getState(self.model,self.data,vector,signature)
        return {'schema':1,'scene_sha256':self.scene_sha256,'mujoco_version':mujoco.__version__,
                'period':[self.period.numerator,self.period.denominator],
                'index':self.index,'integration_state':vector.tolist(),
                'mat_position':self.model.geom_pos[self.model.geom('green_mat').id].tolist()}
    def restore(self,snapshot):
        if snapshot['schema']!=1 or snapshot['scene_sha256']!=self.scene_sha256 or snapshot['mujoco_version']!=mujoco.__version__ or snapshot['period']!=[self.period.numerator,self.period.denominator]:
            raise ValueError('Incompatible snapshot identity/runtime/period')
        signature=mujoco.mjtState.mjSTATE_INTEGRATION
        vector=np.asarray(snapshot['integration_state'],dtype=float);mat=np.asarray(snapshot['mat_position'],dtype=float)
        if vector.shape!=(mujoco.mj_stateSize(self.model,signature),) or not np.isfinite(vector).all() or mat.shape!=(3,) or not np.isfinite(mat).all() or type(snapshot['index']) is not int or snapshot['index']<0:
            raise ValueError('Invalid snapshot state')
        ratio=snapshot['index']*self.period/self.physics_period
        expected_steps=(ratio.numerator+ratio.denominator-1)//ratio.denominator
        if abs(vector[0]-float(expected_steps*self.physics_period))>1e-8:
            raise ValueError('Snapshot clock/index mismatch')
        self.model.geom_pos[self.model.geom('green_mat').id]=mat
        mujoco.mj_setState(self.model,self.data,vector,signature);self.index=snapshot['index']
        mujoco.mj_forward(self.model,self.data)
    def state(self):
        return {'sim_ns':round(self.data.time*1e9),'host_monotonic_ns':time.monotonic_ns(),'joint_names':NAMES,'joint_unit':'sim_radian_unaligned','joint_position':self.data.qpos[self.qadr].tolist(),'tape_position':self.data.xpos[self.tape_id].tolist(),'contacts':self.contacts(),'qpos':self.data.qpos.tolist(),'qvel':self.data.qvel.tolist(),'ctrl':self.data.ctrl.tolist(),'scheduled_control_ns':round(float(self.index*self.period)*1e9),'control_lateness_ns':round((self.data.time-float(self.index*self.period))*1e9)}
    def contacts(self):
        return [{'geom1':mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,c.geom1),'geom2':mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,c.geom2),'distance_m':float(c.dist)} for c in self.data.contact]
    def _validate_targets(self,target):
        a=np.asarray(target,dtype=float)
        if a.shape!=(6,) or not np.isfinite(a).all():raise ValueError('Expected six finite simulation-radian targets')
        for value,j,act in zip(a,self.joint_ids,self.actuator_ids):
            lo=max(self.model.jnt_range[j,0],self.model.actuator_ctrlrange[act,0]);hi=min(self.model.jnt_range[j,1],self.model.actuator_ctrlrange[act,1])
            if not lo<=value<=hi:raise ValueError('Target outside joint/actuator range intersection')
        return a
    def step_sim_targets(self,target):
        a=self._validate_targets(target)
        self.data.ctrl[self.actuator_ids]=a
        ratio=(self.index+1)*self.period/self.physics_period
        required_steps=(ratio.numerator+ratio.denominator-1)//ratio.denominator
        current_steps=round(self.data.time/float(self.physics_period))
        for _ in range(required_steps-current_steps):
            mujoco.mj_step(self.model,self.data)
            if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():raise RuntimeError('Nonfinite simulation state')
        # mj_step position-dependent fields refer to the preceding integration state.
        mujoco.mj_forward(self.model,self.data)
        self.index+=1
        return self.state()
    def step_policy_action(self,action):
        raise RuntimeError('Policy dispatch blocked: joint zero/sign, gripper and camera alignment unverified')
    def render(self):
        if self.renderer is None:self.renderer=mujoco.Renderer(self.model,height=480,width=640)
        images={}
        option=mujoco.MjvOption();option.geomgroup[5]=1
        for name in ('follower','camera2'):
            self.renderer.update_scene(self.data,camera=name,scene_option=option)
            images['observation.images.'+name]=self.renderer.render().copy()
        return images
    def close(self):
        if self.renderer is not None:self.renderer.close();self.renderer=None
