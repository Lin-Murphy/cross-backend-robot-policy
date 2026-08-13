"""MuJoCo evaluator-only geometry/contact extraction. Never passed to policies."""
import numpy as np
import mujoco
from .tape_task import TapeFacts

def observe_task(backend,outer_radius=.045,half_height=.025,boundary_margin=.001):
    m,d=backend.model,backend.data;tid=backend.tape_id
    center=d.xpos[tid];axis=d.xmat[tid].reshape(3,3)[:,2]
    object_geoms=np.flatnonzero(m.geom_bodyid==tid)
    if len(object_geoms)==1 and m.geom_type[object_geoms[0]]==mujoco.mjtGeom.mjGEOM_BOX:
        cube=object_geoms[0]
        extent=np.abs(d.geom_xmat[cube].reshape(3,3))@m.geom_size[cube]
    else:
        # Legacy hollow-roll geometry: conservative radial footprint.
        extent=outer_radius*np.sqrt(np.maximum(0,1-axis*axis))+half_height*np.abs(axis)
    mat=m.geom('green_mat').id;mat_center=d.geom_xpos[mat];mat_size=m.geom_size[mat]
    # Current generated mat is axis-aligned. Reject unsupported rotated target frames.
    if not np.allclose(d.geom_xmat[mat].reshape(3,3),np.eye(3)):raise ValueError('Rotated mat requires explicit frame transform')
    gap=np.abs(center[:2]-mat_center[:2])
    inside=bool(np.all(gap+extent[:2]<mat_size[:2]-boundary_margin))
    outside=bool(np.any(gap-extent[:2]>mat_size[:2]+boundary_margin))
    zone='inside' if inside else 'outside' if outside else 'boundary'
    support_mat=False;support_table=False;fixed=False;moving=False;forbidden=False;deep_object_support_penetration=False
    jaw=m.body('moving_jaw_so101_v1').id;grip=m.body('gripper').id
    base=m.body('base').id
    def kind(g):
        body=int(m.geom_bodyid[g]);name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g)
        if body==tid:return 'tape'
        if name in ('table','green_mat'):return name
        if body==jaw:return 'moving_jaw'
        if body==grip and ((name or '').startswith('fixed_jaw') or m.geom_group[g]==4):return 'fixed_jaw'
        if body==base:return 'base'
        return 'robot' if body else 'world'
    for c in d.contact:
        if c.dist>0:continue
        a,b=kind(c.geom1),kind(c.geom2);pair={a,b}
        if pair=={'tape','green_mat'}:
            support_mat=True
            deep_object_support_penetration |= c.dist < -.002
        elif pair=={'tape','table'}:
            support_table=True
            deep_object_support_penetration |= c.dist < -.002
        elif pair=={'tape','fixed_jaw'}:fixed=True
        elif pair=={'tape','moving_jaw'}:moving=True
        elif pair=={'base','table'}:pass  # intentional mount contact
        elif a in ('robot','fixed_jaw','moving_jaw','base') or b in ('robot','fixed_jaw','moving_jaw','base'):forbidden=True
    velocity=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,tid,velocity,0)
    support_height=mat_center[2]+mat_size[2] if zone=='inside' else 0.
    return TapeFacts(round(d.time*1e9),zone,float(center[2]-extent[2]-support_height),float(np.linalg.norm(velocity[3:])),support_mat,support_table,fixed and moving,fixed or moving,forbidden,deep_object_support_penetration or not np.isfinite(d.qpos).all() or any(w.number for w in d.warning))
