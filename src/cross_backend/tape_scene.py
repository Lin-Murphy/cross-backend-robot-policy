"""MuJoCo development scene for a hollow tape roll; no hardware or policy imports."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET


def build_scene(upstream: Path, output: Path, spec: dict):
    root=ET.parse(upstream/'so101.xml').getroot()
    root.set('model','so101_tape_development')
    root.find('compiler').set('meshdir',str((upstream/'assets').resolve()))
    defaults=spec['development_assumptions']
    root.find('option').set('timestep',str(defaults['physics_timestep_s']))
    visual=root.find('visual')
    ET.SubElement(visual,'global',offwidth='640',offheight='480')
    asset=root.find('asset');world=root.find('worldbody')
    world.find("body[@name='base']").set('pos',' '.join(map(str,defaults.get('robot_base_position_m',[0,0,0]))))
    yaw=defaults.get('robot_base_yaw_rad',0.)
    world.find("body[@name='base']").set('quat',f'{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}')
    ET.SubElement(world,'light',pos='0 -0.3 1.5',dir='0 0 -1',diffuse='.8 .8 .8')
    ET.SubElement(world,'geom',name='table',type='plane',size='1 1 .02',rgba='.55 .48 .39 1',friction='.5 .005 .0001')
    mx,my=defaults['mat_center_xy_m'];th=defaults['mat_thickness_m'];sx,sy=spec['mat']['size_m']
    ET.SubElement(world,'geom',name='green_mat',type='box',size=f'{sx/2} {sy/2} {th/2}',pos=f'{mx} {my} {th/2}',rgba='.08 .5 .17 1',friction='.5 .005 .0001')
    # Camera extrinsics are development assumptions, not recovered physical calibration.
    ET.SubElement(world,'camera',name='follower',pos='-.65 -.65 .65',xyaxes='.7071 -.7071 0 .4082 .4082 .8165',fovy='48')
    wrist=world.find(".//camera[@name='wrist_cam']");wrist.set('name','camera2')
    pot=spec['pot'];ro=pot['outer_diameter_m']/2;ri=pot['inner_diameter_m']/2;h=pot['height_m'];mass=pot['mass_kg']
    if not 0<ri<ro or h<=0 or mass<=0:raise ValueError('Invalid tape geometry/mass')
    n=defaults['ring_segments']
    if type(n) is not int or n<16:raise ValueError('Too few annular sectors')
    px,py=defaults['tape_center_xy_m']
    body=ET.SubElement(world,'body',name='tape',pos=f'{px} {py} {h/2+.003}')
    ET.SubElement(body,'freejoint',name='tape_free')
    iz=mass*(ro*ro+ri*ri)/2;ix=mass*(3*(ro*ro+ri*ri)+h*h)/12
    ET.SubElement(body,'inertial',pos='0 0 0',mass=str(mass),diaginertia=f'{ix} {ix} {iz}')
    for i in range(n):
        a=2*math.pi*i/n;b=2*math.pi*(i+1)/n
        vertices=[(radius*math.cos(angle),radius*math.sin(angle),z) for z in (-h/2,h/2) for radius,angle in [(ri,a),(ro,a),(ro,b),(ri,b)]]
        name=f'tape_sector_{i:02}'
        ET.SubElement(asset,'mesh',name=name,vertex=' '.join(str(v) for xyz in vertices for v in xyz))
        ET.SubElement(body,'geom',name=name,type='mesh',mesh=name,mass='0',group='5',rgba='.88 .91 .83 .58',friction=' '.join(map(str,defaults['tape_friction'])),condim='3')
    output.parent.mkdir(parents=True,exist_ok=True)
    ET.indent(root);ET.ElementTree(root).write(output,encoding='unicode')
    return {'segments':n,'inner_diameter_m':2*ri,'outer_diameter_m':2*ro,'height_m':h,'mass_kg':mass,'inner_chord_max_intrusion_m':ri*(1-math.cos(math.pi/n)),'inertia_kg_m2':[ix,ix,iz],'physical_alignment_validated':False}
