"""Offline S0 inventory. Reads pinned source metadata; never opens hardware."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def inventory(out):
    upstream = out / 'upstream'
    model = ET.parse(upstream / 'so101.xml').getroot()
    calibration_path = ROOT / 'artifacts/r3-real-clock-timing-20260924/calibration-recovery/current-calibration.json'
    calibration = json.loads(calibration_path.read_text())
    joints = {j.attrib['name']: j.attrib for j in model.findall('.//worldbody//joint')}
    actuators = {a.attrib['joint']: a.attrib for a in model.findall('./actuator/position')}
    assert tuple(joints) == NAMES and set(actuators) == set(NAMES)
    rows = []
    for name in NAMES:
        c = calibration[name]
        half = (c['range_max']-c['range_min'])*180/4095
        joint_range = list(map(float, joints[name]['range'].split()))
        ctrl_range = list(map(float, actuators[name]['ctrlrange'].split()))
        rows.append({'policy_name': name+'.pos', 'mjcf_joint': name,
                     'policy_unit': 'calibrated_0_100' if name=='gripper' else 'deg',
                     'policy_calibration_bounds': [0,100] if name=='gripper' else [-half,half],
                     'raw_calibration': c, 'mjcf_range_rad': joint_range,
                     'mjcf_range_deg': [math.degrees(v) for v in joint_range],
                     'mjcf_ctrlrange_rad': ctrl_range,
                     'coordinate_sign': None, 'coordinate_zero_offset_rad': None,
                     'gripper_endpoint_mapping': None if name=='gripper' else 'not_applicable',
                     'mapping_status': 'names_match_physical_coordinates_unverified'})
    listings = json.loads((upstream/'assets-listing.json').read_text())
    referenced = {m.attrib['file'] for m in model.findall('./asset/mesh')}
    declared = {f['name']:f for f in listings}
    assert referenced <= set(declared)
    roots = ['/home/murphy/project', '/home/murphy/Documents']
    command = ['rg','--files','--hidden',*roots,'-g','*.urdf','-g','*.mjcf','-g','*.stl','-g','*.STL','-g','!node_modules','-g','!site-packages','-g','!.git']
    scan = subprocess.run(command,capture_output=True,text=True)
    (out/'local-asset-search.json').write_text(json.dumps({'command':command,'exit_code':scan.returncode,'files':scan.stdout.splitlines(),'stderr':scan.stderr,'scope_note':'Not an exhaustive whole-disk search; fetched upstream XML is separate.'},indent=2)+'\n')
    sources = [ROOT/'PROJECT_PLAN.md', ROOT/'docs/simulation-validation-plan.md', calibration_path,
               ROOT/'src/cross_backend/move_pot_policy.py', ROOT/'src/cross_backend/smolvla_move_pot.py',
               ROOT/'src/cross_backend/degree_execution.py', ROOT/'artifacts/r2-hardware-mapping-recovery-20260924/command-device-parameters.json']
    records = ROOT/'artifacts/move-pot-evaluation-v1-records'
    info = {'stage':'S0', 'status':'inventory_complete_scene_alignment_open',
            'hardware_access':False, 'model_inference':False, 'upstream':json.loads((upstream/'commit.json').read_text()),
            'license':'Apache-2.0 (retained upstream LICENSE)', 'mesh_count_referenced':len(referenced),
            'mesh_bytes_referenced':sum(declared[n]['size'] for n in referenced),
            'mesh_downloaded':False, 'joint_mapping':rows,
            'upstream_option':model.find('option').attrib,
            'upstream_actuator_defaults':model.find("./default/default[@class='sts3215']/position").attrib,
            'upstream_cameras':[c.attrib for c in model.findall('.//camera')],
            'sources':{str(p.relative_to(ROOT)):sha(p) for p in sources},
            'real_trial_records_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sorted(records.glob('*.json'))},
            'limitations':['No physical joint zero/direction alignment verified','No measured gripper opening map','No camera intrinsic/extrinsic calibration found in scoped project search','No measured pot/table/mat geometry or dynamics found in scoped project search','Asset defaults are not measured servo dynamics']}
    (out/'inventory.json').write_text(json.dumps(info,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'status':info['status'],'joints':len(rows),'mesh_count':len(referenced),'mesh_bytes':info['mesh_bytes_referenced']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    inventory(p.parse_args().output.resolve())
