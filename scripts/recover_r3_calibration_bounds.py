"""Recover existing calibration envelopes without connecting motors or approving limits."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
NAMES=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
CURRENT=Path('/home/murphy/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower_arm.json')
EXPECTED='f911b3b1e57ed0d513d27eeb7bbd743086b09a03f9babd4900b3a734c0594434'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out):
    assert sha(CURRENT)==EXPECTED
    cal=json.loads(CURRENT.read_text());assert tuple(cal)==NAMES
    runtime=Path('/home/murphy/project/lerobot/src/lerobot/motors/motors_bus.py')
    tables=Path('/home/murphy/project/lerobot/src/lerobot/motors/feetech/tables.py')
    assert 'normalized_values[id_] = (val - mid) * 360 / max_res' in runtime.read_text()
    assert '"sts3215": 4096' in tables.read_text()
    previous=ROOT/'artifacts/r2-live-readonly-20260924-attempt01/summary.json'
    hardware=json.loads(previous.read_text())['hardware_calibration']
    for name in NAMES:
        assert all(cal[name][k]==hardware[name][k] for k in ('id','homing_offset','range_min','range_max'))
    bounds=[]
    for i,name in enumerate(NAMES):
        row=cal[name];lo,hi=row['range_min'],row['range_max']
        assert row['id']==i+1 and 0<=lo<hi<=4095 and row['drive_mode']==0
        half=(hi-lo)*180/4095
        bounds.append({'joint':name+'.pos','raw_min':lo,'raw_max':hi,
                       'unit':'deg' if i<5 else 'calibrated_0_100',
                       'calibration_lower':-half if i<5 else 0.,'calibration_upper':half if i<5 else 100.,
                       'workspace_safe_lower':None,'workspace_safe_upper':None,
                       'validated_max_step':None,'validated_max_speed_per_s':None})
    candidates=[ROOT/'so101_follower_arm.json',ROOT/'artifacts/r0-refresh-20260923/calibration_so_follower_so101_follower_arm.json',
                ROOT/'artifacts/safety/so101_follower_arm-after-recalibration.json',
                ROOT/'artifacts/safety/so101_follower_arm-after-gripper-recalibration.json']
    inventory=[]
    for path in candidates:
        value=json.loads(path.read_text());same=value==cal
        inventory.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path),'matches_current_calibration':same,
                          'different_joints':[n for n in NAMES if value.get(n)!=cal[n]],
                          'disposition':'matching_snapshot' if same else 'historical_other_calibration_do_not_substitute'})
    lower=np.array([r['calibration_lower'] for r in bounds]);upper=np.array([r['calibration_upper'] for r in bounds])
    checks=[]
    for model in ('act','smolvla'):
        for index in range(3):
            path=ROOT/f'artifacts/r2-captured-dual-policy-20260924-attempt01/{model}-{index:02d}-actions.npz'
            with np.load(path,allow_pickle=False) as data: actions=data['actions'].copy()
            assert actions.shape==(50,6) and np.isfinite(actions).all()
            bad=np.argwhere((actions<lower)|(actions>upper))
            checks.append({'model':model,'sample':index,'actions_sha256':sha(path),
                           'outside_calibration_values':[{'offset':int(i),'joint':NAMES[j]+'.pos','value':float(actions[i,j])} for i,j in bad],
                           'actions_min':actions.min(axis=0).tolist(),'actions_max':actions.max(axis=0).tolist(),
                           'scope':'static calibration-envelope diagnostic only; not gate acceptance'})
    history=Path('/home/murphy/.bash_history')
    matches=[]
    if history.exists():
        for number,line in enumerate(history.read_text(errors='replace').splitlines(),1):
            if 'lerobot-rollout' in line and 'max_relative_target=2' in line and 'move pot' in line:
                matches.append({'line':number,'setting':'robot.max_relative_target=2','task':'move pot',
                                'unit_context':'use_degrees=true','disposition':'historical clipping setting, not verified rate/step limit'})
    result={'status':'recovered_position_calibration_only','calibration_path':str(CURRENT),
            'calibration_sha256':EXPECTED,'prior_hardware_match_evidence':str(previous.relative_to(ROOT)),
            'prior_hardware_match_sha256':sha(previous),'fresh_hardware_read_performed':False,
            'bounds':bounds,'inventory':inventory,'saved_prediction_checks':checks,
            'historical_relative_target_settings':matches,
            'other_found_limits':[{'source':'scripts/live_act_dry_run.py:55','value':'speed_normalized_per_s=[2]*6',
                                   'reusable':False,'reason':'old normalized-unit diagnostic, not current degree-unit safety validation'},
                                  {'source':'artifacts/safety/live-act-20260916T235848811267Z.json',
                                   'value':'gripper max_delta_normalized=3, max_raw_delta=45',
                                   'reusable':False,'reason':'old calibration and gripper-only bounded trial; not six-joint speed limits'}],
            'conversion':{'body':'(raw - (range_min + range_max)/2) * 360 / 4095','gripper':'(raw-range_min)/(range_max-range_min)*100',
                          'runtime_sha256':sha(runtime),'resolution_table_sha256':sha(tables)},
            'unresolved':['workspace-safe range','validated maximum step','validated maximum speed','approved age/control-period limits'],
            'execution_authorized':False,'motor_writes':0}
    out.mkdir(parents=True,exist_ok=False)
    (out/'current-calibration.json').write_bytes(CURRENT.read_bytes())
    (out/'recovery.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'bounds':bounds,'saved_prediction_outside_counts':[len(x['outside_calibration_values']) for x in checks],
                      'historical_relative_target_settings':matches,'execution_authorized':False},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)
