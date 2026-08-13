"""Offline proposal only: no motor/camera imports, no dispatch implementation."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main(capture, output):
    metadata=json.loads((capture/'summary.json').read_text())
    assert metadata['status']=='passed' and metadata['dispatches']==0
    assert metadata['motor_writes_enabled'] is False
    assert metadata['calibration_differences']==[]
    assert metadata['torque_before']==metadata['torque_after']
    assert len(metadata['samples'])==3
    assert all(not s['raw_position_violations'] for s in metadata['samples'])
    states=[s['raw_positions'] for s in metadata['samples']]
    assert all(s==states[0] for s in states)
    cal=metadata['hardware_calibration'];start=states[0]['gripper']
    low,high=cal['gripper']['range_min'],cal['gripper']['range_max']
    # An explicit small *proposed* experiment envelope, NOT an established limit.
    targets=[start+i for i in (1,2,3,4,3,2,1,0)]
    assert all(low<target<high for target in targets)
    assert all(abs(b-a)==1 for a,b in zip([start]+targets,targets))
    assert max(abs(target-start) for target in targets)==4 and targets[-1]==start
    unit_per_tick=100/(high-low)
    bounds={}
    for name,state in states[0].items():
        a,b=cal[name]['range_min'],cal[name]['range_max']
        factor=100/(b-a) if name=='gripper' else 360/4095
        bounds[name]={'raw':state,'raw_min':a,'raw_max':b,
                      'distance_to_lower_units':(state-a)*factor,'distance_to_upper_units':(b-state)*factor,
                      'unit':'calibrated_0_100' if name=='gripper' else 'deg'}
    result={'status':'proposal_only_not_authorized','dispatch_implemented':False,'hardware_dispatches':0,
            'source_capture':str(capture.resolve()),'source_files_sha256':{p.name:sha(p) for p in sorted(capture.iterdir()) if p.is_file()},
            'calibration_sha256':metadata['calibration_sha256'],'read_only_checks':'passed',
            'operator_reports':{'present':True,'power_cut_method':'manual unplug, reachability still to confirm',
                                'no_collision_or_cable_entanglement':True,'camera_and_table_unchanged':True,
                                'additional_dynamic_limits_record':'interpreted absent from user answer 无'},
            'current_calibration_margins':bounds,
            'proposal':{'joint':'gripper','motor_id':6,'unit':'raw_encoder_ticks',
                        'reference_start_raw':start,'proposed_targets_raw':targets,
                        'max_goal_offset_ticks':4,'max_goal_offset_calibrated_units':4*unit_per_tick,
                        'max_goal_step_ticks':1,'minimum_goal_interval_ms':100,
                        'goal_ramp_rate_ticks_per_s_upper':10,'goal_ramp_rate_calibrated_units_per_s_upper':10*unit_per_tick,
                        'note':'Command-ramp limits only; actual servo speed is not established.',
                        'endpoint_hold_ms':500,'total_timeout_s':5,
                        'proposed_early_stop':{'gripper_displacement_from_fresh_start_over_ticks':8,
                                               'non_gripper_drift_from_fresh_start_over_ticks':4,
                                               'state_read_duration_over_ms':50,
                                               'any_communication_error_or_range_violation':True,
                                               'operator_intervention':True},
                        'end_condition':'Return to fresh start only during normal progression; readback within 2 ticks, otherwise failure.',
                        'on_fault':'Stop further goal writes, retain trace, no blind return, no automatic torque toggle; operator cuts power if needed.',
                        'torque_policy':'Do not enable, disable or restore torque; require fresh readbacks all 1 or refuse.',
                        'fresh_preflight':'Recheck same calibration, limits, raw positions and operator readiness immediately before any action; refuse changed posture.',
                        'other_joint_goal_writes':False,'policy_model_used':False},
            'required_before_execution':['operator confirms gripper empty, hands clear and power plug reachable',
                                         'explicit authorization for this specific gripper-only proposal',
                                         'implement bounded allowlist writer and verify with fake bus before hardware use',
                                         'fresh preflight passes; saved capture never substitutes for fresh state'],
            'not_established':['six-joint speed limits','workspace-safe limits','task performance','policy dispatch readiness']}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'raw_targets':targets,'max_calibrated_excursion':4*unit_per_tick,
                      'wrist_upper_margin_deg':bounds['wrist_flex']['distance_to_upper_units'],'hardware_dispatches':0},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise SystemExit('Refusing to overwrite proposal')
    main(a.capture,a.output)
