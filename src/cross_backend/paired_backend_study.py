"""Pair measured simulation/hardware episodes without turning unknowns into failures.

This module checks evidence integrity and declared measurement tolerances. It does
not certify calibration measurements or authorize hardware execution.
"""
import json
import math
from pathlib import Path

from .evaluation import digest
from .study_comparison import paired_statistics

ALIGNMENT_COMPONENTS = ('joint_and_gripper_mapping', 'camera_intrinsics', 'camera_extrinsics',
                        'table_frame', 'object_and_mat_geometry')
RESIDUALS = ('joint_max_error_deg', 'gripper_error_units', 'object_position_error_m',
             'mat_position_error_m', 'camera_reprojection_error_px')


def read_reference(reference, root, *, as_json=True):
    if not isinstance(reference, dict) or set(reference) != {'path', 'sha256'}:
        raise ValueError('Evidence requires a path and SHA256')
    path = Path(reference['path'])
    if not path.is_absolute():
        path = root / path
    if not path.is_file() or path.stat().st_size == 0 or digest(path) != reference['sha256']:
        raise ValueError(f'Evidence missing, empty or changed: {path}')
    return json.loads(path.read_text()) if as_json else path


def validate_alignment(protocol, root):
    issues = []
    if protocol.get('status') != 'frozen' or not protocol.get('frozen_utc'):
        issues.append('protocol_not_frozen')
    try:
        alignment = read_reference(protocol.get('alignment'), root)
        if not alignment.get('reviewer') or not alignment.get('reviewed_utc'):
            raise ValueError('Alignment reviewer and review time required')
        for component in ALIGNMENT_COMPONENTS:
            measure = alignment.get(component, {})
            if measure.get('status') != 'measured_and_reviewed' or not measure.get('evidence'):
                raise ValueError(f'Missing reviewed alignment: {component}')
            for reference in measure['evidence']:
                read_reference(reference, root, as_json=False)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        issues.append('alignment_unverified: ' + str(exc))
    limits = protocol.get('pair_tolerances', {})
    for name in RESIDUALS:
        value = limits.get(name)
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            issues.append('pair_tolerance_unset:' + name)
    if not protocol.get('success_rule'):
        issues.append('success_rule_missing')
    return issues


def inspect_pair(pair, protocol, root):
    """Return verified outcomes only after evidence, identity and timing checks."""
    case, policy = pair['case_id'], pair['policy']
    binding = read_reference(pair.get('binding'), root)
    if (binding.get('case_id'), binding.get('policy')) != (case, policy):
        raise ValueError('Binding case/policy mismatch')
    if not binding.get('reviewer') or not binding.get('reviewed_utc'):
        raise ValueError('Pair measurement review missing')
    if binding.get('success_rule') != protocol['success_rule']:
        raise ValueError('Task success criteria differ')
    for name in RESIDUALS:
        value = binding.get('measured_residuals', {}).get(name)
        if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
            raise ValueError('Missing measured pair residual: ' + name)
        if value > protocol['pair_tolerances'][name]:
            raise ValueError('Pair alignment tolerance exceeded: ' + name)
    if not binding.get('measurement_evidence'):
        raise ValueError('Pair measurement evidence missing')
    for reference in binding['measurement_evidence']:
        read_reference(reference, root, as_json=False)
    sim, real = pair['simulation'], pair['real']
    sim_summary = read_reference(sim['summary'], root)
    sim_directory=read_reference(sim['summary'],root,as_json=False).parent
    for key in ('budget','checkpoint_manifest','events','video'):
        if read_reference(sim[key],root,as_json=False).parent!=sim_directory:
            raise ValueError('Simulation evidence belongs to a different run')
    initial = read_reference(sim['initial'], root)
    sim_budget = read_reference(sim['budget'], root)
    sim_checkpoint = read_reference(sim['checkpoint_manifest'], root)
    real_summary = read_reference(real['summary'], root)
    assessment = read_reference(real['assessment'], root)
    audit = read_reference(real['audit'], root)
    profile = read_reference(real['profile'], root)
    read_reference(real['first_observation'], root, as_json=False)
    read_reference(real['events'], root, as_json=False)
    checkpoint_path=read_reference(real['checkpoint'], root, as_json=False)
    real_directory=read_reference(real['summary'],root,as_json=False).parent
    for key in ('assessment','audit','profile','first_observation','events'):
        if read_reference(real[key],root,as_json=False).parent!=real_directory:
            raise ValueError('Real evidence belongs to a different run')
    if assessment.get('trial_id')!=real_directory.name:
        raise ValueError('Task assessment refers to another real trial')
    if Path(real_summary.get('model_path','')).resolve()!=checkpoint_path.parent.resolve() or Path(profile.get('model_path','')).resolve()!=checkpoint_path.parent.resolve():
        raise ValueError('Real runner checkpoint path differs from supplied checkpoint')
    for name in ('follower_video', 'camera2_video'):
        read_reference(real[name], root, as_json=False)
    read_reference(sim['video'], root, as_json=False)
    read_reference(sim['events'], root, as_json=False)
    model_sha = protocol['checkpoint_sha256'][policy]
    if sim_checkpoint.get('model.safetensors') != model_sha or real['checkpoint']['sha256'] != model_sha:
        raise ValueError('Different checkpoint across backends or frozen study')
    if real_summary.get('policy') != policy or real_summary.get('backend') != 'so101' or assessment.get('policy') != policy:
        raise ValueError('Real policy/backend identity mismatch')
    if not sim_summary.get('trial_id', '').endswith('-' + policy):
        raise ValueError('Simulation policy identity mismatch')
    if sim_summary.get('alignment_spec_sha256')!=protocol['alignment']['sha256']:
        raise ValueError('Simulation used a different alignment specification')
    alignment=read_reference(protocol['alignment'],root)
    if alignment.get('scene_sha256')!=initial.get('scene_sha256') or alignment.get('robot_calibration_sha256')!=real_summary.get('calibration_sha256'):
        raise ValueError('Alignment does not match the scene or real robot calibration')
    if sim_summary.get('physical_alignment_verified') is not True:
        raise ValueError('Nominal simulation cannot establish paired physical performance')
    if binding.get('simulation_initial_sha256') != sim['initial']['sha256'] or sim_summary.get('initial_file_sha256') != sim['initial']['sha256']:
        raise ValueError('Simulation initial identity mismatch')
    if initial.get('scene_sha256') != binding.get('scene_sha256'):
        raise ValueError('Scene identity mismatch')
    if binding.get('real_first_observation_sha256') != real['first_observation']['sha256']:
        raise ValueError('Real initial observation identity mismatch')
    if audit.get('status') != 'passed' or audit.get('summary_sha256') != real['summary']['sha256'] or audit.get('events_sha256') != real['events']['sha256']:
        raise ValueError('Raw hardware audit mismatch')
    if audit.get('paired_observations_actions_packets') != real_summary.get('raw_goal_packets_transmitted'):
        raise ValueError('Audited dispatch count mismatch')
    if real_summary.get('profile_sha256') != real['profile']['sha256']:
        raise ValueError('Real execution profile mismatch')
    hz = protocol['timing']['control_hz']
    seconds = protocol['timing']['policy_seconds']
    if sim_budget.get('control_hz') != hz or profile.get('fps') != hz or sim_budget.get('max_control_ticks') != hz * seconds or profile.get('max_episode_seconds') != seconds:
        raise ValueError('Control frequency or policy time budget mismatch')
    if binding.get('timing_review') != 'matched_with_declared_backend_differences':
        raise ValueError('Scheduling/observation-age differences have not been reviewed')
    status = sim_summary['status']
    sim_outcome = True if status == 'success' else False if status in {'timeout', 'incomplete', 'drop', 'collision', 'placement_outside'} else None
    real_outcome = assessment.get('task_success')
    if real_outcome is not None and type(real_outcome) is not bool:
        raise ValueError('Real assessment must retain boolean/unknown task outcome')
    if real_outcome is not None and not assessment.get('task_evidence'):
        raise ValueError('Operator outcome evidence missing')
    return {'case_id': case, 'policy': policy, 'simulation_outcome': sim_outcome,
            'real_outcome': real_outcome, 'simulation_status': status,
            'real_status': real_summary['status'], 'binding_sha256': pair['binding']['sha256']}


def analyze_sim_real(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    protocol = read_reference(manifest['protocol'], path.parent)
    if protocol.get('schema_version') != 1 or protocol.get('task') != 'move_pot':
        raise ValueError('Unsupported paired study protocol')
    expected = [(case, policy) for case in protocol['cases'] for policy in protocol['policies']]
    pairs = manifest['pairs']
    keys = [(p['case_id'], p['policy']) for p in pairs]
    if len(set(expected)) != len(expected) or len(keys) != len(expected) or set(keys) != set(expected):
        raise ValueError('Keep all unique planned case/policy pairs, including unstarted pairs')
    global_issues = validate_alignment(protocol, path.parent)
    rows, accepted = [], []
    for pair in pairs:
        reasons = list(global_issues)
        if not pair.get('simulation'):
            reasons.append('simulation_not_recorded')
        if not pair.get('real'):
            reasons.append('real_not_recorded')
        if not pair.get('binding'):
            reasons.append('initial_correspondence_not_measured')
        row = {'case_id': pair['case_id'], 'policy': pair['policy'], 'eligible': False, 'reasons': reasons}
        if not reasons:
            try:
                values = inspect_pair(pair, protocol, path.parent)
                row.update(values)
                if values['simulation_outcome'] is None or values['real_outcome'] is None:
                    row['reasons'].append('task_outcome_unknown')
                else:
                    row['eligible'] = True
                    accepted.append(values)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                row['reasons'].append(str(exc))
        rows.append(row)
    groups = {}
    for policy in protocol['policies']:
        selected = [r for r in accepted if r['policy'] == policy]
        stats = paired_statistics([(r['simulation_outcome'], r['real_outcome']) for r in selected])
        stats['agreement_rate'] = sum(r['simulation_outcome'] == r['real_outcome'] for r in selected) / len(selected) if selected else None
        stats['direction'] = 'real minus simulation'
        groups[policy] = stats
    return {'schema_version': 1, 'study_id': protocol['study_id'], 'planned_pairs': len(pairs),
            'eligible_pairs': len(accepted), 'excluded_pairs': len(pairs) - len(accepted),
            'status': 'complete' if len(accepted) == len(pairs) and pairs else 'incomplete',
            'global_issues': global_issues, 'groups': groups, 'pairs': rows,
            'source': {'path': str(path), 'sha256': digest(path)},
            'interpretation': 'Measured correspondence and reviewed evidence required. Statistical agreement is not proof of predictive validity; no imputation of unknown or missing outcomes.'}
