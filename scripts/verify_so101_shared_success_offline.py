"""Read-only verification of the frozen SO101 shared-backend success evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cross_backend.completion_record import CompletionRecord
from audit_so101_shared_live_boundary import audit
from compare_completion_records import compare

DEFAULT_MANIFEST = ROOT / 'artifacts/so101-shared-success-freeze-20260927/manifest.json'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def video_info(path):
    command = ['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height,nb_read_frames', '-of', 'json', str(path)]
    data = json.loads(subprocess.check_output(command, text=True))
    streams = data.get('streams', [])
    if len(streams) != 1:
        raise ValueError(f'exactly one video stream required: {path}')
    stream = streams[0]
    return {key: int(stream[key]) for key in ('width', 'height', 'nb_read_frames')}


def verify(manifest_path=DEFAULT_MANIFEST, root=ROOT):
    manifest = json.loads(manifest_path.read_text())
    if manifest['schema_version'] != 1 or manifest['kind'] != 'so101_shared_success_offline_freeze':
        raise ValueError('wrong evidence manifest')
    checked = []
    for item in manifest['files']:
        path = root / item['path']
        if not path.is_file() or path.stat().st_size != item['size_bytes'] or digest(path) != item['sha256']:
            raise ValueError(f'frozen evidence changed: {item["path"]}')
        checked.append(item['path'])
    paths = manifest['paths']
    run = root / paths['run']
    summary = json.loads((run / 'summary.json').read_text())
    evaluation = json.loads((root / paths['evaluation']).read_text())
    completion = json.loads((root / paths['completion']).read_text())
    if summary['status'] != 'ended_unreviewed' or summary['raw_goal_packets_transmitted'] != 494 or \
       not summary['shared_boundary_enabled'] or summary['formal_trial_count'] != 0:
        raise ValueError('raw run does not match frozen success identity')
    if evaluation['record']['status'] != 'success' or evaluation['record']['outcome'] is not True or \
       evaluation['formal_trial_count'] != 0 or evaluation['shared_receipt_accepted'] != 494:
        raise ValueError('RunRecord does not match frozen success identity')
    reconstructed = CompletionRecord(completion['backend'], completion['task'],
        completion['run_status'], completion['stages'], tuple(completion['task_required']),
        tuple(completion['full_cycle_required']), completion['evidence_sha256']).to_dict()
    if reconstructed != completion or completion['task_state'] != 'yes' or \
       completion['full_cycle_state'] != 'no':
        raise ValueError('completion milestones changed or inconsistent')
    evidence_files = {'raw_summary': run / 'summary.json',
                      'hardware_events': run / 'hardware-events.jsonl',
                      'operator': root / paths['operator'],
                      'post_scene': root / paths['post_scene']}
    if any(digest(path) != completion['evidence_sha256'][key]
           for key, path in evidence_files.items()):
        raise ValueError('completion source digest mismatch')
    packet_audit = audit(run, root / paths['preflight_scene'])
    prior_audit = json.loads((root / paths['ordered_audit']).read_text())
    for key in ('status', 'paired_observations_actions_packets', 'unique_observation_ids',
                'stop_hold_after_last_receipt', 'summary_sha256', 'events_sha256',
                'calibration_summary_sha256'):
        if packet_audit[key] != prior_audit[key]:
            raise ValueError(f'ordered packet audit changed: {key}')
    if packet_audit['paired_observations_actions_packets'] != 494:
        raise ValueError('ordered packet count mismatch')
    compared = compare(root / paths['completion'], root / paths['simulation_completion'])
    if compared['task_states'] != {'left': 'yes', 'right': 'yes'} or \
       compared['full_cycle_states'] != {'left': 'no', 'right': 'unknown'} or \
       compared['formal_performance_comparison']:
        raise ValueError('cross-backend completion semantics changed')
    videos = {camera: video_info(run / f'dataset/videos/observation.images.{camera}/chunk-000/file-000.mp4')
              for camera in ('follower', 'camera2')}
    if any(info != {'width': 640, 'height': 480, 'nb_read_frames': 494}
           for info in videos.values()):
        raise ValueError('video frame count or resolution mismatch')
    return {'status': 'passed', 'manifest_sha256': digest(manifest_path),
            'files_checked': len(checked), 'real_goal_packets': 494,
            'ordered_audit': packet_audit['status'], 'RunRecord': 'success',
            'real_task_state': completion['task_state'],
            'real_full_cycle_state': completion['full_cycle_state'],
            'simulation_task_state': compared['right']['task_state'],
            'formal_trial_count': 0, 'formal_performance_comparison': False,
            'videos': videos, 'hardware_access': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))
