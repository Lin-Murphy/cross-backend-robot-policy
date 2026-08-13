"""Audit one local sim-v1 page from saved evidence; no policy or hardware access."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('page_dir', type=Path)
    args = parser.parse_args()
    page = args.page_dir.resolve()
    report = {'schema_version': 1, 'stage': 'local_sim_v1_release_audit',
              'hardware_access': False, 'model_inference': False,
              'checks': [], 'files': {}, 'status': 'failed'}

    def check(name, condition, detail=None):
        report['checks'].append({'check': name, 'passed': bool(condition), 'detail': detail})
        if not condition:
            raise ValueError(f'{name}: {detail}')

    def record(relative):
        path = page / relative
        check('file_exists:' + relative, path.is_file())
        report['files'][relative] = {'size_bytes': path.stat().st_size, 'sha256': digest(path)}
        return path

    def decode_video(relative):
        path = record(relative)
        process = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'],
                                 capture_output=True, text=True, timeout=60)
        check('video_decodes:' + relative, process.returncode == 0,
              process.stderr[-500:] if process.returncode else None)

    try:
        html = record('index.html').read_text()
        summary = json.loads(record('run/summary.json').read_text())
        budget = json.loads(record('run/budget.json').read_text())
        check('fixed_success', summary.get('status') == 'success')
        check('fixed_is_scripted_sim', summary.get('hardware_access') is False
              and summary.get('policy_used') is False)
        check('fixed_video_exit', summary.get('video_exit') == 0)
        check('zero_wrist', budget.get('initial_wrist_roll_deg') == 0)
        check('declared_scene', 'sim-v1-scene-20260925' in budget.get('scene', ''))
        check('fixed_task_events', [e.get('event') for e in summary.get('task_events', [])]
              == ['started', 'lift_confirmed', 'success'])
        check('html_video_references', all(f'src="{name}/dual-camera.mp4"' in html
              for name in ('run', 'act', 'smolvla')))
        event_rows = [json.loads(line) for line in record('run/events.jsonl').read_text().splitlines()]
        check('event_count_matches_control_steps', len(event_rows) == summary.get('control_steps'))
        check('event_steps_sequential', all(row.get('step') == i for i, row in enumerate(event_rows)))
        times = [row.get('facts', {}).get('sim_ns') for row in event_rows]
        check('event_times_monotonic', all(isinstance(t, int) for t in times)
              and all(later > earlier for earlier, later in zip(times, times[1:])))
        check('event_state_times_match_facts', all(row.get('state', {}).get('sim_ns') == t
              for row, t in zip(event_rows, times)))
        lift_time = summary['task_events'][1]['sim_ns']
        lift_rows = [row for row in event_rows if row['facts']['sim_ns'] == lift_time]
        check('lift_event_has_physical_evidence', len(lift_rows) == 1
              and lift_rows[0]['facts'].get('both_jaws_contact') is True
              and lift_rows[0]['facts'].get('bottom_clearance_m', 0) >= 0.01)
        final = event_rows[-1]
        check('success_event_matches_final_state', final.get('task_status') == 'success'
              and final['facts']['sim_ns'] == summary['task_events'][-1]['sim_ns']
              and final['facts'].get('zone') == 'inside'
              and final['facts'].get('supported_on_mat') is True
              and final['facts'].get('any_jaw_contact') is False)
        check('no_forbidden_or_numerical_events', all(
              row['facts'].get('forbidden_contact') is False
              and row['facts'].get('numerical_error') is False for row in event_rows))
        decode_video('run/dual-camera.mp4')
        analysis = json.loads(record('model-check.json').read_text())
        check('model_analysis_is_development', analysis.get('formal_trial') is False
              and analysis.get('hardware_access') is False
              and analysis.get('physical_mapping_verified') is False)
        for name in ('act', 'smolvla'):
            item = analysis['models'][name]
            model_summary = json.loads(record(name + '/summary.json').read_text())
            check(name + '_timeout', item.get('status') == 'timeout'
                  and model_summary.get('status') == 'timeout')
            check(name + '_not_formal_or_hardware', model_summary.get('formal_trial') is False
                  and model_summary.get('hardware_access') is False)
            check(name + '_count_consistency', item.get('selected_targets')
                  == model_summary.get('selected_actions'))
            check(name + '_no_contact', item.get('any_jaw_contact_ticks') == 0)
            check(name + '_video_exit', item.get('video_exit') == 0
                  and model_summary.get('video_exit') == 0)
            decode_video(name + '/dual-camera.mp4')
        report['status'] = 'passed_local_release_audit'
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        (page / 'release-audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps({'status': report['status'], 'checks': len(report['checks']),
                          'page': str(page), 'error': report.get('error')}, ensure_ascii=False))
    return 0 if report['status'] == 'passed_local_release_audit' else 2


if __name__ == '__main__':
    sys.exit(main())
