"""Import a historical LeRobot pilot into an explicitly non-formal trial ledger."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import pyarrow.parquet as pq


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main(dataset, labels_path, out):
    out.mkdir(parents=True, exist_ok=False)
    info = json.loads((dataset / 'meta/info.json').read_text())
    labels = json.loads(labels_path.read_text())
    episodes_path = dataset / 'meta/episodes/chunk-000/file-000.parquet'
    data_path = dataset / 'data/chunk-000/file-000.parquet'
    task_path = dataset / 'meta/tasks.parquet'
    episodes = pq.read_table(episodes_path, columns=['episode_index', 'length', 'dataset_from_index', 'dataset_to_index',
        'videos/observation.images.follower/from_timestamp', 'videos/observation.images.follower/to_timestamp',
        'videos/observation.images.camera2/from_timestamp', 'videos/observation.images.camera2/to_timestamp']).to_pylist()
    data = pq.read_table(data_path, columns=['episode_index', 'timestamp', 'frame_index', 'index']).to_pydict()
    tasks = pq.read_table(task_path).to_pylist()
    assert len(tasks) == 1 and tasks[0]['task'] == 'move pot'
    assert len(episodes) == info['total_episodes'] == len(labels['episode_status']) == 10
    assert len(data['index']) == info['total_frames'] == 5055
    assert [r['episode_index'] for r in episodes] == list(range(10))
    assert sum(r['length'] for r in episodes) == 5055
    rows = []
    for item in episodes:
        i = item['episode_index']; start = item['dataset_from_index']; end = item['dataset_to_index']
        assert end - start == item['length']
        assert all(v == i for v in data['episode_index'][start:end])
        assert data['frame_index'][start:end] == list(range(item['length']))
        assert data['index'][start:end] == list(range(start, end))
        status = labels['episode_status'][str(i)]
        assert status in ('success', 'failed', 'uncertain')
        segment = {name: {'from_s': item[f'videos/observation.images.{name}/from_timestamp'],
                          'to_s': item[f'videos/observation.images.{name}/to_timestamp']}
                   for name in ('follower', 'camera2')}
        rows.append({'trial_id': f'legacy-so101-smolvla-{i:03d}', 'episode_index': i,
                     'backend': 'so101', 'task': 'move pot', 'policy': 'SmolVLA',
                     'status': status, 'outcome_source': labels['source'],
                     'human_intervention': False if status == 'success' else None,
                     'frame_count': item['length'], 'dataset_frame_range': [start, end],
                     'dataset_duration_s': max(data['timestamp'][start:end]),
                     'video_segment_s': segment, 'camera_capture_ns': None,
                     'state_capture_ns': None, 'dispatch_monotonic_ns': None,
                     'hardware_dispatch_verified': False,
                     'evaluation_scope': 'historical_development_pilot_not_formal'})
    sources = [dataset / 'meta/info.json', episodes_path, data_path, task_path]
    video_info = {}
    for camera in ('follower', 'camera2'):
        path = dataset / f'videos/observation.images.{camera}/chunk-000/file-000.mp4'
        sources.append(path)
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,duration,nb_frames', '-of', 'json', str(path)],
            check=True, capture_output=True, text=True)
        video_info[camera] = json.loads(probe.stdout)['streams'][0]
        assert video_info[camera]['width'] == 640 and video_info[camera]['height'] == 480
        assert float(video_info[camera]['duration']) >= max(r['video_segment_s'][camera]['to_s'] for r in rows) - .1
    payload = {'scope': 'historical_pilot_import_not_formal_or_new_robot_trial',
               'backend': 'so101', 'task': 'move pot', 'policy': 'SmolVLA',
               'source_dataset': str(dataset), 'dataset_info': {'codebase_version': info['codebase_version'],
                    'fps': info['fps'], 'total_episodes': info['total_episodes'], 'total_frames': info['total_frames']},
               'outcome_label_source': labels['source'], 'rows': rows,
               'summary': {'episodes': len(rows), 'operator_labeled_successes': sum(r['status']=='success' for r in rows),
                           'success_fraction': sum(r['status']=='success' for r in rows)/len(rows)},
               'timing_limit': 'Dataset timestamps are not verified physical exposure or dispatch timestamps',
               'action_limit': 'Saved actions are not verified raw motor writes',
               'video_stream_info': video_info,
               'source_sha256': {str(p): sha256(p) for p in sources},
               'labels_sha256': sha256(labels_path)}
    (out / 'historical-trials.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'status':'passed','episodes':len(rows),'successes':payload['summary']['operator_labeled_successes'],
                      'frames':sum(r['frame_count'] for r in rows),'formal_trial_count':0},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    main(args.dataset, args.labels, args.out)
