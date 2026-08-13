"""Read-only R0 refresh; no model loading, hardware access or dataset changes."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import subprocess
import numpy as np
import pyarrow.parquet as pq

PROJECT = Path(__file__).resolve().parents[1]
BASE = Path('/home/murphy/.cache/huggingface/lerobot')
REPO = Path('/home/murphy/project/lerobot')
OUT = PROJECT / 'artifacts/r0-refresh-20260923'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def main():
    OUT.mkdir(exist_ok=False)
    sources = {}
    def identity(p):
        sources[str(p)] = sha(p)
        return {'path': str(p), 'sha256': sources[str(p)], 'bytes': p.stat().st_size}
    result = {'checked_utc': datetime.now(timezone.utc).isoformat(), 'datasets': [],
              'calibrations': [], 'sources': [], 'checkpoints': [],
              'project_git': (PROJECT / '.git').exists()}
    result['lerobot_head'] = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
    result['lerobot_status'] = subprocess.check_output(['git', '-C', str(REPO), 'status', '--porcelain'], text=True)
    all_groups = {}
    for name in ['Murphy-Lin/lerobot101_dataset_a_20260919_114052',
                 'local/move_orange_20260919_194524', 'local/move_pot_20260923_142914']:
        root = BASE / name
        info = json.loads((root / 'meta/info.json').read_text())
        files = sorted(p for p in root.rglob('*') if p.is_file())
        rows = [r for p in sorted((root / 'data').rglob('*.parquet')) for r in pq.read_table(p).to_pylist()]
        d = {'repo_id': name, 'files': [identity(p) for p in files],
             'declared_episodes': info['total_episodes'], 'declared_frames': info['total_frames'],
             'actual_frames': len(rows), 'features': info['features'], 'episodes': []}
        for eid in sorted(set(r['episode_index'] for r in rows)):
            ep = sorted((r for r in rows if r['episode_index'] == eid), key=lambda r: r['frame_index'])
            a = np.asarray([r['action'] for r in ep], dtype='<f4')
            s = np.asarray([r['observation.state'] for r in ep], dtype='<f4')
            h = hashlib.sha256(a.tobytes() + s.tobytes()).hexdigest()
            all_groups.setdefault(h, []).append({'dataset': name, 'episode_index': eid})
            d['episodes'].append({'episode_index': eid, 'frames': len(ep),
                                  'state_action_sha256': h, 'finite': bool(np.isfinite(a).all() and np.isfinite(s).all())})
        result['datasets'].append(d)
    result['exact_state_action_duplicates'] = [g for g in all_groups.values() if len(g) > 1]
    result['duplicate_scope'] = 'Exact complete state/action sequences only; not near-duplicate scene or shared-session independence.'
    old = json.loads((PROJECT / 'artifacts/r0-audit-20260920-run01/provenance.json').read_text())
    old_cal = {c['path']: c['sha256'] for c in old['calibrations']}
    for p in sorted((BASE / 'calibration').rglob('*.json')):
        if 'so101' not in p.name:
            continue
        item = identity(p)
        item['matches_20260920_same_path'] = old_cal.get(str(p)) == item['sha256'] if str(p) in old_cal else None
        item['acquisition_binding'] = 'not_established'
        result['calibrations'].append(item)
        (OUT / ('calibration_' + p.parent.name + '_' + p.name)).write_bytes(p.read_bytes())
    for rel in ['robots/so_follower/config_so_follower.py', 'robots/so_follower/so_follower.py',
                'teleoperators/so_leader/config_so_leader.py', 'teleoperators/so_leader/so_leader.py',
                'scripts/lerobot_record.py', 'motors/motors_bus.py']:
        p = REPO / 'src/lerobot' / rel
        result['sources'].append(identity(p))
        (OUT / rel.replace('/', '__')).write_bytes(p.read_bytes())
    for name, step in [('act_orange_to_box', '020000'), ('smolvla_move_orange_v7', '040000'),
                       ('smolvla_move_pot_dualcam_20260923_v3', '040000')]:
        cp = REPO / 'outputs/train' / name / 'checkpoints' / step / 'pretrained_model'
        cfg = json.loads((cp / 'train_config.json').read_text())
        result['checkpoints'].append({'name': name, 'path': str(cp),
             'files': [identity(p) for p in sorted(cp.iterdir()) if p.is_file()],
             'dataset': cfg.get('dataset'), 'seed': cfg.get('seed'), 'steps': cfg.get('steps')})
    prior = json.loads((PROJECT / 'artifacts/move-pot-video-check-20260923/summary.json').read_text())
    result['move_pot_prior_video_qa_identity'] = [
        {'path': f['path'], 'matches_prior_sha256': sources.get(f['path']) == f['sha256']}
        for camera in prior['cameras'].values() for f in camera['files']]
    save('summary.json', result)
    after = {p: sha(Path(p)) == h for p, h in sources.items()}
    save('input_integrity.json', {'all_unchanged': all(after.values()), 'files': after})
    save('split_constraints.json', {
        'status': 'not_frozen_pending_task_and_provenance',
        'existing_act_seen': list(range(15)), 'existing_move_pot_smolvla_seen': list(range(30)),
        'existing_checkpoint_unseen_test_from_own_training_data': [],
        'move_pot_future_retraining_proposal': {'train': list(range(21)), 'validation': list(range(21, 25)),
            'test': list(range(25, 30)), 'restriction': 'Provisional episode split within one session; not independent-session test; new training only, training-only statistics required.'},
        'rollout_evaluation': 'Keep separate from demonstrations; already observed outcomes are pilot evidence, not fresh final test.',
        'old_combined': 'Use prior combined_lineage.json; copies remain with original episodes.'})
    (OUT / 'executed_script.py').write_bytes(Path(__file__).read_bytes())
    (OUT / 'run.log').write_text(f"Completed {result['checked_utc']}\nDatasets=3; input files={len(after)}; all unchanged={all(after.values())}\n")
    print(json.dumps({'output': str(OUT), 'input_files': len(after), 'all_unchanged': all(after.values()),
                      'duplicates': result['exact_state_action_duplicates'],
                      'calibrations': result['calibrations'], 'video_identity': result['move_pot_prior_video_qa_identity']}, indent=2))

if __name__ == '__main__':
    main()
