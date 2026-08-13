"""Training/development diagnostics only: no model/hardware calls or modifications."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main(out):
    import numpy as np
    import pyarrow.parquet as pq
    from cross_backend.move_pot_policy import JOINTS, UNITS, file_sha256
    dataset = Path('/home/murphy/.cache/huggingface/lerobot/local/move_pot_20260923_142914')
    paths = sorted((dataset / 'data').rglob('*.parquet'))
    before = {str(p): file_sha256(p) for p in paths}
    rows = [r for p in paths for r in pq.read_table(p, columns=['episode_index','frame_index','observation.state','action']).to_pylist()]
    states = np.asarray([r['observation.state'] for r in rows])
    actions = np.asarray([r['action'] for r in rows])
    assert states.shape == actions.shape == (10769, 6)
    assert np.isfinite(states).all() and np.isfinite(actions).all()
    starts = [min((i for i,r in enumerate(rows) if r['episode_index']==e), key=lambda i:rows[i]['frame_index']) for e in sorted({r['episode_index'] for r in rows})]
    cap = ROOT / 'artifacts/r2-live-readonly-20260924-attempt01/summary.json'
    live = np.asarray(json.loads(cap.read_text())['samples'][0]['state'])
    def quantiles(values):
        return {str(p): np.percentile(values, p, axis=0).tolist() for p in (0, 50, 95, 100)}
    def nearest(indices):
        # Body joints share degrees; exclude gripper from distance instead of mixing units.
        errors = np.abs(states[indices, :5] - live[:5]).max(axis=1)
        order = np.argsort(errors)[:5]
        return [{'episode': rows[indices[j]]['episode_index'], 'frame': rows[indices[j]]['frame_index'],
                 'max_body_difference_deg': float(errors[j]), 'state': states[indices[j]].tolist(),
                 'action_minus_state': (actions[indices[j]]-states[indices[j]]).tolist()} for j in order]
    result = {'data_role': 'all train/development; not task evaluation', 'joint_names': JOINTS, 'units': UNITS,
        'dataset_manifest': before, 'capture_summary_sha256': file_sha256(cap), 'live_state': live.tolist(),
        'episode_count': len(starts), 'frame_count': len(rows),
        'episode_start_states': [{'episode': rows[i]['episode_index'], 'frame':rows[i]['frame_index'],
            'state':states[i].tolist(), 'action':actions[i].tolist()} for i in starts],
        'start_state_quantiles': quantiles(states[starts]), 'all_state_quantiles': quantiles(states),
        'absolute_action_minus_state_quantiles': quantiles(np.abs(actions-states)),
        'nearest_episode_starts': nearest(np.asarray(starts)), 'nearest_all_frames': nearest(np.arange(len(rows))),
        'models': {}, 'hardware_dispatches': 0}
    pred_dir = ROOT / 'artifacts/r2-captured-dual-policy-20260924-attempt01'
    for name in ('act', 'smolvla'):
        predicted = np.load(pred_dir / f'{name}-00-actions.npz')['actions'][0]
        delta = np.abs(predicted - live)
        result['models'][name] = {'first_action': predicted.tolist(), 'absolute_delta': delta.tolist(),
            'training_abs_action_state_delta_percentile_rank': (100*(np.abs(actions-states) <= delta).mean(axis=0)).tolist(),
            'interpretation': 'marginal descriptive ranks only, not paired prediction error or causal diagnosis'}
    assert all(file_sha256(p)==h for p,h in before.items())
    result['dataset_unchanged'] = True
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('nearest_episode_starts','nearest_all_frames','models')},indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    main(args.output)
