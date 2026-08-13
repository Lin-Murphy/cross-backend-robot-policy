"""Independent comparison of retained synchronous and six-period lookahead runs."""
import argparse
import json
from pathlib import Path
import numpy as np


def main(root):
    base=json.loads((root/'run01/summary.json').read_text())
    new=json.loads((root/'prefetch06/summary.json').read_text())
    assert base['status']==new['status']=='passed' and new['prefetch_steps']==6
    result={}
    for model in ('act','smolvla'):
        assert base['models'][model]['manifest']==new['models'][model]['manifest']
        result[model]={}
        for condition in ('baseline','host_hold_33ms'):
            a=base['models'][model]['conditions'][condition];b=new['models'][model]['conditions'][condition]
            for i in (1,2):
                launch_tick=b['ticks'][(i-1)*50+44]['selected_ns']
                previous_tail=b['ticks'][(i-1)*50+49]['selected_ns']
                assert launch_tick<=b['blocks'][i]['host_start_ns']<previous_tail
                assert b['blocks'][i]['adapter_return_ns']<=b['blocks'][i]['future_collected_ns']<=b['blocks'][i]['queue_ready_ns']
            with np.load(root/'run01'/f'{model}-{condition}-actions.npz',allow_pickle=False) as x, np.load(root/'prefetch06'/f'{model}-{condition}-actions.npz',allow_pickle=False) as y:
                delta=float(np.max(np.abs(x['actions']-y['actions'])))
            assert delta<=1e-4
            result[model][condition]={'same_saved_input_action_max_abs_difference':delta,
                'synchronous_boundary_ms':a['boundary_intervals_ms'],'prefetch_boundary_ms':b['boundary_intervals_ms'],
                'synchronous_prediction_residence_max_ms':a['max_prediction_residence_ms'],
                'prefetch_prediction_residence_max_ms':b['max_prediction_residence_ms'],
                'prefetch_host_path_age_at_tail_ms':[(b['ticks'][i*50+49]['selected_ns']-row['host_start_ns'])/1e6 for i,row in enumerate(b['blocks'])],
                'prefetch_all_candidate_intervals_over_40ms':b['candidate_intervals_over_40ms'],
                'launch_after_offset44_before_offset49_verified':True}
    for name,fps in [('follower','30.000'),('camera2','60.000')]:
        text=(root/f'{name}-format-after.log').read_text()
        assert '640/480' in text and "'MJPG'" in text and fps in text
    (root/'correction-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS: output identity, clock order, six-period launch and unchanged camera formats')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True)
    main(parser.parse_args().directory)
