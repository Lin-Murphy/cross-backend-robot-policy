"""Create only the unstarted T simulation slot template from the frozen real order."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cross_backend.sim_real_pairing import make_t_template
from move_pot_trial_records import validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-directory',type=Path,
                        default=ROOT/'artifacts/move-pot-evaluation-v1-records')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    verification=validate(args.real_directory)
    if verification['record_structure']!='passed' or verification['slots']!=40:
        raise ValueError('Real ledger structural validation failed')
    real=json.loads((args.real_directory/'trials.json').read_text())
    template=make_t_template(real)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:json.dump(template,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'template':str(args.output.resolve()),'planned_slots':40,
                      'started_slots':0,'formal_T_initial_conditions_frozen':False,
                      'real_order_source_sha256':template['real_order_source_sha256']}))


if __name__=='__main__':main()
