"""C-only reset/settle check for a versioned MuJoCo initial-condition file."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL','egl')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cross_backend.sim_trial_initial import apply_initial_condition
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.tape_task import TapeTaskEvaluator
from cross_backend.tape_task_observer import observe_task
from cross_backend.png_write import write_png


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene',type=Path,required=True)
    parser.add_argument('--initial-condition',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    backend=TapeSimBackend(args.scene)
    try:
        metadata=apply_initial_condition(args.initial_condition,backend,expected_split='C')
        target=backend.data.qpos[backend.qadr].copy()
        for _ in range(50):backend.step_sim_targets(target)
        facts=observe_task(backend)
        evaluator=TapeTaskEvaluator();status=evaluator.update(facts)
        images=backend.render()
        for name,image in images.items():write_png(args.output/(name.rsplit('.',1)[-1]+'.png'),image)
        result={'status':'passed_C_initial_check' if status=='running' else 'rejected_C_initial_check',
                'initial_condition_id':metadata['initial_condition_id'],
                'initial_file_sha256':metadata['initial_file_sha256'],
                'initial_geometry_sha256':metadata['initial_geometry_sha256'],
                'scene_sha256':metadata['scene_sha256'],
                'sim_ns':backend.state()['sim_ns'],'task_zone':facts.zone,
                'supported_on_table':facts.supported_on_table,
                'forbidden_contact':facts.forbidden_contact,
                'minimum_contact_distance_m':min((c['distance_m'] for c in backend.contacts()),default=None),
                'hardware_access':False,'policy_inference':False,'formal_trial':False}
        if result['minimum_contact_distance_m'] is not None and result['minimum_contact_distance_m']<-.002:
            result['status']='rejected_C_initial_check'
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))
        return 0 if result['status']=='passed_C_initial_check' else 2
    finally:backend.close()


if __name__=='__main__':raise SystemExit(main())
