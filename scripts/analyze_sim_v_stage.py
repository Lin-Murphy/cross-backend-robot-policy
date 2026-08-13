"""Analyze a declared V trial manifest; incomplete evidence withholds outcome rates."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cross_backend.sim_study_analysis import analyze_v_stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--evidence-root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bootstrap-samples',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--require-file-complete',action='store_true')
    args=parser.parse_args()
    record=json.loads(args.manifest.read_text())
    result=analyze_v_stage(record,evidence_root=args.evidence_root,bootstrap_samples=args.bootstrap_samples,seed=args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'evidence_label':result['evidence_label'],'full_matrix_file_verified':result['full_matrix_file_verified'],'formal_stage_accepted':result['formal_stage_accepted'],'started':result['started'],'output':str(args.output)}))
    if args.require_file_complete and not result['full_matrix_file_verified']:
        raise SystemExit(2)


if __name__=='__main__':main()
