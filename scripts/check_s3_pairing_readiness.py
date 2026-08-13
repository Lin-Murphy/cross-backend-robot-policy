"""Read-only S3 pairing inspection; does not start simulation or hardware trials."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cross_backend.sim_real_pairing import inspect_t_pairing
from move_pot_trial_records import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sim-t', type=Path, required=True)
    parser.add_argument('--real-directory', type=Path,
                        default=ROOT / 'artifacts/move-pot-evaluation-v1-records')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sim = json.loads(args.sim_t.read_text())
    real = json.loads((args.real_directory / 'trials.json').read_text())
    result = inspect_t_pairing(sim, real, real_validation=validate(args.real_directory))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'pairs'}, ensure_ascii=False))
    # Structural pairing alone never passes the S3 evidence/outcome gate.
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
