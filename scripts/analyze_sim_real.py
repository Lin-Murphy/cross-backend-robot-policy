#!/usr/bin/env python3
"""Check measured sim/real pairs and report agreement; never operates hardware."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cross_backend.evaluation import write_json
from cross_backend.paired_backend_study import analyze_sim_real

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--require-complete', action='store_true')
    a = p.parse_args()
    result = analyze_sim_real(a.manifest)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        p.error('Output exists; choose a new report path')
    write_json(a.output, result)
    print(f"{result['status']}: {result['eligible_pairs']}/{result['planned_pairs']} eligible pairs; {a.output}")
    raise SystemExit(1 if a.require_complete and result['status'] != 'complete' else 0)
