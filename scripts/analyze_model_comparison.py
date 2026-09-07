#!/usr/bin/env python3
"""Verify and compare two policies evaluated by the unified entry point."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cross_backend.evaluation import write_json
from cross_backend.study_comparison import analyze_model_comparison, render_model_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New report directory')
    args = p.parse_args()
    report = analyze_model_comparison(args.run, args.protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'comparison.json', report)
    (args.output / 'report.md').write_text(render_model_report(report))
    print(args.output / 'report.md')


if __name__ == '__main__':
    main()
