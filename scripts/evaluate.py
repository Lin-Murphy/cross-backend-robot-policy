#!/usr/bin/env python3
"""Unified evaluation configuration, launch and result reporting."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cross_backend.evaluation import load_config, make_plan, run_evaluation
from cross_backend.preflight import check_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='New output directory; required for evaluation or dry-run')
    parser.add_argument('--python', help='Override the model Python interpreter')
    preview = parser.add_mutually_exclusive_group()
    preview.add_argument('--dry-run', action='store_true', help='Print resolved plan without launching a model or accessing devices')
    preview.add_argument('--check-environment', action='store_true', help='Read-only resource and interpreter checks; no weights loaded or devices opened')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--validate-only', action='store_true', help='SO101 configuration validation without hardware access')
    mode.add_argument('--execute-approved-once', action='store_true', help='SO101 only: execute one previously approved on-site plan')
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.python:
            config['python'] = os.path.abspath(os.path.expanduser(args.python)) if '/' in args.python else args.python
        if not args.check_environment and args.output is None:
            parser.error('--output is required unless --check-environment is used')
        if args.check_environment:
            if args.execute_approved_once or args.validate_only:
                parser.error('--check-environment cannot be combined with hardware execution/validation flags')
            report = check_config(config, args.output)
            print(json.dumps(report, indent=2))
            return 0 if report['status'] == 'passed' else 2
        plan = make_plan(config, args.output, approved=args.execute_approved_once,
                         validate_only=args.validate_only or (args.dry_run and config['backend'] == 'so101'))
        if args.dry_run:
            print(json.dumps(plan, indent=2))
            return 0
        report = check_config(config, args.output)
        if report['status'] != 'passed':
            print(json.dumps(report, indent=2))
            return 2
        result = run_evaluation(plan, args.output, preflight=report)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f'Evaluation configuration error: {exc}\n')
    print(json.dumps({'results': str(args.output.resolve() / 'results.json'),
                      'report': str(args.output.resolve() / 'report.md'),
                      'execution_status': result['execution_status'], 'comparison': result['comparison']}, indent=2))
    return 0 if result['execution_status'] == 'completed' else 130 if result['execution_status'] == 'interrupted' else 1


if __name__ == '__main__':
    raise SystemExit(main())
