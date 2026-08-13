from __future__ import annotations

import argparse
import json
from pathlib import Path

from cross_backend.evaluator import evaluate_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--expected-steps", type=int, default=20)
    args = parser.parse_args()
    result = evaluate_trace(json.loads(args.trace.read_text(encoding="utf-8")), args.expected_steps)
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
