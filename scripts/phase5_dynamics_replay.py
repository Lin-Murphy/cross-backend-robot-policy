"""Replay one saved ACT action trace against varied toy dynamics (stdlib only)."""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts/phase5-timing-mismatch.json"
OUTPUT = ROOT / "artifacts/phase5-dynamics-replay.json"
RESPONSES = (0.10, 0.25, 0.40)
TARGET = 10.0


def evaluate(actions: list[list[float]], response: float) -> dict:
    state = [0.0] * len(actions[0])
    errors = []
    for action in actions:
        state = [s + response * (a - s) for s, a in zip(state, action)]
        errors.append(math.sqrt(sum((TARGET - s) ** 2 for s in state)))
    return {
        "response": response,
        "steps": len(errors),
        "mean_tracking_error_l2": sum(errors) / len(errors),
        "final_tracking_error_l2": errors[-1],
    }


def main() -> None:
    matrix = json.loads(SOURCE.read_text(encoding="utf-8"))
    baseline = next(run for run in matrix["runs"] if run["name"] == "timing_baseline")
    actions = [step["action"] for step in baseline["steps"]]
    runs = [evaluate(actions, response) for response in RESPONSES]
    result = {
        "experiment": "frozen ACT action trace under varied tracking-simulator response",
        "policy": "ACT actions recorded in the Phase 5 timing baseline",
        "source_artifact": "artifacts/phase5-timing-mismatch.json",
        "hardware_connected": False,
        "closed_loop_policy": False,
        "action_trace_replayed_open_loop": True,
        "target_per_joint": TARGET,
        "interpretation_boundary": (
            "Only the toy simulator response coefficient changes. Policy actions are frozen from "
            "the recorded baseline, so this measures open-loop trace sensitivity, not how an ACT "
            "policy would adapt under changed dynamics or how SO101 behaves."
        ),
        "runs": runs,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT.relative_to(ROOT))
    for run in runs:
        print(f"response={run['response']:.2f} mean={run['mean_tracking_error_l2']:.6f} "
              f"final={run['final_tracking_error_l2']:.6f}")


if __name__ == "__main__":
    main()
