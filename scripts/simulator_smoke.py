from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cross_backend.simulator import TrackingSimulator


def main() -> None:
    backend = TrackingSimulator()
    observation = backend.reset()
    trace = []
    while True:
        action = np.full(6, backend.target, dtype=np.float32)
        observation, done = backend.step(action)
        trace.append({"step": observation.step, "state": observation.state.tolist()})
        if done:
            break
    output = Path("artifacts/simulator/smoke_trace.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"backend": "tracking_simulator", "steps": trace}, indent=2), encoding="utf-8")
    print(f"simulator smoke complete: {len(trace)} steps -> {output}")


if __name__ == "__main__":
    main()
