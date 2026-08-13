from __future__ import annotations

from typing import Any

import numpy as np


def evaluate_trace(trace: dict[str, Any], expected_steps: int) -> dict[str, Any]:
    steps = trace.get("steps", [])
    finite = bool(steps)
    errors = []
    state_dim = None
    for step in steps:
        try:
            action = np.asarray(step['action'], dtype=np.float64)
            state = np.asarray(step['state'], dtype=np.float64)
            error = float(step['tracking_error_l2'])
            if state_dim is None:
                state_dim = state.shape
            valid = (action.ndim == state.ndim == 1 and action.size > 0
                     and action.shape == state.shape == state_dim
                     and np.isfinite(action).all() and np.isfinite(state).all()
                     and np.isfinite(error) and error >= 0)
            finite = finite and bool(valid)
            if np.isfinite(error) and error >= 0:
                errors.append(error)
        except (KeyError, TypeError, ValueError, OverflowError):
            finite = False
    passed = (
        type(expected_steps) is int and expected_steps > 0
        and trace.get("hardware_connected") is False
        and len(steps) == expected_steps
        and finite and len(errors) == len(steps)
    )
    return {
        "passed": passed,
        "hardware_connected": trace.get("hardware_connected"),
        "steps": len(steps),
        "expected_steps": expected_steps,
        "finite_values": finite,
        "mean_tracking_error_l2": float(np.mean(errors)) if errors else None,
        "final_tracking_error_l2": errors[-1] if errors else None,
    }
