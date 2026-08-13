from __future__ import annotations

import numpy as np


class ActionSafetyError(ValueError):
    pass


def validate_action(
    action: np.ndarray,
    previous: np.ndarray | None = None,
    lower: float = -100.0,
    upper: float = 100.0,
    max_delta: float = 25.0,
) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 1:
        raise ActionSafetyError(f"expected 1D action, got {action.shape}")
    if not np.isfinite(action).all():
        raise ActionSafetyError("action contains NaN or Inf")
    if (action < lower).any() or (action > upper).any():
        raise ActionSafetyError(f"action outside [{lower}, {upper}]")
    if previous is not None:
        delta = np.abs(action - previous)
        if (delta > max_delta).any():
            raise ActionSafetyError(f"action delta exceeds {max_delta}")
    return action


class NoDispatch:
    """Explicit sink used while policy outputs are not authorized for hardware."""

    hardware_connected = False

    def send(self, action: np.ndarray) -> None:
        raise RuntimeError("hardware dispatch is disabled in this backend")


class SO101CommandGate:
    """Software-only gate in normalized units; caller supplies verified joint bounds.

    Does not send commands or implement a physical emergency stop.
    A stopped instance cannot be rearmed.
    """

    def __init__(self, lower, upper, max_speed, scale=1.0):
        self.lower = self._vector(lower)
        self.upper = self._vector(upper)
        self.max_speed = self._vector(max_speed)
        envelope_lower = np.array([-100.] * 5 + [0.])
        if (self.lower < envelope_lower).any() or (self.upper > 100).any():
            raise ActionSafetyError("bounds outside normalized SO101 envelope")
        if (self.lower >= self.upper).any() or (self.max_speed <= 0).any():
            raise ActionSafetyError("invalid limits")
        if not np.isfinite(scale) or not 0 < scale <= 1:
            raise ActionSafetyError("scale must be in (0, 1]")
        self.scale = scale
        self.stopped = False

    @staticmethod
    def _vector(value):
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (6,) or not np.isfinite(array).all():
            raise ActionSafetyError("expected six finite joint values")
        return array.copy()

    def stop(self):
        self.stopped = True

    def prepare(self, target, measured, previous_command, dt):
        if self.stopped:
            raise ActionSafetyError("stop is latched")
        try:
            target, measured, previous = map(self._vector, (target, measured, previous_command))
            if not np.isfinite(dt) or not 0 < dt <= 0.1:
                raise ActionSafetyError("invalid or stale control interval")
            for value in (target, measured, previous):
                if (value < self.lower).any() or (value > self.upper).any():
                    raise ActionSafetyError("joint outside configured bounds")
            # Scale relative displacement, never the absolute joint position.
            desired = measured + self.scale * (target - measured)
            budget = self.max_speed * dt
            command = previous + np.clip(desired - previous, -budget, budget)
            if (np.abs(command - measured) > budget + 1e-9).any():
                raise ActionSafetyError("command too far from measured position")
            return command
        except (ValueError, TypeError):
            self.stop()
            raise
