from dataclasses import dataclass

import numpy as np

from .contracts import Observation


@dataclass
class TrackingSimulator:
    """Small deterministic closed-loop backend; not a physics or hardware model."""

    state_dim: int = 6
    target: float = 10.0
    response: float = 0.25
    max_steps: int = 20
    action_delay_steps: int = 0

    def __post_init__(self) -> None:
        self.state = np.zeros(self.state_dim, dtype=np.float32)
        self.step_count = 0
        self._pending_actions: list[np.ndarray] = []
        if self.action_delay_steps < 0:
            raise ValueError("action_delay_steps must be non-negative")

    def reset(self) -> Observation:
        self.state.fill(0.0)
        self.step_count = 0
        self._pending_actions.clear()
        return Observation(self.state.copy(), self.step_count)

    def step(self, action: np.ndarray) -> tuple[Observation, bool]:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (self.state_dim,):
            raise ValueError(f"expected action shape {(self.state_dim,)}, got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("action contains NaN or Inf")
        self._pending_actions.append(action.copy())
        if len(self._pending_actions) > self.action_delay_steps:
            applied_action = self._pending_actions.pop(0)
        else:
            applied_action = self.state.copy()
        self.state += self.response * (applied_action - self.state)
        self.step_count += 1
        done = self.step_count >= self.max_steps
        return Observation(self.state.copy(), self.step_count), done
