from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class Observation:
    state: np.ndarray
    step: int


class Policy(Protocol):
    def predict(self, observation: Observation) -> np.ndarray: ...


class Backend(Protocol):
    def reset(self) -> Observation: ...
    def step(self, action: np.ndarray) -> tuple[Observation, bool]: ...
