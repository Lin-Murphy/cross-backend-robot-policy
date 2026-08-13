"""Load the subset of LeRobot metadata needed for policy inference."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_policy_metadata(root: Path) -> SimpleNamespace:
    meta = root / "meta"
    info = json.loads((meta / "info.json").read_text(encoding="utf-8"))
    raw_stats = json.loads((meta / "stats.json").read_text(encoding="utf-8"))
    stats = {
        feature: {name: np.asarray(value, dtype=np.float32) for name, value in values.items()}
        for feature, values in raw_stats.items()
    }
    return SimpleNamespace(features=info["features"], stats=stats, info=info)
