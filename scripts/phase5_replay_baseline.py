"""Phase 5 replay baseline from one saved observation, with no hardware."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cross_backend.local_metadata import load_policy_metadata
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors

ROOT = Path(r"C:\Users\Murphy\.cache\huggingface\lerobot\Murphy-Lin\lerobot101_dataset_a")
CHECKPOINT = Path(r"D:\project\lerobot-seeed\outputs\train\act_test\checkpoints\000020\pretrained_model")
OBS = Path("artifacts/safety/act-gripper-one-shot-20260918T124427064688Z.png")
REAL_TRACE = Path("artifacts/safety/act-gripper-one-shot-20260918T124427064688Z.json")


def main() -> None:
    from PIL import Image
    image = np.asarray(Image.open(OBS).convert("RGB"))
    meta = load_policy_metadata(ROOT)
    config = ACTConfig.from_pretrained(CHECKPOINT)
    config.pretrained_path, config.device = CHECKPOINT, "cpu"
    policy = make_policy(config, ds_meta=meta).eval()
    pre, post = make_pre_post_processors(config, pretrained_path=str(CHECKPOINT), dataset_stats=meta.stats)
    trace = json.loads(REAL_TRACE.read_text(encoding="utf-8"))
    state = np.asarray(trace["state"], dtype=np.float32)
    batch = pre({"observation.state": torch.from_numpy(state).unsqueeze(0),
                 "observation.images.front": torch.from_numpy(image).permute(2, 0, 1).float().div(255).unsqueeze(0)})
    with torch.inference_mode():
        replay_action = post(policy.predict_action_chunk(batch))[0, 0].cpu().numpy()
    real = trace["action"]
    real_action = np.asarray(real, dtype=np.float32)
    result = {"hardware_connected": False, "checkpoint": str(CHECKPOINT), "observation": str(OBS),
              "replay_action": replay_action.tolist(), "real_trace": str(REAL_TRACE),
              "real_action_reference": real_action.tolist(),
              "max_abs_difference": float(np.max(np.abs(replay_action - real_action))),
              "l2_difference": float(np.linalg.norm(replay_action - real_action))}
    output = Path("artifacts/phase5-replay-baseline.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output, result["max_abs_difference"], result["l2_difference"])


if __name__ == "__main__":
    main()
