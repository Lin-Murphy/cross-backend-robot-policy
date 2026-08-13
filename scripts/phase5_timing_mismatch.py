"""Phase 5 timing comparison using one ACT checkpoint and no hardware."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cross_backend.local_metadata import load_policy_metadata
from cross_backend.safety import NoDispatch, validate_action
from cross_backend.simulator import TrackingSimulator
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors


ROOT = Path(r"C:\Users\Murphy\.cache\huggingface\lerobot\Murphy-Lin\lerobot101_dataset_a")
CHECKPOINT = Path(r"D:\project\lerobot-seeed\outputs\train\act_test\checkpoints\000020\pretrained_model")
OBSERVATION = Path("artifacts/safety/official-env-preflight.npz")


def run(policy, pre, post, image, delay: int, action_scale: float = 1.0, state_noise: float = 0.0) -> dict:
    backend = TrackingSimulator(max_steps=20, action_delay_steps=delay)
    obs = backend.reset()
    previous = None
    records = []
    NoDispatch().hardware_connected = False
    with torch.inference_mode():
        for _ in range(backend.max_steps):
            batch = pre({
                "observation.state": torch.from_numpy(obs.state + state_noise).unsqueeze(0),
                "observation.images.front": torch.from_numpy(image).permute(2, 0, 1).float().div(255).unsqueeze(0),
            })
            action = post(policy.predict_action_chunk(batch))[0, 0].cpu().numpy() * action_scale
            action = validate_action(action, previous)
            obs, done = backend.step(action)
            records.append({"step": obs.step, "action": action.tolist(), "state": obs.state.tolist(),
                            "tracking_error_l2": float(np.linalg.norm(backend.target - obs.state))})
            previous = action.copy()
            if done:
                break
    return {"delay_steps": delay, "hardware_connected": False, "steps": records,
            "mean_tracking_error": float(np.mean([r["tracking_error_l2"] for r in records])),
            "final_tracking_error": records[-1]["tracking_error_l2"]}


def main() -> None:
    with np.load(OBSERVATION) as sample:
        image = sample["bgr"]
    meta = load_policy_metadata(ROOT)
    config = ACTConfig.from_pretrained(CHECKPOINT)
    config.pretrained_path, config.device = CHECKPOINT, "cpu"
    policy = make_policy(config, ds_meta=meta).eval()
    pre, post = make_pre_post_processors(config, pretrained_path=str(CHECKPOINT), dataset_stats=meta.stats)
    configs = [
        {"name": "timing_baseline", "delay": 0},
        {"name": "timing_delay_2", "delay": 2},
        {"name": "action_scale_0_9", "delay": 0, "action_scale": 0.9},
        {"name": "action_scale_1_1", "delay": 0, "action_scale": 1.1},
        {"name": "observation_noise_0_05", "delay": 0, "state_noise": 0.05},
    ]
    runs = []
    for cfg in configs:
        item = run(policy, pre, post, image, cfg.get("delay", 0), cfg.get("action_scale", 1.0), cfg.get("state_noise", 0.0))
        item["name"] = cfg["name"]
        item["action_scale"] = cfg.get("action_scale", 1.0)
        item["state_noise"] = cfg.get("state_noise", 0.0)
        runs.append(item)
    result = {"policy": "ACT", "checkpoint": str(CHECKPOINT), "source_observation": str(OBSERVATION),
              "hardware_connected": False, "runs": runs}
    output = Path("artifacts/phase5-timing-mismatch.json")
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(output)
    for item in result["runs"]:
        print(item["delay_steps"], item["mean_tracking_error"], item["final_tracking_error"])


if __name__ == "__main__":
    main()
