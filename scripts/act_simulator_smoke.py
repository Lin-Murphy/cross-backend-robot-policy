from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch

from cross_backend.simulator import TrackingSimulator
from cross_backend.safety import NoDispatch, validate_action


def main() -> None:
    import av
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

    parser = argparse.ArgumentParser()
    parser.add_argument("--action-delay-steps", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/simulator/act_smoke_trace.json"))
    args = parser.parse_args()
    dataset_root = Path(r"C:\Users\Murphy\.cache\huggingface\lerobot\Murphy-Lin\lerobot101_dataset_a")
    checkpoint = Path(r"D:\project\lerobot-seeed\outputs\train\act_test\checkpoints\000020\pretrained_model")
    dataset = LeRobotDataset("Murphy-Lin/lerobot101_dataset_a", root=dataset_root)
    config = ACTConfig.from_pretrained(checkpoint)
    config.pretrained_path = checkpoint
    config.device = "cpu"
    policy = make_policy(config, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=str(checkpoint), dataset_stats=dataset.meta.stats
    )
    policy.eval()

    sample = dataset[0]
    state_image = sample["observation.images.front"]
    backend = TrackingSimulator(max_steps=20, action_delay_steps=args.action_delay_steps)
    dispatch = NoDispatch()
    observation = backend.reset()
    trace = []
    with torch.no_grad():
        for _ in range(backend.max_steps):
            batch = preprocessor(
                {
                    "observation.state": torch.from_numpy(observation.state).unsqueeze(0),
                    "observation.images.front": state_image.unsqueeze(0),
                }
            )
            action = postprocessor(policy.predict_action_chunk(batch)[:, :1])
            action_np = action.squeeze(0).squeeze(0).cpu().numpy()
            previous_action = trace[-1]["action"] if trace else None
            action_np = validate_action(action_np, np.asarray(previous_action, dtype=np.float32) if previous_action else None)
            observation, done = backend.step(action_np)
            trace.append(
                {
                    "step": observation.step,
                    "action": action_np.tolist(),
                    "state": observation.state.tolist(),
                    "tracking_error_l2": float(np.linalg.norm(backend.target - observation.state)),
                    "action_norm": float(np.linalg.norm(action_np)),
                    "action_delay_steps": args.action_delay_steps,
                    "dispatch_enabled": False,
                    "done": done,
                }
            )

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "backend": "tracking_simulator",
                "policy": "ACT",
                "lerobot_version": "0.6.1",
                "hardware_connected": False,
                "steps": trace,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"ACT simulator smoke complete: {len(trace)} steps -> {output}")


if __name__ == "__main__":
    main()
