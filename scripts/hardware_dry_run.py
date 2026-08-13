from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from cross_backend.safety import NoDispatch, validate_action


def main() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors

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
    batch = preprocessor(
        {
            "observation.state": sample["observation.state"].unsqueeze(0),
            "observation.images.front": sample["observation.images.front"].unsqueeze(0),
        }
    )
    with torch.no_grad():
        action = postprocessor(policy.predict_action_chunk(batch)[:, :1])
    action_np = validate_action(action.squeeze(0).squeeze(0).cpu().numpy())

    dispatch = NoDispatch()
    blocked = False
    block_reason = None
    try:
        dispatch.send(action_np)
    except RuntimeError as error:
        blocked = True
        block_reason = str(error)

    report = {
        "mode": "hardware_dry_run",
        "hardware_connected": False,
        "serial_port_opened": False,
        "camera_opened": False,
        "dispatch_enabled": False,
        "dispatch_blocked": blocked,
        "block_reason": block_reason,
        "action_shape": list(action_np.shape),
        "action_finite": bool(np.isfinite(action_np).all()),
        "action": action_np.tolist(),
    }
    output = Path("artifacts/safety/hardware_dry_run.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not blocked:
        raise RuntimeError("dry-run failed: action dispatch was not blocked")
    print(f"hardware dry-run passed: dispatch blocked -> {output}")


if __name__ == "__main__":
    main()
