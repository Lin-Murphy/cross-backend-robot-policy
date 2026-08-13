"""Run ACT inference on saved LeRobot observations without touching hardware."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--repeat-index", type=int, default=None)
    parser.add_argument("--mode", choices=("select_action", "predict_action_chunk"), default="select_action")
    parser.add_argument("--action-mode", choices=("select_action", "predict_action_chunk"), default="select_action")
    parser.add_argument("--output", type=Path, default=Path("artifacts/offline_replay/trace.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be at least 1")

    import av
    import pandas as pd
    from lerobot.datasets.utils import load_info, load_stats, load_tasks
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.configs.types import FeatureType, PolicyFeature

    info = load_info(args.dataset_root)
    stats = load_stats(args.dataset_root)
    tasks = load_tasks(args.dataset_root)
    data_files = sorted((args.dataset_root / "data").glob("*/*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No data parquet files found under {args.dataset_root / 'data'}")
    data = pd.concat((pd.read_parquet(path) for path in data_files), ignore_index=True)
    metadata = SimpleNamespace(features=info["features"], stats=stats, tasks=tasks)
    config_data = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    # Seeed 0.4.4 registers ACTConfig directly; its exported config contains a
    # discriminator field (`type`) that this older decoder does not accept.
    config_data.pop("type", None)
    for field_name in ("input_features", "output_features"):
        config_data[field_name] = {
            key: PolicyFeature(type=FeatureType(value["type"]), shape=tuple(value["shape"]))
            for key, value in config_data[field_name].items()
        }
    config = ACTConfig(**config_data)
    config.pretrained_path = args.checkpoint
    config.device = "cpu"
    policy = make_policy(config, ds_meta=metadata)
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=str(args.checkpoint), dataset_stats=metadata.stats
    )
    policy.eval()

    video_path = next((args.dataset_root / "videos").glob("observation.images.front/*/*.mp4"), None)
    if video_path is None:
        raise FileNotFoundError("No front camera video found")
    if args.repeat_index is not None and not 0 <= args.repeat_index < len(data):
        raise ValueError(f"--repeat-index must be between 0 and {len(data) - 1}")
    limit = args.num_samples if args.repeat_index is not None else min(args.num_samples, len(data))
    records: list[dict[str, object]] = []
    with av.open(str(video_path)) as container, torch.no_grad():
        frames = container.decode(video=0)
        repeated_frame = None
        for index in range(limit):
            source_index = args.repeat_index if args.repeat_index is not None else index
            row = data.iloc[source_index]
            if repeated_frame is None:
                repeated_frame = next(frames).to_ndarray(format="rgb24")
            frame = repeated_frame if args.repeat_index is not None else (repeated_frame if index == 0 else next(frames).to_ndarray(format="rgb24"))
            image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            observation = {
                "observation.state": torch.tensor(row["observation.state"], dtype=torch.float32),
                "observation.images.front": image,
            }
            observation["observation.state"] = observation["observation.state"].unsqueeze(0)
            observation["observation.images.front"] = observation["observation.images.front"].unsqueeze(0)
            processed = preprocessor(observation)
            start = time.perf_counter()
            if args.mode == "predict_action_chunk":
                processed_action = policy.predict_action_chunk(processed)[:, :1]
            else:
                processed_action = policy.select_action(processed)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            action = postprocessor(processed_action)
            action_tensor = action.detach().cpu()
            if not torch.isfinite(action_tensor).all():
                raise ValueError(f"non-finite action at dataset index {index}")
            records.append(
                {
                    "dataset_index": index,
                    "source_dataset_index": source_index,
                    "observation_shapes": {key: list(value.shape) for key, value in observation.items()},
                    "action_shape": list(action_tensor.shape),
                    "action_min": float(action_tensor.min()),
                    "action_max": float(action_tensor.max()),
                    "inference_latency_ms": elapsed_ms,
                    "action": action_tensor.tolist(),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "backend": "offline_replay",
                "hardware_connected": False,
                "dataset_repo_id": args.dataset_repo_id,
                "dataset_root": str(args.dataset_root),
                "checkpoint": str(args.checkpoint),
                "num_samples": limit,
                "repeat_index": args.repeat_index,
                "mode": args.mode,
                "action_mode": args.action_mode,
                "records": records,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"offline replay complete: {limit} samples -> {args.output}")


if __name__ == "__main__":
    main()
