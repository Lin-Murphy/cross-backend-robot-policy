#!/usr/bin/env python3
"""Reconcile two LeRobot generations using ALOHA's shared max-reward criterion."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def video_metadata(path: Path) -> dict:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=nb_frames,width,height", "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    )
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], check=True, capture_output=True, text=True)
    streams = json.loads(probe.stdout)["streams"]
    if len(streams) != 1 or int(streams[0].get("nb_frames", 0)) < 2:
        raise ValueError(f"Invalid video stream: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path), **streams[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--act", type=Path, required=True)
    parser.add_argument("--dot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-seed", type=int, default=11)
    args = parser.parse_args()

    act = json.loads((args.act / "eval_info.json").read_text())
    dot = json.loads((args.dot / "eval_info.json").read_text())
    act_task = act["per_task"][0]["metrics"]
    act_rewards = act_task["max_rewards"]
    dot_episodes = dot["per_episode"]
    if len(act_rewards) != len(dot_episodes):
        raise ValueError("Episode counts differ")
    if any(ep["seed"] != args.first_seed + i for i, ep in enumerate(dot_episodes)):
        raise ValueError("DOT seeds do not match requested sequence")

    n = len(act_rewards)
    rows = []
    for i in range(n):
        act_reward = float(act_rewards[i])
        dot_reward = float(dot_episodes[i]["max_reward"])
        rows.append({
            "seed": args.first_seed + i,
            "act_max_reward": act_reward,
            "act_success_reward4": act_reward >= 4,
            "act_native_success": bool(act_task["successes"][i]),
            "dot_max_reward": dot_reward,
            "dot_success_reward4": dot_reward >= 4,
            "dot_native_success": bool(dot_episodes[i]["success"]),
        })
    videos = {
        "act": [video_metadata(args.act / "videos" / "aloha_0" / f"eval_episode_{i}.mp4") for i in range(min(n, 10))],
        "dot": [video_metadata(args.dot / "videos" / f"eval_episode_{i}.mp4") for i in range(min(n, 10))],
    }
    result = {
        "task": "gym_aloha/AlohaTransferCube-v0",
        "criterion": "episode maximum reward >= 4",
        "episode_length": 400,
        "seeds": [args.first_seed, args.first_seed + n - 1],
        "act_successes": sum(row["act_success_reward4"] for row in rows),
        "dot_successes": sum(row["dot_success_reward4"] for row in rows),
        "n_episodes_each": n,
        "rows": rows,
        "videos": videos,
        "act_native_success_pct": act["overall"]["pc_success"],
        "dot_native_success_pct": dot["aggregated"]["pc_success"],
        "note": "DOT legacy evaluator reports false success despite reward 4; shared reward criterion follows gym-aloha task documentation.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"ACT {result['act_successes']}/{n}; DOT {result['dot_successes']}/{n}; verified {sum(map(len, videos.values()))} videos")


if __name__ == "__main__":
    main()
