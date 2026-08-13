#!/usr/bin/env python3
"""Render a paired ALOHA rollout with task outcome labels, using saved videos only."""

import argparse
import json
import subprocess
from pathlib import Path


def duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True, help="Directory containing comparison.json and act/dot videos")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads((args.results / "comparison.json").read_text())
    rows = results["rows"]
    index = next((i for i, row in enumerate(rows) if row["seed"] == args.seed), None)
    if index is None or index >= 10:
        raise ValueError("Seed must be among the first 10 recorded episodes")
    act = args.results / "act" / "videos" / "aloha_0" / f"eval_episode_{index}.mp4"
    dot = args.results / "dot" / "videos" / f"eval_episode_{index}.mp4"
    if not act.is_file() or not dot.is_file():
        raise FileNotFoundError("Both saved videos are required")
    target_duration = max(duration(act), duration(dot))
    labels = []
    for name, video in (("ACT", act), ("DOT", dot)):
        row = rows[index]
        reward = row[f"{name.lower()}_max_reward"]
        status = "SUCCESS" if reward >= 4 else "FAIL"
        label = f"{name} | seed {args.seed} | reward {reward:g} | {status}"
        pad_seconds = max(0.0, target_duration - duration(video))
        labels.append(
            f"tpad=stop_mode=clone:stop_duration={pad_seconds:.3f},"
            f"drawtext=text='{label}':fontcolor=white:fontsize=21:x=12:y=12:"
            "box=1:boxcolor=black@0.78:boxborderw=7"
        )
    filter_graph = f"[0:v]{labels[0]}[a];[1:v]{labels[1]}[b];[a][b]hstack=inputs=2[v]"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(act), "-i", str(dot), "-filter_complex", filter_graph,
         "-map", "[v]", "-t", f"{target_duration:.3f}", "-r", "50", "-c:v", "libx264", "-crf", "22",
         "-preset", "veryfast", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output)],
        check=True,
    )
    subprocess.run(["ffmpeg", "-v", "error", "-i", str(args.output), "-f", "null", "-"], check=True)
    print(args.output)


if __name__ == "__main__":
    main()
