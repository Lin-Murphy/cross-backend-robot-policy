#!/usr/bin/env python3
"""Audit paired ALOHA observation-age trials and summarize per-seed changes."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np


def read_trial(directory: Path, model: str, delay: int):
    comparison = json.loads((directory / "comparison.json").read_text())
    rows = comparison["rows"]
    trace_file = directory / f"{model}.observation-age.jsonl"
    expected = {row["seed"]: row for row in rows}
    seeds = []
    rewards = {}
    ages = Counter()
    current_seed = None
    inactive_autoreset = False
    for line in trace_file.read_text().splitlines():
        record = json.loads(line)
        if record["event"] == "reset":
            if record["delay_steps"] != delay:
                raise ValueError("Delay configuration differs from the trial label")
            if record["seed"] is None:
                inactive_autoreset = True
                current_seed = None
                continue
            current_seed = int(record["seed"])
            seeds.append(current_seed)
            rewards[current_seed] = []
            inactive_autoreset = False
            continue
        if inactive_autoreset or current_seed is None:
            raise ValueError("Step after an unseeded autoreset or before a trial reset")
        step = record["step"]
        action_age = min(delay, step - 1)
        output_age = min(delay, step)
        if record["seed"] != current_seed or record["action_observation_age_steps"] != action_age:
            raise ValueError("Action used a different observation age or trial seed")
        if record["action_observation_source_step"] != step - 1 - action_age:
            raise ValueError("Action observation source step is inconsistent")
        if record["observation_age_steps"] != output_age:
            raise ValueError("Returned observation age is inconsistent")
        if record["observation_source_step"] != step - output_age:
            raise ValueError("Returned observation source step is inconsistent")
        action = record["action"]
        if len(action) != 14 or any(not math.isfinite(value) for value in action):
            raise ValueError("Invalid action vector")
        rewards[current_seed].append(float(record["reward"]))
        ages[action_age] += 1
    if seeds != list(expected):
        raise ValueError("Seed order or number of trials differs from comparison result")
    for seed, values in rewards.items():
        if not values or max(values) != expected[seed][f"{model}_max_reward"]:
            raise ValueError(f"Trace reward does not match evaluator result at seed {seed}")
    return rows, {"steps": sum(ages.values()), "action_ages": dict(sorted(ages.items()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--delayed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"task": "gym_aloha/AlohaTransferCube-v0", "delay_steps": [0, 3], "criterion": "max reward >= 4", "models": {}, "per_seed": []}
    base_rows, delay_rows = None, None
    for model in ("act", "dot"):
        base_rows, base_audit = read_trial(args.baseline, model, 0)
        delay_rows, delayed_audit = read_trial(args.delayed, model, 3)
        if [r["seed"] for r in base_rows] != [r["seed"] for r in delay_rows]:
            raise ValueError("Baseline and delayed seed lists differ")
        base_success = [r[f"{model}_success_reward4"] for r in base_rows]
        delay_success = [r[f"{model}_success_reward4"] for r in delay_rows]
        transitions = {
            "both_success": sum(a and b for a, b in zip(base_success, delay_success)),
            "lost_success": sum(a and not b for a, b in zip(base_success, delay_success)),
            "gained_success": sum(not a and b for a, b in zip(base_success, delay_success)),
            "both_fail": sum(not a and not b for a, b in zip(base_success, delay_success)),
        }
        n = len(base_rows)
        result["models"][model] = {
            "n": n,
            "baseline_success": sum(base_success),
            "delayed_success": sum(delay_success),
            "change_percentage_points": round((sum(delay_success) - sum(base_success)) * 100 / n, 3),
            "baseline_mean_max_reward": round(sum(r[f"{model}_max_reward"] for r in base_rows) / n, 4),
            "delayed_mean_max_reward": round(sum(r[f"{model}_max_reward"] for r in delay_rows) / n, 4),
            "transitions": transitions,
            "baseline_trace": base_audit,
            "delayed_trace": delayed_audit,
        }
    result["per_seed"] = [
        {"seed": a["seed"], **{f"{model}_{condition}_success": rows[i][f"{model}_success_reward4"]
          for model in ("act", "dot")
          for condition, rows in (("baseline", base_rows), ("delayed", delay_rows))}}
        for i, a in enumerate(base_rows)
    ]
    # The per-seed table above uses the shared result rows, which contain both models.
    result["difference_in_change_percentage_points"] = round(
        result["models"]["dot"]["change_percentage_points"] - result["models"]["act"]["change_percentage_points"], 3
    )
    changes = np.array([
        [float(b[f"{model}_success_reward4"]) - float(a[f"{model}_success_reward4"])
         for model in ("act", "dot")]
        for a, b in zip(base_rows, delay_rows)
    ])
    rng = np.random.default_rng(20260926)
    draws = changes[rng.integers(0, len(changes), size=(20000, len(changes)))].mean(axis=1) * 100
    result["paired_seed_bootstrap_95pct"] = {
        "resamples": 20000, "random_seed": 20260926,
        "act_change_pp": np.quantile(draws[:, 0], [0.025, 0.975]).tolist(),
        "dot_change_pp": np.quantile(draws[:, 1], [0.025, 0.975]).tolist(),
        "dot_minus_act_change_pp": np.quantile(draws[:, 1] - draws[:, 0], [0.025, 0.975]).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print({model: (item["baseline_success"], item["delayed_success"], item["transitions"]) for model, item in result["models"].items()})


if __name__ == "__main__":
    main()
