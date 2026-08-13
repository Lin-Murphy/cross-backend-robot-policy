#!/usr/bin/env python3
"""Launch either LeRobot evaluator with the same simulation observation-age wrapper."""

import argparse
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import runpy
import sys

import gymnasium as gym
import numpy as np


class DelayedAlohaObservation(gym.Wrapper):
    def __init__(self, env: gym.Env, delay_steps: int, trace_file: Path):
        super().__init__(env)
        if delay_steps < 0:
            raise ValueError("delay_steps must be nonnegative")
        self.delay_steps = delay_steps
        self.trace_file = trace_file
        self.history = deque(maxlen=delay_steps + 1)
        self.step_index = 0
        self.seed = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.history = deque([deepcopy(observation) for _ in range(self.delay_steps + 1)], maxlen=self.delay_steps + 1)
        self.step_index = 0
        seed = kwargs.get("seed")
        self.seed = int(seed) if seed is not None else None
        self._write({"event": "reset", "seed": self.seed, "delay_steps": self.delay_steps})
        return deepcopy(self.history[0]), info

    def step(self, action):
        values = np.asarray(action, dtype=np.float64)
        if values.shape != (14,) or not np.isfinite(values).all():
            raise ValueError("ALOHA action must be a finite 14-dimensional vector")
        action_step = self.step_index
        action_source_step = max(0, action_step - self.delay_steps)
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.step_index += 1
        self.history.append(deepcopy(observation))
        source_step = max(0, self.step_index - self.delay_steps)
        self._write({
            "event": "step", "seed": self.seed, "step": self.step_index,
            "action_observation_source_step": action_source_step,
            "action_observation_age_steps": action_step - action_source_step,
            "observation_source_step": source_step,
            "observation_age_steps": self.step_index - source_step,
            "action": values.tolist(), "reward": float(reward),
            "terminated": bool(terminated), "truncated": bool(truncated),
        })
        return deepcopy(self.history[0]), reward, terminated, truncated, info

    def _write(self, record):
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)
        with self.trace_file.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, separators=(",", ":")) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay-steps", type=int, required=True)
    parser.add_argument("--trace-file", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.delay_steps < 0 or not args.command:
        parser.error("Specify nonnegative delay steps and an evaluator command after --")
    command = args.command[1:] if args.command[0] == "--" else args.command
    if not command:
        parser.error("Missing evaluator command")
    args.trace_file.parent.mkdir(parents=True, exist_ok=True)
    if args.trace_file.exists():
        raise FileExistsError(f"Trace already exists: {args.trace_file}")
    original_make = gym.make

    def make_with_delay(env_id, *positional, **keyword):
        env = original_make(env_id, *positional, **keyword)
        if env_id == "gym_aloha/AlohaTransferCube-v0":
            return DelayedAlohaObservation(env, args.delay_steps, args.trace_file)
        return env

    gym.make = make_with_delay
    sys.argv = command
    runpy.run_path(command[0], run_name="__main__")


if __name__ == "__main__":
    main()
