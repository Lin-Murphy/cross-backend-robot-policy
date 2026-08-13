#!/usr/bin/env python3
"""Read-only preflight for the independent 3D ALOHA benchmark."""

import argparse
import json
from pathlib import Path

import gymnasium as gym
import gym_aloha  # noqa: F401 - registers the environment
import imageio.v3 as iio
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    env = gym.make(
        "gym_aloha/AlohaTransferCube-v0",
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
    )
    try:
        observation, info = env.reset(seed=args.seed)
        assert env.observation_space.contains(observation)
        assert observation["pixels"]["top"].shape == (480, 640, 3)
        assert observation["agent_pos"].shape == (14,)
        assert env.action_space.shape == (14,)
        iio.imwrite(args.output / "initial.png", env.render())

        # Hold the observed joints/grippers for one step; no learned policy is used.
        action = np.asarray(observation["agent_pos"], dtype=np.float32)
        observation, reward, terminated, truncated, step_info = env.step(action)
        assert env.observation_space.contains(observation)
        result = {
            "environment": "gym_aloha/AlohaTransferCube-v0",
            "seed": args.seed,
            "observation_image_shape": list(observation["pixels"]["top"].shape),
            "observation_state_shape": list(observation["agent_pos"].shape),
            "action_shape": list(env.action_space.shape),
            "max_episode_steps": env.spec.max_episode_steps,
            "reset_info": info,
            "one_step_reward": float(reward),
            "one_step_terminated": bool(terminated),
            "one_step_truncated": bool(truncated),
            "one_step_info": step_info,
            "policy_evaluated": False,
            "success_claimed": False,
        }
        (args.output / "preflight.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
