#!/usr/bin/env python3
"""Run a native evaluator through the common per-action execution session."""
import argparse
import json
from pathlib import Path
import runpy
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import gymnasium as gym
from cross_backend.aloha_execution import AlohaBackendAdapter
from cross_backend.execution_contract import ActionRequest
from cross_backend.execution_session import ExecutionSession


class SharedAlohaExecution(gym.Wrapper):
    def __init__(self, env, record):
        super().__init__(env)
        self.record = record
        self.session = None
        self.episode = 0

    def reset(self, **kwargs):
        if self.session is not None:
            self.session.stop()
        observation, info = self.env.reset(**kwargs)
        self.episode += 1
        self.adapter = AlohaBackendAdapter(self.env, observation, self.episode)
        self.session = ExecutionSession(self.adapter, self.record)
        return observation, info

    def step(self, action):
        if self.session is None:
            raise RuntimeError('reset required before execution')
        self.session.step(lambda frame: ActionRequest(frame.observation_id,
            frame.joint_names, frame.joint_units, tuple(float(v) for v in action)))
        transition = self.adapter.transition
        if transition[2] or transition[3]:
            self.session.stop()
        return transition

    def close(self):
        try:
            if self.session is not None:
                self.session.stop()
        finally:
            self.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace-file', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('native evaluator command required')
    target = shutil.which(command[0]) or command[0]
    args.trace_file.parent.mkdir(parents=True, exist_ok=True)
    original = gym.make
    with args.trace_file.open('x') as stream:
        def record(event):
            stream.write(json.dumps(event) + '\n')
            stream.flush()
        def make(env_id, *positional, **keyword):
            env = original(env_id, *positional, **keyword)
            return SharedAlohaExecution(env, record) if env_id == 'gym_aloha/AlohaTransferCube-v0' else env
        gym.make = make
        try:
            sys.argv = [target, *command[1:]]
            runpy.run_path(target, run_name='__main__')
        finally:
            gym.make = original


if __name__ == '__main__':
    main()
