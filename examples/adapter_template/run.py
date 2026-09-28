#!/usr/bin/env python3
"""Run the standalone adapter example without touching hardware."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from adapters import MemoryBackend, ExamplePolicy
from cross_backend.execution_session import ExecutionSession


def main():
    backend = MemoryBackend()
    with ExecutionSession(backend, lambda event: print(json.dumps(event))) as session:
        for _ in range(3):
            session.step(ExamplePolicy(backend.capabilities))
    print(json.dumps({'example_only': True, 'steps': backend.index,
                      'position': backend.position, 'stopped': backend.stopped}))


if __name__ == '__main__':
    main()
