#!/usr/bin/env python3
"""Run the portable core regression suite with only NumPy installed."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'src', ROOT/'scripts', ROOT/'tests'):
    sys.path.insert(0, str(path))

MODULES = [
    'test_execution_contract', 'test_execution_session', 'test_aloha_execution',
    'test_so101_lerobot_backend_adapter', 'test_so101_shared_action_hook',
    'test_evaluation', 'test_preflight', 'test_adapter_template',
]

if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(MODULES))
    raise SystemExit(not result.wasSuccessful())
