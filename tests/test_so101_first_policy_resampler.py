import time
import unittest
from types import SimpleNamespace

import torch

from scripts.run_so101_legacy30_audited_pilot import install_first_policy_sample_resampler, load_profile
from cross_backend.legacy_so101_sync_guard import LegacySyncGuard, LegacyGuardRejected, NAMES


CURRENT = (1973, 1207, 2373, 2953, 1909, 2221)
LARGE = (1976, 1491, 1940, 3144, 1923, 2236)
SMALL = (1975, 1246, 2271, 2958, 1905, 2231)


class FakeBus:
    motors = {name: SimpleNamespace(id=i) for i, name in enumerate(NAMES, 1)}

    def _unnormalize(self, values):
        return {key: int(value) for key, value in values.items()}


class FakeEngine:
    def __init__(self, samples):
        self.samples = list(samples)
        self.reset_count = 0
        self.calls = 0

    def get_action(self, _frame):
        self.calls += 1
        return torch.tensor(self.samples.pop(0), dtype=torch.float32)

    def reset(self):
        self.reset_count += 1


class FirstPolicySampleResamplerTest(unittest.TestCase):
    def setUp(self):
        _, limits = load_profile()
        self.events = []
        self.guard = LegacySyncGuard(limits, self.events.append)
        self.guard.observe(dict(zip(NAMES, CURRENT)), observed_ns=time.perf_counter_ns())
        self.robot = SimpleNamespace(bus=FakeBus())

    def test_rejects_large_draw_without_dispatch_then_returns_bounded_draw(self):
        engine = FakeEngine([LARGE, SMALL])
        restore = install_first_policy_sample_resampler(
            engine, self.robot, self.guard, self.events.append, 4)
        try:
            action = engine.get_action({})
            self.assertEqual(tuple(int(v) for v in action), SMALL)
            self.assertEqual(engine.calls, 2)
            self.assertEqual(engine.reset_count, 1)
            self.assertEqual(self.guard.goal_packets, 0)
            screens = [event for event in self.events
                       if event['event'] == 'shared_first_policy_sample_screen']
            self.assertEqual(len(screens), 2)
            self.assertEqual(screens[0]['reasons'], [
                'target_from_feedback_exceeded:shoulder_lift',
                'target_from_feedback_exceeded:elbow_flex',
                'target_from_feedback_exceeded:wrist_flex'])
            self.assertEqual(screens[1]['reasons'], [])
        finally:
            restore()

    def test_four_bad_draws_stop_without_dispatch(self):
        engine = FakeEngine([LARGE] * 4)
        restore = install_first_policy_sample_resampler(
            engine, self.robot, self.guard, self.events.append, 4)
        try:
            with self.assertRaisesRegex(LegacyGuardRejected,
                                        'first_policy_samples_outside_trial_profile'):
                engine.get_action({})
            self.assertEqual(engine.calls, 4)
            self.assertEqual(engine.reset_count, 3)
            self.assertEqual(self.guard.goal_packets, 0)
            self.assertTrue(self.guard.latched)
        finally:
            restore()


if __name__ == '__main__':
    unittest.main()
