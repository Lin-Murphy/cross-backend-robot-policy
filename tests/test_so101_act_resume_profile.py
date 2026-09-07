import unittest
from unittest.mock import patch

from cross_backend.legacy_so101_sync_guard import LegacySyncGuard, NAMES
from scripts import run_so101_legacy30_audited_pilot as runner


class ACTResumeProfileTest(unittest.TestCase):
    def setUp(self):
        with patch.object(runner, "PROFILE", runner.ACT_PROFILE):
            _, self.profile = runner.load_profile()

    def reasons(self, feedback, target):
        events = []
        guard = LegacySyncGuard(self.profile, events.append, clock=lambda: 1_020_000_000)
        guard.observe(dict(zip(NAMES, feedback)), observed_ns=1_000_000_000)
        return guard._goal_reasons(1_020_000_000, 42, 2, dict(enumerate(target, 1)))[0]

    def test_current_first_act_target_is_within_reviewed_envelope(self):
        self.assertEqual(
            self.reasons((2022, 1675, 2352, 2692, 1833, 2267),
                         (2007, 1704, 2413, 2604, 1797, 2299)), [])

    def test_prior_chunk_boundary_candidate_is_not_rejected_for_three_ticks(self):
        self.assertEqual(
            self.reasons((2022, 1673, 2350, 2693, 1833, 2267),
                         (2002, 1699, 2449, 2579, 1796, 2298)), [])

    def test_new_limits_still_reject_excess_gap_and_range(self):
        feedback = (2022, 1675, 2352, 2692, 1833, 2267)
        self.assertIn("target_from_feedback_exceeded:wrist_flex",
                      self.reasons(feedback, (2007, 1704, 2413, 2561, 1797, 2299)))
        self.assertIn("target_outside_trial_profile:wrist_roll",
                      self.reasons(feedback, (2007, 1704, 2413, 2604, 1749, 2299)))


if __name__ == "__main__":
    unittest.main()
