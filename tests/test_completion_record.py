import unittest
from cross_backend.completion_record import CompletionRecord


class CompletionRecordTest(unittest.TestCase):
    def record(self, release='unknown', home='no'):
        stages = {
            'placed': {'state': 'yes', 'source': 'human'},
            'released': {'state': release, 'source': 'camera'},
            'home': {'state': home, 'source': 'joint_feedback'},
        }
        return CompletionRecord('so101', 'move pot', 'aborted', stages,
                                ('placed', 'released'), ('placed', 'released', 'home'),
                                {'source': 'a' * 64}).to_dict()

    def test_unknown_release_does_not_become_success(self):
        result = self.record()
        self.assertEqual(result['task_state'], 'unknown')
        self.assertEqual(result['full_cycle_state'], 'no')
        self.assertEqual(result['run_status'], 'aborted')

    def test_task_success_does_not_imply_full_cycle(self):
        result = self.record(release='yes', home='unknown')
        self.assertEqual(result['task_state'], 'yes')
        self.assertEqual(result['full_cycle_state'], 'unknown')

    def test_invalid_evidence_or_missing_required_phase_rejected(self):
        with self.assertRaises(ValueError):
            CompletionRecord('so101', 'move pot', 'aborted',
                {'placed': {'state': 'yes', 'source': 'human'}},
                ('placed', 'released'), ('placed',), {'source': 'a' * 64}).to_dict()
        with self.assertRaises(ValueError):
            CompletionRecord('so101', 'move pot', 'aborted',
                {'placed': {'state': 'yes', 'source': 'human'}},
                ('placed',), ('placed',), {'source': 'bad'}).to_dict()


if __name__ == '__main__':
    unittest.main()
