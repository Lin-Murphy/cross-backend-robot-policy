import json
import tempfile
import unittest
from pathlib import Path

from cross_backend.completion_record import CompletionRecord
from scripts.compare_completion_records import compare


class CompareCompletionRecordsTest(unittest.TestCase):
    def record(self, backend, task='move pot', release='yes'):
        return CompletionRecord(backend, task, 'success', {
            'object_on_target': {'state': 'yes', 'source': 'post_run_observation'},
            'object_released': {'state': release, 'source': 'task_evaluator'},
            'returned_to_start': {'state': 'unknown', 'source': 'not_evaluated'},
        }, ('object_on_target', 'object_released'),
            ('object_on_target', 'object_released', 'returned_to_start'),
            {'source': 'a' * 64}).to_dict()

    def test_same_task_rule_compares_success_without_claiming_paired_performance(self):
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp) / 'left.json', Path(tmp) / 'right.json'
            left.write_text(json.dumps(self.record('so101')))
            right.write_text(json.dumps(self.record('mujoco')))
            result = compare(left, right)
            self.assertEqual(result['task_states'], {'left': 'yes', 'right': 'yes'})
            self.assertEqual(result['full_cycle_states'], {'left': 'unknown', 'right': 'unknown'})
            self.assertFalse(result['formal_performance_comparison'])
            self.assertFalse(result['paired_initial_state_verified'])

    def test_rejects_changed_task_and_tampered_derived_outcome(self):
        with tempfile.TemporaryDirectory() as tmp:
            left, right = Path(tmp) / 'left.json', Path(tmp) / 'right.json'
            left.write_text(json.dumps(self.record('so101')))
            right.write_text(json.dumps(self.record('mujoco', task='other task')))
            with self.assertRaisesRegex(ValueError, 'incompatible task'):
                compare(left, right)
            changed = self.record('mujoco', release='unknown')
            changed['task_state'] = 'yes'
            right.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, 'derived milestones'):
                compare(left, right)


if __name__ == '__main__':
    unittest.main()
