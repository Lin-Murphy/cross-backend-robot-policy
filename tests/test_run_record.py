import unittest
from cross_backend.run_record import RunRecord


class RunRecordTest(unittest.TestCase):
    def make(self,**changes):
        args=dict(schema_version=1,backend='so101',task='move pot',policy='SmolVLA',
                  scope='saved_candidate_rejection',status='rejected',outcome=None,outcome_source=None,
                  predicted_actions=50,applied_actions=0,physical_dispatches=0,
                  task_feedback_available=False,capture_timestamps_verified=False,
                  action_clock_kind=None,evidence_sha256={'summary':'a'*64})
        args.update(changes)
        return RunRecord(**args)
    def test_rejection_record(self):self.assertEqual(self.make().to_dict()['status'],'rejected')
    def test_rejection_cannot_claim_execution(self):
        with self.assertRaises(ValueError):self.make(applied_actions=1).validate()
    def test_unknown_feedback_cannot_claim_success(self):
        with self.assertRaises(ValueError):self.make(status='success',outcome=True,outcome_source='human').validate()
    def test_bad_evidence_rejected(self):
        with self.assertRaises(ValueError):self.make(evidence_sha256={'summary':'bad'}).validate()


if __name__=='__main__':unittest.main()
