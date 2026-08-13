import unittest
from scripts.evaluate_so101_bounded_pilot_record import build_record


class EvaluationTest(unittest.TestCase):
    def base(self,status,applied):
        return {'scope':'single_bounded_development_pilot_not_formal',
                'formal_trial_count':0,'status':status,'applied_actions':applied}
    def test_operator_success(self):
        self.assertEqual(build_record(self.base('ended_unreviewed',100),'success','operator'),
                         ('success',True,'operator',True))
    def test_uncertain_keeps_null_outcome(self):
        self.assertEqual(build_record(self.base('ended_unreviewed',100),'uncertain','operator')[1],None)
    def test_rejection_after_action_is_aborted(self):
        self.assertEqual(build_record(self.base('gate_rejected',3),None,None)[0],'aborted')
    def test_zero_action_rejection_stays_rejected(self):
        self.assertEqual(build_record(self.base('gate_rejected',0),None,None)[0],'rejected')
    def test_aborted_cannot_be_called_success(self):
        with self.assertRaises(ValueError):build_record(self.base('aborted',3),'success','operator')


if __name__=='__main__':unittest.main()
