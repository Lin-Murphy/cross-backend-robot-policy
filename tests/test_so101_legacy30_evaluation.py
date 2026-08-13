import unittest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from scripts.evaluate_so101_legacy30_pilot_record import build_record, main


class LegacyEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.run={'scope':'single_legacy30_development_pilot_not_formal','formal_trial_count':0,
                  'status':'gate_rejected','raw_goal_packets_transmitted':1}
        self.events=[{'event':'legacy_goal_candidate'},{'event':'legacy_goal_packet_transmitted'},
                     {'event':'legacy_goal_candidate'}]
        self.evidence={'summary':'a'*64}
    def test_rejection_after_packet_is_aborted_unknown_outcome(self):
        record=build_record(self.run,self.events,None,None,self.evidence)
        self.assertEqual(record['status'],'aborted')
        self.assertEqual(record['physical_dispatches'],1)
        self.assertIsNone(record['outcome'])
    def test_operator_success_during_guard_aborted_run_keeps_aborted_status(self):
        record=build_record(self.run,self.events,'success','operator_on_site',self.evidence)
        self.assertEqual(record['status'],'aborted')
        self.assertTrue(record['outcome'])
        self.assertTrue(record['task_feedback_available'])
        self.assertEqual(record['physical_dispatches'],1)

    def test_act_policy_identity_is_preserved(self):
        self.run['policy']='act'
        record=build_record(self.run,self.events,None,None,self.evidence)
        self.assertEqual(record['policy'],'ACT')

    def test_no_packet_rejection(self):
        self.run['raw_goal_packets_transmitted']=0
        record=build_record(self.run,[self.events[0]],None,None,self.evidence)
        self.assertEqual(record['status'],'rejected')
        with self.assertRaises(ValueError):
            build_record(self.run,[self.events[0]],'success','operator_on_site',self.evidence)
    def test_completed_run_requires_operator_verdict(self):
        self.run['status']='ended_unreviewed'
        with self.assertRaises(ValueError):build_record(self.run,self.events,None,None,self.evidence)
        record=build_record(self.run,self.events,'success','operator_on_site',self.evidence)
        self.assertTrue(record['outcome'])
    def test_shared_live_events_must_match_raw_transport_count(self):
        with TemporaryDirectory() as temp:
            root=Path(temp);run=root/'run';run.mkdir()
            summary={**self.run,'shared_boundary_enabled':True}
            (run/'summary.json').write_text(json.dumps(summary))
            (run/'profile.json').write_text('{}')
            events=[{'event':'legacy_goal_candidate','reasons':[]},
                    {'event':'legacy_goal_packet_transmitted','raw':[2000]*6,
                     'transport_returned':True,'per_motor_acknowledged':False,'host_ns':100},
                    {'event':'shared_backend_observation','observation_id':'frame:1'},
                    {'event':'shared_backend_action_receipt','receipt':
                        {'accepted':True,'physical_dispatches':1,'ack_scope':'sync_transport_return'}}]
            path=run/'hardware-events.jsonl'
            path.write_text('\n'.join(json.dumps(e) for e in events)+'\n')
            main(run,root/'good',None,None)
            self.assertTrue((root/'good'/'evaluation.json').exists())
            path.write_text('\n'.join(json.dumps(e) for e in events[:-1])+'\n')
            with self.assertRaisesRegex(ValueError,'shared live boundary'):
                main(run,root/'missing',None,None)

    def test_packet_count_mismatch_rejected(self):
        self.run['raw_goal_packets_transmitted']=2
        with self.assertRaises(ValueError):build_record(self.run,self.events,None,None,self.evidence)

if __name__=='__main__':unittest.main()
