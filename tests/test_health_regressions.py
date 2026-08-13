import copy
import json
from pathlib import Path
import tempfile
import unittest

from cross_backend.evaluator import evaluate_trace
from cross_backend.trial_analysis import analyze_trials
from scripts.move_pot_trial_records import verify_trial_evidence, initial_condition_sha256, sha


class EvidenceTests(unittest.TestCase):
    def test_file_integrity_and_frozen_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            initial=[{'id':'P01','robot_start_state':[0]*6}]
            row={'trial_id':'t','status':'success','policy':'ACT','initial_condition_id':'P01',
                 'checkpoint_sha256':'a'*64,'initial_condition_sha256':initial_condition_sha256(initial[0])}
            for key in ('event_log','follower_video','camera2_video'):
                p=out/key;p.write_bytes(b'fixture evidence')
                row[key]=key;row[key+'_sha256']=sha(p)
            lock={'act_checkpoint_sha256':'a'*64}
            def check(r=row, ready=True):
                return verify_trial_evidence(out,[r],initial,lock,ready)
            self.assertEqual(check()['verified_trial_ids'],['t'])
            self.assertFalse(check(ready=False)['verified_trial_ids'])
            bad=copy.deepcopy(row);bad['checkpoint_sha256']='b'*64
            self.assertIn('checkpoint_does_not_match_execution_lock',check(bad)['trial_evidence_issues']['t'])
            bad=copy.deepcopy(row);bad['initial_condition_sha256']='not-a-hash'
            self.assertFalse(check(bad)['verified_trial_ids'])
            (out/'event_log').write_bytes(b'changed')
            self.assertIn('event_log_hash_missing_or_mismatch',check()['trial_evidence_issues']['t'])
            (out/'camera2_video').unlink()
            self.assertIn('camera2_video_missing_or_empty',check()['trial_evidence_issues']['t'])

    def test_unverified_real_records_cannot_be_verified_pairs(self):
        from test_trial_analysis import fixture, set_trial
        record=fixture();record['synthetic_fixture']=False
        set_trial(record,'P01','ACT','baseline','success')
        set_trial(record,'P01','ACT','added_observation_age','failed')
        result=analyze_trials(record)
        self.assertEqual(result['overall']['started'],2)
        self.assertEqual(result['paired_comparisons'][0]['complete_verified_pairs'],0)
        self.assertEqual(result['paired_comparisons'][0]['excluded_pairs'][0]['reason'],'trial_evidence_not_verified')
        record['synthetic_fixture']=True
        result=analyze_trials(record)
        self.assertEqual(result['paired_comparisons'][0]['complete_recorded_pairs'],1)
        self.assertEqual(result['paired_comparisons'][0]['complete_verified_pairs'],0)


class TraceTests(unittest.TestCase):
    def check_trace(self,step):
        return evaluate_trace({'hardware_connected':False,'steps':[step]},1)

    def test_good_trace_and_reject_invalid_payloads(self):
        good={'action':[1,2],'state':[0,1],'tracking_error_l2':1.4}
        self.assertTrue(self.check_trace(good)['passed'])
        for bad in ({'tracking_error_l2':float('nan')},
                    {**good,'action':[]},{**good,'action':[1]},
                    {**good,'state':[[0,1]]},{**good,'action':[float('inf'),2]},
                    {**good,'tracking_error_l2':float('nan')},
                    {**good,'tracking_error_l2':-1}):
            result=self.check_trace(bad)
            self.assertFalse(result['passed'])
            json.dumps(result,allow_nan=False)
        self.assertFalse(evaluate_trace({'hardware_connected':False,'steps':[]},0)['passed'])

if __name__=='__main__':unittest.main()
