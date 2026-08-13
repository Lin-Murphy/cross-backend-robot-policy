import copy
import json
from pathlib import Path
import unittest
from cross_backend.trial_analysis import analyze_trials

ROOT=Path(__file__).resolve().parents[1]


def fixture():
    data=json.loads((ROOT/'artifacts/r1-parallel-preparation-20260924/record_templates/trials.json').read_text())
    data['synthetic_fixture']=True
    return data


def set_trial(data,initial,policy,condition,status,seconds=10):
    r=next(r for r in data['trials'] if (r['initial_condition_id'],r['policy'],r['condition'])==(initial,policy,condition))
    r.update(status=status,started_utc='2026-01-01T00:00:00+00:00',
             ended_utc=f'2026-01-01T00:00:{seconds:02d}+00:00',human_intervention=False,
             independent_success=status=='success',checkpoint_sha256=policy,initial_condition_sha256=initial,
             failure_class=None if status=='success' else status,event_log='synthetic',reviewer='synthetic',
             follower_video='synthetic',camera2_video='synthetic')
    return r


class AnalysisTests(unittest.TestCase):
    def test_all_unstarted_is_null_not_zero(self):
        result=analyze_trials(fixture());self.assertIsNone(result['overall']['success_rate_among_started'])
        self.assertEqual(result['overall']['not_run'],40)
        self.assertTrue(all(c['complete_recorded_pairs']==0 for c in result['paired_comparisons']))
    def test_rejections_abort_uncertain_in_denominator(self):
        data=fixture()
        for i,s in enumerate(('success','rejected','aborted','uncertain'),1):set_trial(data,f'P{i:02d}','ACT','baseline',s)
        result=analyze_trials(data)['groups']['ACT/baseline']
        self.assertEqual(result['started'],4);self.assertEqual(result['success_rate_among_started'],.25)
        self.assertEqual(result['successful_completion_duration']['n'],1)
    def test_paired_direction_and_success_only_times(self):
        data=fixture()
        set_trial(data,'P01','ACT','baseline','success');set_trial(data,'P01','ACT','added_observation_age','failed')
        set_trial(data,'P02','ACT','baseline','failed');set_trial(data,'P02','ACT','added_observation_age','success')
        set_trial(data,'P03','ACT','baseline','success',10);set_trial(data,'P03','ACT','added_observation_age','success',15)
        result=analyze_trials(data)['paired_comparisons'][0]
        self.assertEqual(result['complete_recorded_pairs'],3);self.assertEqual(result['paired_success_difference'],0)
        self.assertEqual(result['duration_difference_both_successful']['mean_s'],5)
        self.assertEqual(result['duration_difference_both_successful']['n'],1)
    def test_unmatched_initial_excluded(self):
        data=fixture();set_trial(data,'P01','ACT','baseline','success')
        r=set_trial(data,'P01','ACT','added_observation_age','failed');r['initial_condition_sha256']='changed'
        result=analyze_trials(data)['paired_comparisons'][0]
        self.assertEqual(result['complete_recorded_pairs'],0)
        self.assertEqual(result['excluded_pairs'][0]['reason'],'initial_identity_mismatch')
    def test_changed_checkpoint_invalidates_pair_comparison(self):
        data=fixture();set_trial(data,'P01','ACT','baseline','success')
        r=set_trial(data,'P01','ACT','added_observation_age','success');r['checkpoint_sha256']='other'
        self.assertEqual(analyze_trials(data)['paired_comparisons'][0]['complete_recorded_pairs'],0)
    def test_running_is_provisional_not_complete_pair(self):
        data=fixture();r=data['trials'][0];r.update(status='running',started_utc='2026-01-01T00:00:00Z')
        result=analyze_trials(data);self.assertTrue(result['overall']['rate_is_provisional'])
        self.assertEqual(result['overall']['started'],1);self.assertEqual(result['overall']['terminal'],0)
    def test_assisted_success_invalid(self):
        data=fixture();r=set_trial(data,'P01','ACT','baseline','success');r['human_intervention']=True
        with self.assertRaises(ValueError):analyze_trials(data)
    def test_cross_model_difference_sign(self):
        data=fixture()
        set_trial(data,'P01','ACT','baseline','success')
        set_trial(data,'P01','SmolVLA','baseline','rejected')
        result=analyze_trials(data)['paired_comparisons'][2]
        self.assertEqual(result['paired_success_difference'],-1)
        self.assertEqual(result['left_better'],1)
    def test_time_and_duplicate_validation(self):
        data=fixture();r=set_trial(data,'P01','ACT','baseline','success');r['ended_utc']='2025-01-01T00:00:00Z'
        with self.assertRaises(ValueError):analyze_trials(data)
        data=fixture();data['trials'][0]=copy.deepcopy(data['trials'][1])
        with self.assertRaises(ValueError):analyze_trials(data)

if __name__=='__main__':unittest.main()
