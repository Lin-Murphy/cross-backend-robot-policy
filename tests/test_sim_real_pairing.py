import copy
import unittest
from cross_backend.sim_real_pairing import inspect_t_pairing, make_t_template


def records():
    sim=[];real=[];number=0
    for i in range(1,11):
        for policy in ('ACT','SmolVLA'):
            for condition in ('baseline','added_observation_age'):
                number+=1
                base={'initial_condition_id':f'P{i:02d}','policy':policy,'condition':condition,
                      'status':'not_run'}
                sim.append({**base,'trial_id':f'T{number:02d}','backend':'mujoco','split':'T',
                            'initial_sha256':None,'paired_real_trial_id':number})
                real.append({**base,'trial_id':number,'initial_condition_sha256':None})
    return ({'stage':'T','synthetic_fixture':False,'real_order_source_sha256':'a'*64,'trials':sim},
            {'order_source_sha256':'a'*64,'trials':real},
            {'record_structure':'passed','slots':40,'field_ready':False})


class TPairingTests(unittest.TestCase):
    def test_unstarted_is_not_an_outcome(self):
        s,r,v=records();out=inspect_t_pairing(s,r,real_validation=v)
        self.assertEqual(out['planned_pairs'],40)
        self.assertTrue(out['order_and_trial_id_binding_verified'])
        self.assertEqual(out['pair_state_counts'],{'not_started':40})
        self.assertEqual((out['sim_started'],out['real_started']),(0,0))
        self.assertIsNone(out['outcome_statistics'])
        self.assertFalse(out['formal_stage_accepted'])

    def test_no_v_or_wrong_backend_or_duplicate_slot(self):
        s,r,v=records();s['stage']='V'
        with self.assertRaisesRegex(ValueError,'non-synthetic T'):inspect_t_pairing(s,r,real_validation=v)
        s,r,v=records();s['trials'][0]['backend']='so101'
        with self.assertRaisesRegex(ValueError,'MuJoCo/T'):inspect_t_pairing(s,r,real_validation=v)
        s,r,v=records();s['trials'][1]=copy.deepcopy(s['trials'][0])
        with self.assertRaisesRegex(ValueError,'duplicate'):inspect_t_pairing(s,r,real_validation=v)

    def test_template_is_bound_before_real_trials_start(self):
        _,r,v=records();template=make_t_template(r)
        out=inspect_t_pairing(template,r,real_validation=v)
        self.assertTrue(out['order_and_trial_id_binding_verified'])
        self.assertEqual(out['pair_state_counts'],{'not_started':40})
        r['trials'][0]['status']='running'
        with self.assertRaisesRegex(ValueError,'before real trials start'):
            make_t_template(r)

    def test_frozen_order_and_real_trial_id_binding(self):
        s,r,v=records();s['real_order_source_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'frozen order'):
            inspect_t_pairing(s,r,real_validation=v)
        s,r,v=records();s['trials'][0]['paired_real_trial_id']=2
        with self.assertRaisesRegex(ValueError,'paired real trial ID mismatch'):
            inspect_t_pairing(s,r,real_validation=v)

    def test_started_pair_requires_and_compares_frozen_initial_identity(self):
        s,r,v=records();s['trials'][0]['status']='success';r['trials'][0]['status']='failed'
        with self.assertRaisesRegex(ValueError,'missing initial identity'):
            inspect_t_pairing(s,r,real_validation=v)
        s['trials'][0]['initial_sha256']='b'*64
        r['trials'][0]['initial_condition_sha256']='c'*64
        out=inspect_t_pairing(s,r,real_validation=v)
        self.assertEqual(out['pair_state_counts']['initial_identity_mismatch'],1)
        self.assertIsNone(out['outcome_statistics'])
        r['trials'][0]['initial_condition_sha256']='b'*64
        out=inspect_t_pairing(s,r,real_validation=v)
        self.assertEqual(out['pair_state_counts']['both_started_evidence_pending'],1)
        self.assertFalse(out['formal_stage_accepted'])


if __name__=='__main__':unittest.main()
