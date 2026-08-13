import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from cross_backend.sim_study_analysis import analyze_v_stage, wilson_95
from cross_backend.sim_trial_initial import initial_geometry_sha256


def fixture(synthetic=True):
    rows=[]
    for i in range(1,11):
        for policy in ('ACT','SmolVLA'):
            for condition in ('baseline','added_observation_age'):
                success=(i<=5 if policy=='ACT' and condition=='baseline' else
                         i<=2 if policy=='ACT' else
                         i<=3 if condition=='baseline' else i<=1)
                rows.append({'trial_id':f'V{i:02d}-{policy}-{condition}',
                             'initial_condition_id':f'V{i:02d}', 'policy':policy,
                             'condition':condition, 'backend':'mujoco','split':'V',
                             'status':'success' if success else 'timeout',
                             'initial_sha256':f'{i:064x}', 'scene_sha256':'a'*64,
                             'protocol_sha256':'b'*64,
                             'checkpoint_sha256':('c' if policy=='ACT' else 'd')*64})
    return {'stage':'V','synthetic_fixture':synthetic,'trials':rows}


def add_fake_evidence(record, root):
    for row in record['trials']:
        index=int(row['initial_condition_id'][1:])
        initial={'schema_version':1,'backend':'mujoco','split':'V',
                 'initial_condition_id':row['initial_condition_id'],
                 'scene_sha256':row['scene_sha256'],'joint_unit':'sim_radian_unaligned',
                 'robot_q_rad':[index*.01,0,0,0,0,0],
                 'tape_xy_m':[-.18,-.15],'mat_xy_m':[-.18,.08],
                 'tape_quaternion_wxyz':[1,0,0,0]}
        row['initial_sha256']=initial_geometry_sha256(initial)
        base=Path(root)/row['trial_id'];base.mkdir()
        files={'summary':json.dumps({**row,'formal_trial':True,'physical_alignment_verified':True}).encode(),'events':json.dumps({'trial_id':row['trial_id'],'event':'started'}).encode()+b'\n',
               'video':b'\x00\x00\x00\x18ftypisomsynthetic-test-only',
               'initial':json.dumps(initial).encode()}
        row['evidence']={}
        for kind,data in files.items():
            path=base/({'summary':'summary.json','events':'events.jsonl','video':'dual.mp4','initial':'initial.json'}[kind]);path.write_bytes(data)
            row['evidence'][kind]={'path':str(path.relative_to(root)),'sha256':hashlib.sha256(data).hexdigest()}


class VStudyAnalysisTests(unittest.TestCase):
    def test_wilson_known_half_success_and_empty(self):
        lo,hi=wilson_95(5,10)
        self.assertAlmostEqual(lo,.2365930905,places=8)
        self.assertAlmostEqual(hi,.7634069095,places=8)
        self.assertIsNone(wilson_95(0,0))

    def test_block_contrasts_and_reproducible_bootstrap(self):
        result=analyze_v_stage(fixture(),bootstrap_samples=500,seed=37)
        self.assertFalse(result['full_matrix_file_verified'])
        self.assertFalse(result['formal_stage_accepted'])
        self.assertEqual(result['complete_blocks'],10)
        self.assertEqual(result['groups']['ACT/baseline']['success_rate'],.5)
        self.assertAlmostEqual(result['paired_contrasts']['ACT_delay_minus_baseline']['estimate'],-.3)
        self.assertAlmostEqual(result['paired_contrasts']['SmolVLA_delay_minus_baseline']['estimate'],-.2)
        self.assertAlmostEqual(result['paired_contrasts']['difference_of_delay_effects']['estimate'],.1)
        self.assertEqual(result['paired_contrasts'],analyze_v_stage(fixture(),bootstrap_samples=500,seed=37)['paired_contrasts'])
        self.assertIn('SYNTHETIC',result['evidence_label'])

    def test_file_integrity_gate_withholds_unverified_rates(self):
        record=fixture(False)
        result=analyze_v_stage(record)
        self.assertFalse(result['full_matrix_file_verified'])
        self.assertIsNone(result['groups']['ACT/baseline']['success_rate'])
        self.assertIsNone(result['paired_contrasts'])
        self.assertEqual(result['complete_blocks'],0)
        with tempfile.TemporaryDirectory() as tmp:
            add_fake_evidence(record,tmp)
            verified=analyze_v_stage(record,evidence_root=tmp,bootstrap_samples=200)
            self.assertTrue(verified['full_matrix_file_verified'])
            self.assertFalse(verified['formal_stage_accepted'])
            self.assertEqual(verified['groups']['ACT/baseline']['success_rate'],.5)
            initial=Path(tmp)/record['trials'][0]['evidence']['initial']['path']
            original_initial=initial.read_bytes();initial.write_bytes(original_initial+b'changed')
            initial_rejected=analyze_v_stage(record,evidence_root=tmp)
            self.assertIsNone(initial_rejected['groups']['ACT/baseline']['success_rate'])
            self.assertEqual(initial_rejected['evidence_failures'][record['trials'][0]['trial_id']],'initial_hash_mismatch')
            initial.write_bytes(original_initial)
            video=Path(tmp)/record['trials'][0]['evidence']['video']['path'];video.write_bytes(b'changed')
            rejected=analyze_v_stage(record,evidence_root=tmp)
            self.assertFalse(rejected['full_matrix_file_verified'])
            self.assertIsNone(rejected['groups']['ACT/baseline']['success_rate'])
            self.assertEqual(rejected['evidence_failures'][record['trials'][0]['trial_id']],'video_hash_mismatch')

    def test_missing_or_mismatched_slots_rejected(self):
        record=fixture();record['trials'].pop()
        with self.assertRaises(ValueError):analyze_v_stage(record)
        record=fixture();record['trials'][1]['initial_sha256']='e'*64
        with self.assertRaisesRegex(ValueError,'identity mismatch'):analyze_v_stage(record)
        record=fixture();record['trials'][1]['split']='T'
        with self.assertRaisesRegex(ValueError,'backend/split'):analyze_v_stage(record)

    def test_c_development_summary_cannot_be_relabelled_as_formal_v(self):
        record=fixture(False)
        with tempfile.TemporaryDirectory() as tmp:
            add_fake_evidence(record,tmp)
            first=record['trials'][0]
            path=Path(tmp)/first['evidence']['summary']['path']
            summary=json.loads(path.read_text());summary['formal_trial']=False
            path.write_text(json.dumps(summary))
            first['evidence']['summary']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            result=analyze_v_stage(record,evidence_root=tmp)
            self.assertEqual(result['evidence_failures'][first['trial_id']],'trial_not_formal')
            self.assertIsNone(result['groups']['ACT/baseline']['success_rate'])
            summary['formal_trial']=True;summary['physical_alignment_verified']=False
            path.write_text(json.dumps(summary))
            first['evidence']['summary']['sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            result=analyze_v_stage(record,evidence_root=tmp)
            self.assertEqual(result['evidence_failures'][first['trial_id']],'physical_alignment_not_verified')
            self.assertIsNone(result['paired_contrasts'])

    def test_repeated_initial_geometry_across_v_blocks_rejected(self):
        record=fixture()
        for row in record['trials']:
            if row['initial_condition_id']=='V02':row['initial_sha256']=f'{1:064x}'
        with self.assertRaisesRegex(ValueError,'Repeated V initial geometry'):
            analyze_v_stage(record)

    def test_planned_and_running_are_not_successes(self):
        record=fixture();record['trials'][0]['status']='not_run';record['trials'][1]['status']='running'
        result=analyze_v_stage(record)
        self.assertEqual(result['started'],39)
        self.assertEqual(result['complete_blocks'],9)
        self.assertEqual(result['excluded_blocks'][0]['reason'],'not_all_terminal')
        self.assertEqual(result['groups']['ACT/baseline']['started'],9)


if __name__=='__main__':unittest.main()
