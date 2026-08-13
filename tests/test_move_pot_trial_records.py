import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from scripts.move_pot_trial_records import initialize, validate


class TrialRecordTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name)/'records'
        with contextlib.redirect_stdout(io.StringIO()):initialize(self.out)
    def tearDown(self):self.tmp.cleanup()
    def edit(self,filename,fn):
        path=self.out/filename;data=json.loads(path.read_text());fn(data)
        path.write_text(json.dumps(data))
    def test_template_is_complete_but_not_field_ready(self):
        result=validate(self.out)
        self.assertEqual(result['slots'],40);self.assertEqual(result['started'],0)
        self.assertIsNone(result['success_fraction_among_started'])
        self.assertFalse(result['field_ready'])
    def test_reordered_slot_is_rejected(self):
        self.edit('trials.json',lambda d:d['trials'].reverse())
        with self.assertRaises(ValueError):validate(self.out)
    def test_assisted_success_is_rejected(self):
        def edit(d):d['trials'][0].update(status='success',started_utc='t1',ended_utc='t2',human_intervention=True,independent_success=True)
        self.edit('trials.json',edit)
        with self.assertRaises(ValueError):validate(self.out)
    def test_gate_rejection_stays_in_started_denominator(self):
        def edit(d):d['trials'][0].update(status='rejected',started_utc='t1',ended_utc='t2',
            human_intervention=False,independent_success=False,failure_class='gate_rejection',
            event_log='synthetic_fixture',reviewer='test',checkpoint_sha256='fixture',initial_condition_sha256='fixture')
        self.edit('trials.json',edit)
        result=validate(self.out)
        self.assertEqual(result['started'],1);self.assertEqual(result['success_fraction_among_started'],0)
    def test_normalized_unit_gate_is_rejected(self):
        self.edit('execution_lock.json',lambda d:d.update(joint_units=['normalized']*6))
        with self.assertRaises(ValueError):validate(self.out)
    def test_bad_limits_rejected(self):
        self.edit('execution_lock.json',lambda d:d['common_limits'].update(lower=-100))
        with self.assertRaises(ValueError):validate(self.out)

if __name__=='__main__':unittest.main()
