import hashlib
import tempfile
import unittest
from pathlib import Path
from cross_backend.move_pot_policy import CAMERAS,JOINTS,UNITS
from cross_backend.sim_prediction_identity import verify_worker_request,verify_prediction_response


class WorkerIdentityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.input=self.root/'input-000.npz';self.input.write_bytes(b'known simulation input')
        self.input_hash=hashlib.sha256(self.input.read_bytes()).hexdigest()
        self.actions=self.root/'prediction-000.npz';self.actions.write_bytes(b'known output')
        self.request={'trial_id':'C01-act','index':0,'capture_sim_ns':35_000_000,
                      'observation_id':'C01-act:frame:0','input':str(self.input),'input_sha256':self.input_hash}
        self.response={'actions':str(self.actions)}
        self.metadata={'trial_id':'C01-act','input_sha256':self.input_hash,
                       'actions_sha256':hashlib.sha256(self.actions.read_bytes()).hexdigest(),
                       'source':{'observation_id':'C01-act:frame:0','state_capture_ns':35_000_000,
                                 'camera_capture_ns':{k:35_000_000 for k in CAMERAS},
                                 'model_sha256':'a'*64,'joint_names':list(JOINTS),'joint_units':list(UNITS)},
                       'prediction_host_start_ns':100,'prediction_host_end_ns':200,
                       'forward_ms':.01,'processing_ms':.02}
    def tearDown(self):self.tmp.cleanup()
    def verify_reply(self):
        return verify_prediction_response(self.response,self.metadata,trial_id='C01-act',
            observation_id='C01-act:frame:0',capture_sim_ns=35_000_000,prediction_index=0,
            input_sha256=self.input_hash,model_sha256='a'*64,output_dir=self.root)
    def test_matching_input_and_reply(self):
        self.assertEqual(verify_worker_request(self.request,trial_id='C01-act',output_dir=self.root),self.input)
        self.assertEqual(self.verify_reply(),self.actions)
    def test_changed_input_or_trial_rejected(self):
        self.input.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'input hash'):
            verify_worker_request(self.request,trial_id='C01-act',output_dir=self.root)
        self.input.write_bytes(b'known simulation input')
        self.request['trial_id']='other'
        with self.assertRaisesRegex(ValueError,'trial identity'):
            verify_worker_request(self.request,trial_id='C01-act',output_dir=self.root)
    def test_stale_prediction_and_wrong_checkpoint_rejected(self):
        self.metadata['source']['observation_id']='C01-act:frame:previous'
        with self.assertRaisesRegex(ValueError,'observation identity'):self.verify_reply()
        self.metadata['source']['observation_id']='C01-act:frame:0'
        self.metadata['source']['model_sha256']='b'*64
        with self.assertRaisesRegex(ValueError,'checkpoint identity'):self.verify_reply()
    def test_changed_action_file_rejected(self):
        self.actions.write_bytes(b'changed output')
        with self.assertRaisesRegex(ValueError,'actions hash'):self.verify_reply()

    def test_wrong_action_path_and_capture_time_rejected(self):
        self.response['actions']=str(self.input)
        with self.assertRaisesRegex(ValueError,'action path'):self.verify_reply()
        self.response['actions']=str(self.actions)
        self.metadata['source']['camera_capture_ns'][CAMERAS[0]]=0
        with self.assertRaisesRegex(ValueError,'capture time'):self.verify_reply()


if __name__=='__main__':unittest.main()
