import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from cross_backend.smolvla_move_pot import SmolVLAMovePotAdapter

CP=Path('/home/murphy/project/lerobot/outputs/train/smolvla_move_pot_dualcam_20260923_v3/checkpoints/040000/pretrained_model')


class SmolProfileTests(unittest.TestCase):
    def setUp(self):
        data=json.loads((CP/'config.json').read_text())
        self.cfg=SimpleNamespace(**data)
        self.cfg.input_features={k:SimpleNamespace(**v) for k,v in data['input_features'].items()}
        self.cfg.image_features={k:v for k,v in self.cfg.input_features.items() if '.images.' in k}
        self.cfg.action_feature=SimpleNamespace(**data['output_features']['action'])
        self.pre=json.loads((CP/'policy_preprocessor.json').read_text())
    def test_actual_profile_and_mapping(self):
        SmolVLAMovePotAdapter.validate_profile(self.cfg,self.pre)
    def test_wrong_mapping_rejected(self):
        self.pre['steps'][0]['config']['rename_map']['observation.images.follower']='observation.images.camera2'
        with self.assertRaises(ValueError):SmolVLAMovePotAdapter.validate_profile(self.cfg,self.pre)
    def test_aloha_or_bad_padding_rejected(self):
        for name,value in [('adapt_to_pi_aloha',True),('empty_cameras',0),('n_action_steps',51)]:
            cfg=copy.deepcopy(self.cfg);setattr(cfg,name,value)
            with self.assertRaises(ValueError):SmolVLAMovePotAdapter.validate_profile(cfg,self.pre)
    def test_missing_task_rejected_before_preprocessing(self):
        from cross_backend.move_pot_policy import MovePotObservation,CAMERAS
        import numpy as np
        obs=MovePotObservation({k:np.zeros((480,640,3),dtype=np.uint8) for k in CAMERAS},np.zeros(6),
                              'test',{k:'f' for k in CAMERAS},{k:None for k in CAMERAS},None,task='')
        adapter=object.__new__(SmolVLAMovePotAdapter)
        with self.assertRaises(ValueError):adapter._prepare(obs)

if __name__=='__main__':unittest.main()
