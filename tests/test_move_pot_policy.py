"""Contract tests use synthetic CPU fixtures; no robotics or training dependencies."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cross_backend.move_pot_policy import (MovePotObservation, MovePotChunk, ReplayChunkQueue,
                                           CAMERAS, JOINTS)


class ContractTests(unittest.TestCase):
    def observation(self):
        return MovePotObservation({k: np.zeros((480,640,3),dtype=np.uint8) for k in CAMERAS},
            np.zeros(6), 'o1', {k: 'frame1' for k in CAMERAS}, {k: None for k in CAMERAS}, None)

    def test_input_rejects_missing_camera_bad_order_dtype_and_nan(self):
        from dataclasses import replace
        obs = self.observation(); obs.validate()
        for bad in [replace(obs, images_rgb={CAMERAS[0]:obs.images_rgb[CAMERAS[0]]}),
                    replace(obs, joint_names=JOINTS[::-1]),
                    replace(obs, joint_units=('normalized',)*6),
                    replace(obs, images_rgb={k: v.astype(float) for k,v in obs.images_rgb.items()}),
                    replace(obs, state=np.full(6,np.nan)),
                    replace(obs, camera_capture_ns={k:-1 for k in CAMERAS})]:
            with self.assertRaises(ValueError):bad.validate()

    def test_queue_provenance_prefix_and_reset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'events.jsonl';q=ReplayChunkQueue(path,'trial1')
            source={'camera_capture_ns':{CAMERAS[0]:10,CAMERAS[1]:12},'state_capture_ns':15}
            actions=np.arange(300,dtype=float).reshape(50,6)
            chunk=MovePotChunk(actions,source,20,30,0.01,0.02)
            q.enqueue(chunk,2)
            actions[0]=999;source['state_capture_ns']=999
            np.testing.assert_array_equal(q.pop(),np.arange(6))
            q.reset()
            with self.assertRaises(IndexError):q.pop()
            q.close()
            events=[json.loads(s) for s in path.read_text().splitlines()]
            self.assertEqual(events[1]['source']['state_capture_ns'],15)
            self.assertEqual(events[-1]['discarded_actions'],1)
            self.assertIsNone(events[1]['dispatch_monotonic_ns'])
            self.assertFalse(events[1]['hardware_dispatched'])

    def test_rejection_is_persisted_before_raise_and_latched(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'events.jsonl';q=ReplayChunkQueue(path,'trial1')
            q.enqueue(MovePotChunk(np.zeros((50,6)),{'observation_id':'original'},1,2,0,0),50)
            def reject(action):raise ValueError('synthetic gate rejection')
            with self.assertRaises(ValueError):q.pop(reject)
            event=json.loads(path.read_text().splitlines()[-1])
            self.assertEqual(event['gate_status'],'rejected')
            self.assertEqual(event['source']['observation_id'],'original')
            self.assertEqual(event['chunk_offset'],0)
            with self.assertRaises(ValueError):q.pop()
            q.reset();self.assertFalse(q.pending);q.close()



if __name__=='__main__':unittest.main()
