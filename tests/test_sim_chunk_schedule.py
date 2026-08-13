import unittest,tempfile
from pathlib import Path
import numpy as np
from cross_backend.move_pot_policy import ReplayChunkQueue
from cross_backend.sim_chunk_schedule import SimChunkSchedule
class ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.q=ReplayChunkQueue(Path(self.tmp.name)/'events.jsonl','test')
    def tearDown(self):self.q.close();self.tmp.cleanup()
    def test_late_prediction_holds_and_never_refreshes_source(self):
        s=SimChunkSchedule(self.q,lambda obs:np.zeros((3,6)),latency_ns=25,execute_steps=3,prefetch_remaining=1)
        seen=[]
        for t in range(0,100,10):s.capture(t,{'observation_id':str(t)});seen.append(s.tick(t,lambda a,e:None))
        self.assertTrue(all(a is None for a in seen[:3]))
        selected=[e for e in s.events if e['event']=='sim_target_selected']
        self.assertEqual([e['source']['capture_sim_ns'] for e in selected[:3]],[0]*3)
        self.assertEqual([e['source_age_ns'] for e in selected[:3]],[30,40,50])
        self.assertEqual([e['chunk_offset'] for e in selected[:3]],[0,1,2])
    def test_delay_uses_old_observation(self):
        s=SimChunkSchedule(self.q,lambda obs:np.zeros((3,6)),latency_ns=0,observation_delay_ns=15,execute_steps=3,prefetch_remaining=1)
        for t in (0,10,20):s.capture(t,{'observation_id':str(t)});s.tick(t,lambda a,e:None)
        e=[e for e in s.events if e['event']=='sim_target_selected'][0];self.assertEqual(e['source']['capture_sim_ns'],0);self.assertEqual(e['sim_ns'],20)
    def test_rejection_latches_no_recovery_action(self):
        s=SimChunkSchedule(self.q,lambda obs:np.zeros((3,6)),latency_ns=0,execute_steps=3,prefetch_remaining=1);s.capture(0,{'observation_id':'x'})
        def reject(a,e):raise ValueError('gate')
        with self.assertRaises(ValueError):s.tick(0,reject)
        with self.assertRaises(ValueError):s.tick(1,lambda a,e:None)
        self.assertFalse(any(e['event']=='sim_target_selected' for e in s.events))
    def test_reset_discards_old_job_and_history(self):
        s=SimChunkSchedule(self.q,lambda obs:np.zeros((3,6)),latency_ns=20,execute_steps=3,prefetch_remaining=1)
        s.capture(0,{'observation_id':'old'});s.tick(0,lambda a,e:None);s.reset()
        self.assertIsNone(s.tick(0,lambda a,e:None));self.assertIsNone(s.job)
    def test_one_frame_with_uneven_capture_intervals(self):
        s=SimChunkSchedule(self.q,lambda obs:np.zeros((3,6)),latency_ns=0,observation_delay_frames=1,execute_steps=3,prefetch_remaining=1)
        for t in (0,35,70,100,135,170,200,235):
            s.capture(t,{'observation_id':str(t)});s.tick(t,lambda a,e:None)
        jobs=[e for e in s.events if e['event']=='prediction_scheduled']
        self.assertTrue(jobs);self.assertTrue(all(e['realized_delay_frames']==1 for e in jobs))
        self.assertEqual(jobs[0]['capture_sim_ns'],0);self.assertEqual(jobs[0]['request_sim_ns'],35)
    def test_future_capture_not_selected(self):
        s=SimChunkSchedule(self.q,lambda obs:np.zeros((3,6)),latency_ns=0,observation_delay_frames=1,execute_steps=3,prefetch_remaining=1)
        for t in (0,30,1000):s.capture(t,{'observation_id':str(t)})
        s.tick(35,lambda a,e:None)
        e=[e for e in s.events if e['event']=='prediction_scheduled'][0]
        self.assertEqual(e['capture_sim_ns'],0);self.assertEqual(e['latest_observation_sequence'],1)
    def test_delay_modes_cannot_mix(self):
        for frames in (-1,True,1.5):
            with self.assertRaises(ValueError):SimChunkSchedule(self.q,None,latency_ns=0,observation_delay_frames=frames)
        with self.assertRaises(ValueError):SimChunkSchedule(self.q,None,latency_ns=0,observation_delay_ns=10,observation_delay_frames=1)
    def test_measured_latency_is_injected_once(self):
        from unittest.mock import patch
        s=SimChunkSchedule(self.q,lambda obs:np.zeros((3,6)),latency_ns='measured_host',execute_steps=3,prefetch_remaining=1)
        s.capture(0,{'observation_id':'x'})
        with patch('cross_backend.sim_chunk_schedule.time.perf_counter_ns',side_effect=[100,125]):
            self.assertIsNone(s.tick(0,lambda a,e:None))
        job=[e for e in s.events if e['event']=='prediction_scheduled'][0]
        self.assertEqual(job['available_sim_ns'],25);self.assertEqual(job['host_compute_ns'],25)
        self.assertIsNone(s.tick(24,lambda a,e:None));self.assertIsNotNone(s.tick(25,lambda a,e:None))
    def test_model_timing_and_source_preserved(self):
        from cross_backend.move_pot_policy import MovePotChunk
        chunk=MovePotChunk(np.zeros((3,6)),{'noise_sha256':'fixture'},100,120,0.00001,0.00002)
        s=SimChunkSchedule(self.q,lambda obs:chunk,latency_ns=0,execute_steps=3,prefetch_remaining=1)
        s.capture(0,{'observation_id':'x'});s.tick(0,lambda a,e:None)
        self.assertEqual(self.q.pending[0][3]['model_source']['noise_sha256'],'fixture')
        self.assertEqual(self.q.pending[0][4:6],(100,120))
if __name__=='__main__':unittest.main()
