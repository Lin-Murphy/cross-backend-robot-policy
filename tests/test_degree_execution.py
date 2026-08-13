"""Synthetic thresholds/timestamps only. No model loading or hardware imports."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from cross_backend.degree_execution import DegreeCandidateGate, DegreeGateRejected, DegreeLimits, pop_checked
from cross_backend.move_pot_policy import CAMERAS, JOINTS, UNITS, MovePotChunk, ReplayChunkQueue

MS = 1_000_000


def fixture():
    limits = DegreeLimits((-180.,)*5+(0.,), (180.,)*5+(100.,), (2.,)*6, (100.,)*6,
                          10*MS, 40*MS, 50*MS, 200*MS, 200*MS, 'synthetic_test_only', True)
    source = {'state': [0.]*6, 'joint_names': list(JOINTS), 'joint_units': list(UNITS),
              'observation_id': 'synthetic-o1', 'frame_ids': {k: 'synthetic-frame1' for k in CAMERAS},
              'camera_capture_ns': {k: 0 for k in CAMERAS}, 'state_capture_ns': 0,
              'clock_kind': 'synthetic', 'clock_session': 'synthetic-session'}
    current = {k: copy.deepcopy(source[k]) for k in ('state', 'joint_names', 'joint_units', 'state_capture_ns', 'clock_kind', 'clock_session')}
    return limits, source, current


class DegreeGateTests(unittest.TestCase):
    def setUp(self):
        self.limits, self.source, self.current = fixture()
        self.gate = DegreeCandidateGate(self.limits, 'synthetic-session', 'synthetic')
        self.gate.reset(0)

    def check(self, action=None, now=20*MS, start=0, end=10*MS):
        return self.gate.check(np.ones(6) if action is None else action, self.source, self.current, start, end, now)

    def reject(self, reason, **kwargs):
        with self.assertRaises(DegreeGateRejected) as ctx:
            self.check(**kwargs)
        self.assertIn(reason, str(ctx.exception))
        self.assertTrue(self.gate.latched)

    def test_boundary_first_reference_and_no_clipping(self):
        target = np.full(6, 2.)
        info = self.check(target)
        self.assertEqual(info['reference'], 'current_state_first_candidate')
        np.testing.assert_array_equal(self.gate.last_action, target)
        self.assertEqual(info['action_source_age_ns'], 20*MS)
        with self.assertRaises(RuntimeError): self.gate.sink.send(target)

    def test_step_and_speed(self):
        self.reject('step_exceeded', action=np.full(6, 2.01))
        self.gate.reset(0)
        self.reject('speed_exceeded', action=np.full(6, 1.01), now=10*MS)

    def test_position_and_gripper(self):
        for action in (np.full(6, 181.), np.array([0.]*5+[-.1])):
            self.gate.reset(0)
            self.reject('outside_position_limits', action=action)

    def test_nonfinite_and_shape(self):
        for action in (np.full(6, np.nan), np.full(6, np.inf), np.zeros(5)):
            self.gate.reset(0)
            self.reject('expected_six_finite_values', action=action)
        self.gate.reset(0); self.current['state'][0] = float('nan')
        self.reject('expected_six_finite_values')

    def test_missing_invalid_limits(self):
        for limits in (None, replace(self.limits, max_speed=(0.,)*6),
                       replace(self.limits, max_step=(float('nan'),)*6),
                       replace(self.limits, max_period_ns=None), replace(self.limits, provenance='')):
            self.gate = DegreeCandidateGate(limits, 'synthetic-session', 'synthetic'); self.gate.reset(0)
            with self.assertRaises(DegreeGateRejected): self.check()
            self.assertTrue(self.gate.latched)

    def test_units_order_and_clock(self):
        for key, value, reason in [('joint_units', ['normalized']*6, 'units'),
                                  ('joint_names', list(JOINTS[::-1]), 'joint_order'),
                                  ('clock_session', 'other-boot', 'clock_session'),
                                  ('clock_kind', 'monotonic', 'clock_kind')]:
            for data in (self.source, self.current):
                old = data[key]; data[key] = value; self.gate.reset(0)
                self.reject(reason); data[key] = old

    def test_unknown_future_and_reversed_time(self):
        self.source['camera_capture_ns'][CAMERAS[1]] = None
        self.reject('capture_unknown'); self.gate.reset(0)
        self.source['camera_capture_ns'][CAMERAS[1]] = 1
        self.reject('source_capture_after_prediction'); self.gate.reset(0)
        self.source['camera_capture_ns'][CAMERAS[1]] = 0
        self.reject('prediction_time_future_or_reversed', end=30*MS); self.gate.reset(0)
        self.reject('prediction_time_future_or_reversed', start=15*MS); self.gate.reset(0)
        self.current['state_capture_ns'] = 30*MS
        self.reject('current_state_future')

    def test_period_and_missing_first_baseline(self):
        for now in (0, 9*MS, 41*MS):
            self.gate.reset(0); self.reject('control_period', now=now, end=0)
        self.gate.reset(None); self.reject('invalid_control_time')

    def test_independent_freshness(self):
        for field, reason in [('max_state_age_ns','current_state_stale'),
                              ('max_camera_age_ns','camera_source_stale'),
                              ('max_action_age_ns','action_source_stale')]:
            self.gate = DegreeCandidateGate(replace(self.limits, **{field: 19*MS}), 'synthetic-session', 'synthetic')
            self.gate.reset(0); self.reject(reason)
        self.gate = DegreeCandidateGate(replace(self.limits, max_action_age_ns=20*MS), 'synthetic-session', 'synthetic')
        self.gate.reset(0); self.check()  # equality permitted

    def test_previous_candidate_and_clock_reversal(self):
        self.current['state'] = [10.]*6; self.check(np.full(6, 10.))
        self.current['state'] = [0.]*6
        self.reject('previous_candidate_step_exceeded', action=np.zeros(6), now=40*MS)
        self.gate.reset(0); self.check(np.zeros(6))
        self.reject('control_period', action=np.zeros(6), now=19*MS)

    def test_capture_regression(self):
        self.source['state_capture_ns'] = 5*MS; self.current['state_capture_ns'] = 5*MS
        self.check(start=5*MS)
        self.current['state_capture_ns'] = 4*MS; self.source['state_capture_ns'] = 4*MS
        self.reject('capture_time_reversed', now=40*MS, start=5*MS)

    def test_nonfinite_prediction_and_current_state_are_logged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'events.jsonl'; q = ReplayChunkQueue(path, 'synthetic')
            bad = np.zeros((50,6)); bad[49,0] = np.inf
            with self.assertRaises(ValueError):
                q.enqueue(MovePotChunk(bad,self.source,0,10*MS,10,10),50)
            self.assertTrue(q.failed)
            event = json.loads(path.read_text().splitlines()[-1])
            self.assertEqual(event['actions'][49][0], {'nonfinite':'inf'})
            q.reset(); q.enqueue(MovePotChunk(np.zeros((50,6)),self.source,0,10*MS,10,10),50)
            self.current['state'][0] = float('nan')
            with self.assertRaises(DegreeGateRejected): pop_checked(q,self.gate,self.current,20*MS)
            self.assertTrue(q.failed)
            event = json.loads(path.read_text().splitlines()[-1])
            self.assertEqual(event['evaluation_context']['current']['state'][0], {'nonfinite':'nan'})
            q.reset(); q.close()

    def test_rejection_log_latches_and_reset_discards(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'events.jsonl'; q = ReplayChunkQueue(path, 'synthetic')
            q.enqueue(MovePotChunk(np.full((50,6),3.), self.source, 0,10*MS,10,10),50)
            with self.assertRaises(DegreeGateRejected): pop_checked(q,self.gate,self.current,20*MS)
            with self.assertRaises(ValueError): q.pop()
            q.reset(); self.gate.reset(20*MS)
            q.enqueue(MovePotChunk(np.zeros((50,6)),self.source,0,10*MS,10,10),50)
            pop_checked(q,self.gate,self.current,40*MS); q.reset(); q.close()
            events = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(events[1]['action'], [3.]*6)
            self.assertTrue(events[1]['rejection_reasons'])
            self.assertEqual(len(events[2]['discarded_candidates']),49)
            self.assertEqual(events[4]['gate_status'],'accepted')
            self.assertTrue(all(not e.get('hardware_dispatched', False) for e in events))


if __name__ == '__main__': unittest.main()
