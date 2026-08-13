import unittest

from cross_backend.gripper_recovery_microtrial import (NAMES, OFFSETS, TrialStop,
                                                       restricted_bus_class, run_trial, verify_disabled_preflight)


class Clock:
    def __init__(self): self.t = 0.0
    def now(self): return self.t
    def sleep(self, amount): self.t += amount


class FakeBus:
    def __init__(self):
        self.state = {n: 2000 for n in NAMES}
        self.goal = 1999
        self.torque = {n: 0 for n in NAMES}
        self.writes = []
        self.fail_on = None
        self.move = True

    def write(self, name, motor, value, **kwargs):
        addr, length = ((40, 1) if name == 'Torque_Enable' else (42, 2))
        return self._write(addr, length, 6, value, num_retry=kwargs['num_retry'])

    def _write(self, addr, length, motor_id, value, **kwargs):
        self.writes.append((addr, length, motor_id, value))
        if len(self.writes) == self.fail_on:
            raise ConnectionError('Injected communication failure')
        if addr == 40:
            self.torque['gripper'] = value
        else:
            self.goal = value
            if self.move and self.torque['gripper']:
                self.state['gripper'] = value

    def read(self, name, motor, **kwargs):
        if name == 'Present_Position': return self.state[motor]
        if name == 'Goal_Position': return self.goal
        if name == 'Torque_Enable': return self.torque[motor]
        raise AssertionError(name)

    def disconnect(self, disable_torque=True):
        self.disconnect_value = disable_torque


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.bus = restricted_bus_class(FakeBus)()
        self.clock = Clock()
        self.events = []
        self.reference = self.bus.state.copy()
        self.cal = {n: {'range_min': 1000, 'range_max': 3000} for n in NAMES}

    def run_probe(self):
        return run_trial(self.bus, self.reference, self.cal,
                         self.clock.now, self.clock.sleep, self.events.append)

    def test_single_gripper_round_trip_and_torque_cleanup(self):
        result = self.run_probe()
        self.bus.disable_gripper_only()
        self.assertEqual(result['acknowledged_goals'], [2000 + x for x in OFFSETS])
        self.assertTrue(result['motion_response_detected'])
        self.assertEqual(self.bus.writes[0:2], [(42, 2, 6, 2000), (40, 1, 6, 1)])
        self.assertEqual(self.bus.writes[-1], (40, 1, 6, 0))
        self.assertTrue(all(row[2] == 6 for row in self.bus.writes))
        self.assertTrue(all(v == 0 for v in self.bus.torque.values()))
        times = [e['monotonic_s'] for e in self.events if e.get('event') == 'write_attempt' and 'monotonic_s' in e]
        self.assertTrue(all(b - a >= .1 for a, b in zip(times, times[1:])))
        self.assertLess(result['elapsed_s'], 8)

    def test_no_response_is_reported_without_false_success(self):
        self.bus.move = False
        result = self.run_probe()
        self.bus.disable_gripper_only()
        self.assertFalse(result['motion_response_detected'])
        self.assertEqual(result['gripper_max_observed_excursion_ticks'], 0)

    def test_initial_goal_auto_enables_only_gripper_and_probe_completes(self):
        original = self.bus._write
        def auto_enable(*args, **kwargs):
            original(*args, **kwargs)
            if len(self.bus.writes) == 1 and args[:2] == (42, 2):
                self.bus.torque['gripper'] = 1
        self.bus._write = auto_enable
        result = self.run_probe()
        self.bus.disable_gripper_only()
        self.assertEqual(result['acknowledged_goals'], [2000 + x for x in OFFSETS])
        self.assertEqual(len([w for w in self.bus.writes if w == (40, 1, 6, 1)]), 0)
        self.assertEqual(self.bus.writes[-1], (40, 1, 6, 0))
        self.assertTrue(any(e['event'] == 'gripper_torque_enabled_observed_after_goal' for e in self.events))
        self.assertTrue(all(v == 0 for v in self.bus.torque.values()))

    def test_unexpected_other_joint_torque_blocks_movement(self):
        original = self.bus._write
        def other_torque(*args, **kwargs):
            original(*args, **kwargs)
            if len(self.bus.writes) == 1:
                self.bus.torque['shoulder_lift'] = 1
        self.bus._write = other_torque
        with self.assertRaisesRegex(TrialStop, 'Unexpected torque state'):
            self.run_probe()
        self.assertEqual(self.bus.writes, [(42, 2, 6, 2000)])

    def test_unexpected_joint_drift_stops_sequence(self):
        original = self.bus._write
        def drift(*args, **kwargs):
            original(*args, **kwargs)
            if len(self.bus.writes) == 3: self.bus.state['shoulder_pan'] += 5
        self.bus._write = drift
        with self.assertRaisesRegex(TrialStop, 'Excess displacement'):
            self.run_probe()
        self.bus.disable_gripper_only()
        self.assertEqual(self.bus.writes[-1], (40, 1, 6, 0))
        self.assertEqual(len([w for w in self.bus.writes if w[0] == 42]), 2)

    def test_communication_failure_stops_and_disables(self):
        self.bus.fail_on = 3
        with self.assertRaises(ConnectionError): self.run_probe()
        self.bus.disable_gripper_only()
        self.assertEqual(self.bus.writes[-1], (40, 1, 6, 0))
        self.assertEqual(len(self.bus.writes), 4)

    def test_disabled_preflight_accepts_stale_zero_goals_only_while_torque_off(self):
        self.bus.goal = 0
        result = verify_disabled_preflight(self.bus, self.reference, self.cal, self.clock.sleep)
        self.assertEqual(result['existing_goals']['gripper'], 0)
        self.assertIn('gripper', result['stale_goal_outside_calibration'])
        self.assertEqual(self.bus.writes, [])
        self.bus.torque['gripper'] = 1
        with self.assertRaisesRegex(TrialStop, 'torque changed'):
            verify_disabled_preflight(self.bus, self.reference, self.cal, self.clock.sleep)

    def test_disabled_preflight_rejects_position_drift(self):
        self.bus.state['wrist_flex'] += 1
        with self.assertRaisesRegex(TrialStop, 'Position or torque changed'):
            verify_disabled_preflight(self.bus, self.reference, self.cal, self.clock.sleep)
        self.assertEqual(self.bus.writes, [])

    def test_forbidden_writes_and_rearm(self):
        for name in ('write', 'sync_write', 'enable_torque', 'disable_torque', 'write_calibration'):
            with self.assertRaises(TrialStop): getattr(self.bus, name)('anything')
        with self.assertRaises(TrialStop): self.bus._write(42, 2, 5, 2000)
        with self.assertRaises(TrialStop): self.bus.disconnect(True)
        self.bus.arm(2000, self.clock.now)
        with self.assertRaises(TrialStop): self.bus.arm(2000, self.clock.now)
        with self.assertRaises(TrialStop): self.bus.enable_gripper_only()
        with self.assertRaises(TrialStop): self.bus.write_gripper_goal(2001)
        self.assertFalse(self.bus.writes)


if __name__ == '__main__': unittest.main()
