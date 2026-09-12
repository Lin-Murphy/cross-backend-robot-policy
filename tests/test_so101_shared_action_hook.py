import time
import unittest
import tempfile
from pathlib import Path
import numpy as np
from scripts.run_so101_legacy30_audited_pilot import install_shared_action_boundary
from cross_backend.execution_contract import StopReceipt
from cross_backend.legacy_so101_sync_guard import LegacyGuardRejected, NAMES


class Guard:
    def __init__(self):
        self.feedback = {name: 2000 for name in NAMES}
        self.source_observation_ns = None
        self.goal_packets = 0
        self.latched = False


class Robot:
    def __init__(self, guard):
        self.guard = guard
        self.sent = []
        self.reject = False
        self.reads = 0

    def get_observation(self):
        self.reads += 1
        raise AssertionError('hook must reuse already captured observation')

    def send_action(self, action):
        if self.reject:
            self.guard.latched = True
            raise LegacyGuardRejected(['fake_gate'])
        self.sent.append(action)
        self.guard.goal_packets += 1
        return action


def frame():
    row = {name: np.zeros((480, 640, 3), dtype=np.uint8)
           for name in ('follower', 'camera2')}
    row.update({name + '.pos': float(index) for index, name in enumerate(NAMES)})
    return row


class SharedHookTest(unittest.TestCase):
    def test_saves_exact_first_shared_observation_for_offline_replay(self):
        guard = Guard()
        robot = Robot(guard)
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'first-shared-observation.npz'
            on_observation, restore, _ = install_shared_action_boundary(
                robot, guard, lambda: StopReceipt(True, 'mock_hold', 1, 'sync_transport_return'),
                events.append, 'offline-unit', first_observation_path=path)
            try:
                source = frame()
                source['camera2'][0, 0] = [17, 23, 31]
                started = time.monotonic_ns()
                guard.source_observation_ns = started
                on_observation(source, started)
                on_observation(source, started)
                with np.load(path) as saved:
                    self.assertEqual(saved['observation.state'].tolist(),
                                     [float(i) for i in range(6)])
                    self.assertEqual(saved['observation.images.camera2'][0, 0].tolist(),
                                     [17, 23, 31])
                self.assertEqual(sum(e['event']=='shared_first_live_observation_saved'
                                     for e in events), 1)
                self.assertEqual(robot.reads, 0)
            finally:
                restore()

    def test_reuses_existing_observation_and_original_sender(self):
        guard = Guard()
        robot = Robot(guard)
        events = []
        stops = []
        def stop():
            stops.append(True)
            return StopReceipt(True, 'mock_hold', 1, 'sync_transport_return')
        original = robot.send_action
        on_observation, restore, _ = install_shared_action_boundary(
            robot, guard, stop, events.append, 'offline-unit')
        try:
            started = time.monotonic_ns()
            guard.source_observation_ns = started
            on_observation(frame(), started)
            action = {name + '.pos': float(index) for index, name in enumerate(NAMES)}
            robot.send_action(action)
            self.assertEqual(robot.sent, [action])
            self.assertEqual(robot.reads, 0)
            self.assertEqual(guard.goal_packets, 1)
            self.assertEqual([event['event'] for event in events],
                             ['shared_backend_observation', 'execution_observation', 'execution_request',
                              'execution_receipt', 'shared_backend_action_receipt'])
            robot.reject = True
            started = time.monotonic_ns()
            guard.source_observation_ns = started
            on_observation(frame(), started)
            with self.assertRaises(LegacyGuardRejected):
                robot.send_action(action)
            self.assertEqual(stops, [True])
            self.assertEqual(guard.goal_packets, 1)
        finally:
            restore()
        self.assertEqual(robot.send_action, original)


if __name__ == '__main__':
    unittest.main()
