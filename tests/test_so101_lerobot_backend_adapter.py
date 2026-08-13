import unittest
import numpy as np
from cross_backend.execution_contract import ActionRequest, StopReceipt, execute_one
from cross_backend.so101_lerobot_backend_adapter import SO101LeRobotBackendAdapter
from cross_backend.legacy_so101_sync_guard import LegacyGuardRejected


class FakeGuard:
    def __init__(self):
        self.feedback = {'gripper': 2000}
        self.source_observation_ns = None
        self.latched = False
        self.goal_packets = 0


class FakeRobot:
    def __init__(self, guard, reject=False):
        self.guard = guard
        self.reject = reject
        self.actions = []

    def get_observation(self):
        import time
        self.guard.source_observation_ns = time.monotonic_ns()
        observation = {name: np.zeros((480, 640, 3), dtype=np.uint8)
                       for name in ('follower', 'camera2')}
        observation.update({name + '.pos': float(index)
                            for index, name in enumerate(('shoulder_pan', 'shoulder_lift',
                                'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'))})
        return observation

    def send_action(self, action):
        if self.reject:
            raise LegacyGuardRejected(['fake_gate'])
        self.actions.append(action)
        self.guard.goal_packets += 1


class AdapterTest(unittest.TestCase):
    def make(self, reject=False):
        guard = FakeGuard()
        robot = FakeRobot(guard, reject=reject)
        stops = []
        def stop():
            stops.append(True)
            return StopReceipt(True, 'fake_current_position_hold', 1,
                               'sync_transport_return')
        adapter = SO101LeRobotBackendAdapter(robot, guard, stop,
                                              approved_trial_id='offline-test-only')
        return adapter, robot, stops

    def target(self, observation):
        return ActionRequest(observation.observation_id, observation.joint_names,
                             observation.joint_units, observation.joint_position)

    def test_single_audited_dispatch_shape_and_no_auto_connect(self):
        adapter, robot, stops = self.make()
        observation, request, receipt, stop = execute_one(adapter, self.target)
        self.assertEqual(observation.camera_capture_ns, {'follower': None, 'camera2': None})
        self.assertEqual(len(robot.actions), 1)
        self.assertEqual(tuple(robot.actions[0].values()), request.target)
        self.assertEqual(receipt.ack_scope, 'sync_transport_return')
        self.assertIsNone(stop)
        self.assertEqual(stops, [])

    def test_guard_rejection_invokes_supplied_stop(self):
        adapter, robot, stops = self.make(reject=True)
        with self.assertRaises(LegacyGuardRejected):
            execute_one(adapter, self.target)
        self.assertTrue(adapter.stopped)
        self.assertEqual(stops, [True])
        self.assertEqual(robot.actions, [])

    def test_missing_guard_feedback_rejects_without_send(self):
        adapter, robot, stops = self.make()
        adapter.guard.feedback = None
        _, _, receipt, stop = execute_one(adapter, self.target)
        self.assertFalse(receipt.accepted)
        self.assertEqual(receipt.reason, 'guard_missing_fresh_observation')
        self.assertTrue(stop.completed)
        self.assertEqual(robot.actions, [])


if __name__ == '__main__':
    unittest.main()
