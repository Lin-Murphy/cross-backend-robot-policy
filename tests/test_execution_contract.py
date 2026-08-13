import unittest
from cross_backend.execution_contract import (BackendCapabilities, BackendObservation,
    ActionRequest, ActionReceipt, StopReceipt, action_request_from_policy, execute_one)


JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')


class FakeBackend:
    def __init__(self):
        self.capabilities = BackendCapabilities('fake', JOINTS, ('rad',) * 6, (),
            'simulation', 'automatic', 'none', 'fake_only', False)
        self.calls = 0
        self.stopped = False

    def observe(self):
        return BackendObservation('frame:0', JOINTS, ('rad',) * 6,
            (0.,) * 6, {}, {}, {}, 0, 'simulation')

    def dispatch(self, request):
        self.calls += 1
        return ActionReceipt(True, request.target, 'fake_only', 0, 0)

    def stop(self):
        self.stopped = True
        return StopReceipt(True, 'fake_latch', 0, 'fake_only')


class ExecutionContractTest(unittest.TestCase):
    def test_valid_action_receipt(self):
        backend = FakeBackend()
        _, _, receipt, stop = execute_one(backend, lambda obs: ActionRequest(
            obs.observation_id, obs.joint_names, obs.joint_units, obs.joint_position))
        self.assertTrue(receipt.accepted)
        self.assertEqual(backend.calls, 1)
        self.assertFalse(backend.stopped)
        self.assertIsNone(stop)

    def test_wrong_units_or_source_rejected_before_dispatch_and_stopped(self):
        for source, units in (('other', ('rad',) * 6), ('frame:0', ('deg',) * 6)):
            backend = FakeBackend()
            with self.assertRaisesRegex(ValueError, 'action source'):
                execute_one(backend, lambda obs: ActionRequest(source, JOINTS, units, (0.,) * 6))
            self.assertEqual(backend.calls, 0)
            self.assertTrue(backend.stopped)

    def test_fake_cannot_claim_physical_dispatch(self):
        backend = FakeBackend()
        backend.dispatch = lambda request: ActionReceipt(True, request.target, 'fake_only', 1, 0)
        with self.assertRaisesRegex(ValueError, 'nonphysical backend'):
            execute_one(backend, lambda obs: ActionRequest(
                obs.observation_id, obs.joint_names, obs.joint_units, obs.joint_position))
        self.assertTrue(backend.stopped)

    def test_other_robot_joint_count_is_supported_without_so101_names(self):
        backend = FakeBackend()
        backend.capabilities = BackendCapabilities('other_robot', ('joint_a', 'joint_b'),
            ('meter', 'meter'), (), 'simulation', 'automatic', 'none', 'fake_only', False)
        backend.observe = lambda: BackendObservation('other:0', ('joint_a', 'joint_b'),
            ('meter', 'meter'), (0.1, 0.2), {}, {}, {}, 0, 'simulation')
        _, request, receipt, stop = execute_one(backend, lambda obs: ActionRequest(
            obs.observation_id, obs.joint_names, obs.joint_units, obs.joint_position))
        self.assertEqual(request.target, (0.1, 0.2))
        self.assertTrue(receipt.accepted)

    def test_bad_observation_cannot_reach_dispatch(self):
        backend = FakeBackend()
        backend.observe = lambda: BackendObservation('frame:0', JOINTS, ('deg',) * 6,
            (0.,) * 6, {}, {}, {}, 0, 'simulation')
        with self.assertRaisesRegex(ValueError, 'observation identity'):
            execute_one(backend, lambda obs: ActionRequest(
                obs.observation_id, obs.joint_names, obs.joint_units, obs.joint_position))
        self.assertEqual(backend.calls, 0)
        self.assertTrue(backend.stopped)


    def test_named_policy_action_maps_order_for_other_robot(self):
        backend = FakeBackend()
        backend.capabilities = BackendCapabilities('other_robot', ('joint_a', 'joint_b'),
            ('meter', 'radian'), (), 'simulation', 'automatic', 'none', 'fake_only', False)
        observation = BackendObservation('other:0', backend.capabilities.joint_names,
            backend.capabilities.action_units, (0.1, 0.2), {}, {}, {}, 0, 'simulation')
        request = action_request_from_policy(observation, backend.capabilities,
            policy_joint_names=('policy_b', 'policy_a'), policy_action_units=('radian', 'meter'),
            target=(2.0, 1.0), backend_to_policy={'joint_a': 'policy_a', 'joint_b': 'policy_b'})
        self.assertEqual(request.target, (1.0, 2.0))
        self.assertEqual(request.joint_names, ('joint_a', 'joint_b'))

    def test_named_policy_action_rejects_unit_and_mapping_mismatch(self):
        backend = FakeBackend()
        observation = backend.observe()
        mapping = {name: name + '.pos' for name in JOINTS}
        names = tuple(mapping.values())
        with self.assertRaisesRegex(ValueError, 'unit mismatch'):
            action_request_from_policy(observation, backend.capabilities,
                policy_joint_names=names, policy_action_units=('deg',) * 6,
                target=(0.0,) * 6, backend_to_policy=mapping)
        with self.assertRaisesRegex(ValueError, 'not bijective'):
            action_request_from_policy(observation, backend.capabilities,
                policy_joint_names=names, policy_action_units=('rad',) * 6,
                target=(0.0,) * 6, backend_to_policy={**mapping, JOINTS[0]: names[1]})


if __name__ == '__main__':
    unittest.main()
