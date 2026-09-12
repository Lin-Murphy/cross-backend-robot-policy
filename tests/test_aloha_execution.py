import unittest
import numpy as np
from cross_backend.aloha_execution import AlohaBackendAdapter
from cross_backend.execution_contract import ActionRequest
from cross_backend.execution_session import ExecutionSession


class NativeEnv:
    def __init__(self):
        self.observation = {'agent_pos': np.zeros(14), 'pixels': {'top': np.zeros((2, 2, 3), dtype=np.uint8)}}
        self.calls = 0
    def step(self, action):
        self.calls += 1
        self.action = action
        return self.observation, 4, True, False, {'native': 'preserved'}


class AlohaExecutionTests(unittest.TestCase):
    def test_native_transition_and_action_preserved(self):
        env = NativeEnv(); backend = AlohaBackendAdapter(env, env.observation, 1)
        values = tuple(float(x) / 14 for x in range(14))
        with ExecutionSession(backend) as session:
            _, _, receipt, _ = session.step(lambda frame: ActionRequest(frame.observation_id,
                frame.joint_names, frame.joint_units, values))
        np.testing.assert_array_equal(env.action, values)
        self.assertIs(backend.transition[0], env.observation)
        self.assertEqual(backend.transition[1:], (4, True, False, {'native': 'preserved'}))
        self.assertEqual(receipt.physical_dispatches, 0)
        self.assertEqual(env.calls, 1)
        self.assertTrue(backend.stopped)

    def test_wrong_action_shape_never_reaches_native_environment(self):
        env = NativeEnv(); backend = AlohaBackendAdapter(env, env.observation, 1)
        with self.assertRaises(ValueError):
            ExecutionSession(backend).step(lambda frame: ActionRequest(frame.observation_id,
                frame.joint_names, frame.joint_units, (0.,)))
        self.assertEqual(env.calls, 0)
        self.assertTrue(backend.stopped)
