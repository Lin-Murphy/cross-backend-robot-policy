import unittest
from examples.adapter_template.adapters import MemoryBackend, ExamplePolicy
from cross_backend.execution_contract import ActionRequest
from cross_backend.execution_session import ExecutionSession


class TemplateTests(unittest.TestCase):
    def test_model_order_is_mapped_and_normal_exit_stops(self):
        backend = MemoryBackend(); events = []
        with ExecutionSession(backend, events.append) as session:
            for _ in range(3): session.step(ExamplePolicy(backend.capabilities))
        self.assertAlmostEqual(backend.position[0], .03)
        self.assertAlmostEqual(backend.position[1], .06)
        self.assertTrue(backend.stopped)
        self.assertEqual(sum(e['event']=='execution_receipt' for e in events), 3)
        self.assertEqual(events[-1]['event'], 'execution_stop')

    def test_wrong_units_stop_before_dispatch(self):
        backend = MemoryBackend()
        with self.assertRaises(ValueError):
            ExecutionSession(backend).step(lambda frame: ActionRequest(frame.observation_id,
                frame.joint_names, ('rad', 'rad'), (.1, .2)))
        self.assertEqual(backend.index, 0)
        self.assertTrue(backend.stopped)

    def test_inference_failure_stops_backend(self):
        backend = MemoryBackend()
        def broken(frame): raise RuntimeError('model error')
        with self.assertRaisesRegex(RuntimeError, 'model error'):
            ExecutionSession(backend).step(broken)
        self.assertTrue(backend.stopped)
        self.assertEqual(backend.index, 0)
