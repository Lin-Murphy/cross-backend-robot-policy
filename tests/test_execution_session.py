import unittest
from cross_backend.execution_contract import ActionRequest, ActionReceipt
from cross_backend.execution_session import ExecutionSession
from test_execution_contract import FakeBackend


def hold(frame):
    return ActionRequest(frame.observation_id, frame.joint_names, frame.joint_units, frame.joint_position)


class SessionTests(unittest.TestCase):
    def test_multiple_steps_and_normal_exit_stop_once(self):
        backend = FakeBackend(); events = []; stops = []
        original = backend.stop
        def stop():
            stops.append(True)
            return original()
        backend.stop = stop
        with ExecutionSession(backend, events.append) as session:
            session.step(hold); session.step(hold)
        session.stop()
        self.assertEqual(backend.calls, 2)
        self.assertEqual(stops, [True])
        self.assertEqual(events[-1]['event'], 'execution_stop')
        with self.assertRaises(RuntimeError): session.step(hold)

    def test_captured_frame_does_not_read_again(self):
        backend = FakeBackend(); frame = backend.observe()
        backend.observe = lambda: self.fail('extra observation')
        with ExecutionSession(backend) as session: session.step(hold, observation=frame)

    def test_invalid_observation_never_reaches_policy(self):
        backend = FakeBackend()
        with self.assertRaises(AttributeError):
            ExecutionSession(backend).step(lambda frame: self.fail('policy invoked'), observation=object())
        self.assertTrue(backend.stopped)
        self.assertEqual(backend.calls, 0)

    def test_rejection_latches_session(self):
        backend = FakeBackend()
        backend.dispatch = lambda req: ActionReceipt(False, req.target, 'fake_only', 0, None, 'test_reject')
        session = ExecutionSession(backend)
        self.assertFalse(session.step(hold)[2].accepted)
        self.assertTrue(session.closed)
        with self.assertRaises(RuntimeError): session.step(hold)

    def test_recording_error_after_send_stops_without_retry(self):
        backend = FakeBackend()
        def record(event):
            if event['event'] == 'execution_receipt': raise OSError('disk full')
        session = ExecutionSession(backend, record)
        with self.assertRaisesRegex(OSError, 'disk full'): session.step(hold)
        self.assertEqual(backend.calls, 1)
        self.assertTrue(backend.stopped)

    def test_stop_error_preserves_original_error(self):
        backend = FakeBackend()
        def stop(): raise OSError('stop failed')
        backend.stop = stop
        def provider(frame): raise ValueError('inference failed')
        with self.assertRaisesRegex(ValueError, 'inference failed') as error:
            ExecutionSession(backend).step(provider)
        self.assertIn('stop failed', error.exception.__notes__[0])
