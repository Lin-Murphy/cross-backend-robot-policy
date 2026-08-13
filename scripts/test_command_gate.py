"""Offline safety properties; no hardware imports."""
import unittest
import numpy as np
from cross_backend.safety import SO101CommandGate, ActionSafetyError


class GateTests(unittest.TestCase):
    def gate(self, scale=1):
        return SO101CommandGate([-100]*5+[0], [100]*6, [2]*6, scale)

    def test_velocity_and_scaling(self):
        state = np.array([10.]*6)
        command = self.gate(.5).prepare(state+.1, state, state, .1)
        np.testing.assert_allclose(command, state+.05)
        command = self.gate().prepare(state+10, state, state, .02)
        np.testing.assert_allclose(command-state, [.04]*6)

    def test_invalid_inputs_latch_stop(self):
        for target, dt in [([0]*5+[-1], .02), ([float('nan')]*6, .02),
                           ([0]*5, .02), ([0]*6, 0), ([0]*6, .11)]:
            gate = self.gate()
            with self.assertRaises((ActionSafetyError, ValueError)):
                gate.prepare(target, [0]*6, [0]*6, dt)
            self.assertTrue(gate.stopped)

    def test_stop_and_tracking_error(self):
        gate = self.gate()
        gate.stop()
        with self.assertRaises(ActionSafetyError):
            gate.prepare([0]*6, [0]*6, [0]*6, .02)
        with self.assertRaises(ActionSafetyError):
            self.gate().prepare([0]*6, [0]*6, [10]*6, .02)


if __name__ == '__main__':
    unittest.main()
