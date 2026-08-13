import unittest
from scripts.capture_move_pot_readonly import readonly_class


class FakeBus:
    def __init__(self):
        self.disconnections = []

    def disconnect(self, disable_torque=True):
        self.disconnections.append(disable_torque)


class ReadOnlyBusTests(unittest.TestCase):
    def test_write_paths_refuse_without_base_calls(self):
        bus = readonly_class(FakeBus)()
        for name in ('write', 'sync_write', '_write', '_sync_write', 'write_calibration',
                     'configure_motors', 'enable_torque', 'disable_torque', '_enable_torque', '_disable_torque'):
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                getattr(bus, name)('Goal_Position', 'gripper', 123)

    def test_disconnect_never_requests_torque_write(self):
        bus = readonly_class(FakeBus)()
        bus.disconnect()
        bus.disconnect(disable_torque=False)
        with self.assertRaises(RuntimeError):
            bus.disconnect(disable_torque=True)
        self.assertEqual(bus.disconnections, [False, False])


if __name__ == '__main__':
    unittest.main()
