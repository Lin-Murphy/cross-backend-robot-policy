import unittest
from types import SimpleNamespace

from scripts.run_so101_legacy30_audited_pilot import prepare_postconnect_registers
from cross_backend.legacy_so101_sync_guard import NAMES


POSITION = dict(zip(NAMES, (2091, 1267, 2443, 2722, 2037, 2142)))
SPEED = dict(zip(NAMES, (1200, 1200, 1200, 1200, 400, 250)))


class FakeBus:
    def __init__(self, *, powered_reset):
        self.motors = {name: SimpleNamespace(id=i) for i, name in enumerate(NAMES, 1)}
        self.registers = {
            'Present_Position': POSITION.copy(),
            'Goal_Position': ({name: 0 for name in NAMES} if powered_reset else POSITION.copy()),
            'Goal_Velocity': ({name: 0 for name in NAMES} if powered_reset else SPEED.copy()),
            'Torque_Enable': {name: 1 for name in NAMES},
        }
        self.actions = []

    def read(self, register, name, **_):
        return self.registers[register][name]

    def write(self, register, name, value, **_):
        assert register == 'Goal_Velocity'
        assert self.registers['Goal_Position'] == POSITION
        self.actions.append(('velocity', name, value))
        self.registers[register][name] = value

    def direct_sync(self, address, length, ids, **_):
        assert (address, length) == (42, 2)
        assert ids == {i: POSITION[name] for i, name in enumerate(NAMES, 1)}
        self.actions.append(('hold', ids))
        self.registers['Goal_Position'] = POSITION.copy()
        return 0


class StartupTest(unittest.TestCase):
    def test_power_reset_holds_current_before_setting_velocity(self):
        bus = FakeBus(powered_reset=True)
        events = []
        result = prepare_postconnect_registers(bus, bus.direct_sync, SPEED, events.append)
        self.assertEqual(result, SPEED)
        self.assertEqual(bus.actions[0][0], 'hold')
        self.assertEqual(len(bus.actions), 7)
        self.assertEqual(bus.registers['Goal_Position'], POSITION)
        self.assertEqual([e['event'] for e in events],
                         ['legacy_postconnect_registers_before_prepare',
                          'legacy_startup_current_hold_transport_return',
                          'legacy_startup_velocity_set'])

    def test_preserves_already_prepared_registers(self):
        bus = FakeBus(powered_reset=False)
        result = prepare_postconnect_registers(bus, bus.direct_sync, SPEED, lambda _: None)
        self.assertEqual(result, SPEED)
        self.assertEqual(bus.actions, [])

    def test_mixed_velocity_refused_before_writes(self):
        bus = FakeBus(powered_reset=True)
        bus.registers['Goal_Velocity'][NAMES[0]] = 1
        with self.assertRaisesRegex(RuntimeError, 'unexpected_goal_velocity'):
            prepare_postconnect_registers(bus, bus.direct_sync, SPEED, lambda _: None)
        self.assertEqual([a[0] for a in bus.actions], ['hold'])


if __name__ == '__main__':
    unittest.main()
