import unittest
from cross_backend.legacy_receipt_bridge import bridge_legacy_receipts, bridge_legacy_stop


class LegacyReceiptBridgeTest(unittest.TestCase):
    def events(self):
        return [
            {'event': 'legacy_goal_packet_transmitted', 'raw': [2000] * 6,
             'transport_returned': True, 'per_motor_acknowledged': False, 'host_ns': 100},
            {'event': 'legacy_goal_candidate', 'raw_ids_values': {str(i): 2100 for i in range(1, 7)},
             'reasons': ['trial_gate'], 'hardware_dispatched': False, 'host_ns': 200},
        ]

    def test_transport_send_and_gate_rejection_remain_distinct(self):
        capabilities, receipts = bridge_legacy_receipts(self.events(), 1)
        self.assertEqual(capabilities['dispatch_ack_scope'], 'sync_transport_return')
        self.assertEqual([r['physical_dispatches'] for r in receipts], [1, 0])
        self.assertEqual([r['accepted'] for r in receipts], [True, False])

    def test_stop_hold_transport_requires_return_after_attempt(self):
        events = self.events() + [
            {'event': 'legacy_stop_hold_attempt', 'host_ns': 300},
            {'event': 'legacy_stop_hold_transport_return', 'host_ns': 400,
             'per_motor_acknowledged': False},
        ]
        stop = bridge_legacy_stop(events)
        self.assertEqual(stop['physical_hold_dispatches'], 1)
        self.assertEqual(stop['mechanism'], 'current_position_sync_hold')
        events[-1]['host_ns'] = 200
        with self.assertRaisesRegex(ValueError, 'stop-hold'):
            bridge_legacy_stop(events)

    def test_false_servo_ack_or_count_mismatch_rejected(self):
        events = self.events()
        events[0]['per_motor_acknowledged'] = True
        with self.assertRaisesRegex(ValueError, 'transport evidence'):
            bridge_legacy_receipts(events, 1)
        events = self.events()
        with self.assertRaisesRegex(ValueError, 'count'):
            bridge_legacy_receipts(events, 2)


if __name__ == '__main__':
    unittest.main()
