import unittest
from cross_backend.legacy_bus_audit import install_legacy_bus_audit
from cross_backend.legacy_so101_sync_guard import LegacyGuardRejected


class FakeBus:
    def __init__(self):self.calls=[]
    def _write(self,*args,**kwargs):
        self.calls.append(('_write',args,kwargs));return 0
    def _sync_write(self,*args,**kwargs):
        self.calls.append(('_sync_write',args,kwargs));return 1


class LegacyBusAuditTest(unittest.TestCase):
    def test_preserves_raw_payload_return_and_call_count(self):
        bus=FakeBus();events=[];times=iter(range(10,100))
        restore=install_legacy_bus_audit(bus,events.append,lambda:next(times))
        self.assertEqual(bus._sync_write(42,2,{1:2100,2:1400},num_retry=0),1)
        self.assertEqual(bus._write(46,2,1,1200,num_retry=0),0)
        restore()
        self.assertEqual(bus.calls,[('_sync_write',(42,2,{1:2100,2:1400}),{'num_retry':0}),
                                    ('_write',(46,2,1,1200),{'num_retry':0})])
        self.assertEqual([e['event'] for e in events],
            ['legacy_bus_transmit','legacy_bus_transmit_return']*2)
        self.assertEqual(events[0]['args'][2],{1:2100,2:1400})

    def test_goal_guard_rejects_before_packet_and_allows_configuration(self):
        class Guard:
            def check_goal_packet(self,*args):raise LegacyGuardRejected(['stop'])
            def reject(self,reasons):raise LegacyGuardRejected(reasons)
        bus=FakeBus();events=[];install_legacy_bus_audit(bus,events.append,lambda:1,goal_guard=Guard())
        self.assertEqual(bus._sync_write(40,1,{1:0}),1)
        with self.assertRaises(LegacyGuardRejected):bus._sync_write(42,2,{1:2100})
        with self.assertRaises(LegacyGuardRejected):bus._write(42,2,1,2100)
        self.assertEqual(len(bus.calls),1)

    def test_exception_preserved_and_logged_without_retry(self):
        class Broken(FakeBus):
            def _sync_write(self,*args,**kwargs):
                self.calls.append(('_sync_write',args,kwargs));raise ConnectionError('lost packet')
        bus=Broken();events=[];install_legacy_bus_audit(bus,events.append,lambda:1)
        with self.assertRaisesRegex(ConnectionError,'lost packet'):bus._sync_write(42,2,{1:2100})
        self.assertEqual(len(bus.calls),1)
        self.assertEqual(events[-1]['event'],'legacy_bus_transmit_error')


if __name__=='__main__':unittest.main()
