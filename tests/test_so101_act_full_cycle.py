import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cross_backend.legacy_so101_sync_guard import LegacySyncGuard, NAMES
from scripts import run_so101_legacy30_audited_pilot as runner


class FakeBus:
    def __init__(self, start):
        self.raw = dict(zip(NAMES, start))
        self.motors = {name: SimpleNamespace(id=i) for i, name in enumerate(NAMES, 1)}

    def sync_read(self, register, normalize=False, num_retry=0):
        assert register == 'Present_Position'
        return dict(self.raw)


class FullCycleTest(unittest.TestCase):
    def setUp(self):
        with patch.object(runner, 'PROFILE', runner.ACT_FULL_CYCLE_PROFILE):
            self.config, self.profile = runner.load_profile()

    def test_reset_first_action_and_old_roll_bound_are_allowed(self):
        events = []
        guard = LegacySyncGuard(self.profile, events.append, clock=lambda: 1_020_000_000)
        guard.observe(dict(zip(NAMES, (1964, 1178, 2274, 2990, 1773, 1992))),
                      observed_ns=1_000_000_000)
        for target in ((2009, 1135, 2279, 3046, 1867, 2116),
                       (2034, 1837, 2390, 2570, 1748, 1993)):
            reasons, _ = guard._goal_reasons(1_020_000_000, 42, 2,
                                              dict(enumerate(target, 1)))
            self.assertEqual(reasons, [])

    def test_return_reaches_captured_start_and_records_each_packet(self):
        bus = FakeBus((2000, 1500, 2200, 3100, 1900, 2300))
        start = dict(zip(NAMES, (1964, 1178, 2274, 2990, 1773, 1992)))
        events = []
        now = [0]
        def clock(): return now[0]
        def sleep(seconds): now[0] += round(seconds * 1e9)
        def write(address, length, ids, num_retry=0):
            self.assertEqual((address, length), (42, 2))
            bus.raw = {name: ids[i] for i, name in enumerate(NAMES, 1)}
            return None
        final = runner.return_to_captured_start(bus, write, start, self.config,
                                                events.append, clock=clock, sleep=sleep)
        self.assertEqual(final, start)
        self.assertEqual(events[-1]['event'], 'formal_return_end')
        self.assertTrue(events[-1]['within_tolerance'])
        packets = [e for e in events if e['event'] == 'formal_return_goal_transport_return']
        self.assertEqual(len(packets), events[0]['steps'])
        for earlier, later in zip(packets, packets[1:]):
            self.assertLessEqual(max(abs(later['raw_ids_values'][i] - earlier['raw_ids_values'][i])
                                     for i in range(1, 7)), 6)

    def test_preopen_moves_only_gripper_and_return_targets_original_start(self):
        with patch.object(runner,'PROFILE',runner.ACT_PREOPEN_PROFILE):
            approved,_=runner.load_profile()
        start=dict(zip(NAMES,(1979,1186,2821,2897,1791,1989)))
        bus=FakeBus(tuple(start.values()))
        events=[];now=[0]
        def clock():return now[0]
        def sleep(seconds):now[0]+=round(seconds*1e9)
        def write(address,length,ids,num_retry=0):
            self.assertEqual((address,length),(42,2))
            bus.raw={name:ids[i] for i,name in enumerate(NAMES,1)}
            return 0
        opened=runner.preopen_gripper(bus,write,start,approved,events.append,clock=clock,sleep=sleep)
        self.assertEqual(opened['gripper'],2309)
        self.assertEqual({name:opened[name] for name in NAMES[:-1]},
                         {name:start[name] for name in NAMES[:-1]})
        packets=[x for x in events if x['event']=='formal_gripper_preopen_goal_transport_return']
        self.assertGreaterEqual(len(packets),75)
        self.assertLessEqual(max(abs(b['raw_ids_values'][6]-a['raw_ids_values'][6])
                                 for a,b in zip(packets,packets[1:])),4)
        self.assertEqual(events[-1]['event'],'formal_gripper_preopen_end')
        self.assertTrue(events[-1]['within_tolerance'])
        runner.return_to_captured_start(bus,write,start,approved,events.append,clock=clock,sleep=sleep)
        self.assertEqual(bus.raw,start)

    def test_preopen_refuses_unreached_feedback(self):
        with patch.object(runner,'PROFILE',runner.ACT_PREOPEN_PROFILE):
            approved,_=runner.load_profile()
        start=dict(zip(NAMES,(1979,1186,2821,2897,1791,1989)))
        bus=FakeBus(tuple(start.values()))
        now=[0]
        def sleep(seconds):now[0]+=round(seconds*1e9)
        with self.assertRaisesRegex(RuntimeError,'gripper_preopen_not_reached'):
            runner.preopen_gripper(bus,lambda *a,**k:0,start,approved,lambda e:None,
                                   clock=lambda:now[0],sleep=sleep)

    def test_approved_one_time_profile_accepts_targets_beyond_cached_calibration(self):
        with patch.object(runner,'PROFILE',runner.ACT_UNRESTRICTED_PROFILE):
            approved,limits=runner.load_profile()
        self.assertEqual(approved['approved_unrestricted_start_raw'],
                         [1969,1111,2899,2838,1971,2301])
        self.assertEqual(approved['return_original_preposition_start_raw'],
                         [1969,1191,2812,2904,1811,2002])
        self.assertEqual(limits.raw_lower,(0,)*6)
        self.assertEqual(limits.raw_upper,(4095,)*6)
        self.assertEqual(limits.max_feedback_raw,(4095,)*6)
        events=[]
        guard=LegacySyncGuard(limits,events.append,clock=lambda:1_020_000_000)
        guard.observe(dict(zip(NAMES,(1969,1111,2899,2838,1971,2301))),
                      observed_ns=1_000_000_000)
        reasons,_=guard._goal_reasons(1_020_000_000,42,2,
            dict(enumerate((1965,866,3032,2815,1976,2299),1)))
        self.assertEqual(reasons,[])
        reasons,_=guard._goal_reasons(1_020_000_000,42,2,
            dict(enumerate((0,4095,4096,0,4095,0),1)))
        self.assertIn('target_outside_trial_profile:elbow_flex',reasons)


if __name__ == '__main__': unittest.main()
