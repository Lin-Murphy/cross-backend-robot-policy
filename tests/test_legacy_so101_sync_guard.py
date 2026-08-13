import unittest
from dataclasses import replace
from cross_backend.legacy_so101_sync_guard import LegacyTrialProfile,LegacySyncGuard,LegacyGuardRejected,NAMES


class LegacyGuardTest(unittest.TestCase):
    def setUp(self):
        self.now=[1_000_000_000];self.events=[]
        profile=LegacyTrialProfile((1800,1150,1100,2500,1800,1877),
            (2460,2520,2710,3265,2120,2440),(2460,2520,2710,3189,2120,2440),(100,210,320,170,60,240),
            (60,100,180,80,50,180),(5000,5000,5000,5000,5000,5000),
            2_000_000_000,20_000_000,2_000_000_000,15_000_000_000,450,'test proposal')
        self.guard=LegacySyncGuard(profile,self.events.append,lambda:self.now[0])
        self.state=dict(zip(NAMES,(2093,1374,2601,3090,1976,2285)))
        self.guard.observe(self.state)
    def packet(self,values):return {i:v for i,v in enumerate(values,1)}
    def test_accepts_two_historical_style_goals_without_mutating(self):
        first=self.packet((2114,1333,2497,3056,1983,2288))
        at,raw=self.guard.check_goal_packet(42,2,first);self.guard.accepted_transmit(at,raw)
        self.now[0]+=33_333_333
        second=self.packet((2107,1322,2405,3080,1986,2286))
        at,raw=self.guard.check_goal_packet(42,2,second);self.guard.accepted_transmit(at,raw)
        self.assertEqual(second[3],2405)
        self.assertEqual(len([e for e in self.events if e['event']=='legacy_goal_packet_transmitted']),2)
    def test_disabled_previous_target_gate_keeps_feedback_gate(self):
        profile=replace(self.guard.profile, max_from_last_target=(4095,)*6,
                        max_from_feedback=(100,160,200,110,45,160))
        guard=LegacySyncGuard(profile,self.events.append,lambda:self.now[0])
        feedback=dict(zip(NAMES,(2015,1525,2141,2962,1881,2301)))
        guard.observe(feedback)
        guard.last_target=(2018,1526,2175,2918,1874,2300)
        guard.last_goal_ns=self.now[0]-40_000_000
        candidate=self.packet((1996,1493,2190,2880,1862,2320))
        reasons,_=guard._goal_reasons(self.now[0],42,2,candidate)
        self.assertEqual(reasons,[])
        too_far=self.packet((1996,1493,2190,2840,1862,2320))
        reasons,_=guard._goal_reasons(self.now[0],42,2,too_far)
        self.assertIn('target_from_feedback_exceeded:wrist_flex',reasons)

    def test_early_second_goal_waits_to_minimum_period(self):
        first=self.packet((2114,1333,2497,3056,1983,2288))
        at,raw=self.guard.check_goal_packet(42,2,first);self.guard.accepted_transmit(at,raw)
        self.now[0]+=9_000_000
        waits=[]
        def advance(seconds):
            waits.append(seconds)
            self.now[0]+=round(seconds*1e9)
        self.guard.sleep=advance
        second=self.packet((2107,1322,2405,3080,1986,2286))
        at,raw=self.guard.check_goal_packet(42,2,second)
        self.assertEqual(at-self.guard.last_goal_ns,20_000_000)
        self.assertEqual(waits,[0.011])
        self.guard.accepted_transmit(at,raw)
        self.assertEqual(self.guard.goal_packets,2)
        self.assertEqual(len([e for e in self.events if e['event']=='legacy_goal_paced']),1)

    def test_period_is_measured_from_previous_transport_completion(self):
        first=self.packet((2114,1333,2497,3056,1983,2288))
        at,raw=self.guard.check_goal_packet(42,2,first)
        self.now[0]+=3_000_000
        self.guard.accepted_transmit(at,raw)
        completed=self.now[0]
        self.now[0]+=6_000_000
        def advance(seconds):self.now[0]+=round(seconds*1e9)
        self.guard.sleep=advance
        second=self.packet((2107,1322,2405,3080,1986,2286))
        at,_=self.guard.check_goal_packet(42,2,second)
        self.assertEqual(at-completed,20_000_000)

    def _arm_gripper_follow_case(self):
        self.now[0]+=1_000_000_000
        state=dict(zip(NAMES,(1979,1833,2417,2588,1947,2277)))
        self.guard.observe(state)
        first=self.packet((1981,1811,2410,2583,1943,2062))
        at,raw=self.guard.check_goal_packet(42,2,first)
        self.guard.accepted_transmit(at,raw)
        self.now[0]+=33_000_000
        state['gripper']=2269;state['shoulder_pan']=1980
        self.guard.observe(state)
        return state,self.packet((1978,1804,2411,2585,1943,1996))

    def test_gripper_follow_wait_uses_fresh_motor_feedback_without_refreshing_camera_age(self):
        state,candidate=self._arm_gripper_follow_case()
        source_ns=self.guard.source_observation_ns
        self.guard.max_gripper_follow_wait_ns=300_000_000
        def advance(seconds):self.now[0]+=round(seconds*1e9)
        def refresh():
            state['gripper']-=7
            return state.copy()
        self.guard.sleep=advance;self.guard.refresh_feedback=refresh
        at,raw=self.guard.check_goal_packet(42,2,candidate)
        self.assertEqual(raw[5],1996)
        self.assertEqual(state['gripper'],2234)
        self.assertEqual(self.guard.source_observation_ns,source_ns)
        self.assertGreaterEqual(at-self.guard.last_goal_ns,20_000_000)
        self.assertEqual(len([e for e in self.events if e['event']=='legacy_gripper_follow_wait_end']),1)
        self.assertFalse(self.guard.latched)

    def test_gripper_follow_timeout_rejects_without_sending(self):
        state,candidate=self._arm_gripper_follow_case()
        self.guard.max_gripper_follow_wait_ns=300_000_000
        self.guard.sleep=lambda seconds:self.now.__setitem__(0,self.now[0]+round(seconds*1e9))
        self.guard.refresh_feedback=lambda:state.copy()
        with self.assertRaisesRegex(LegacyGuardRejected,'gripper_follow_wait_timeout'):
            self.guard.check_goal_packet(42,2,candidate)
        self.assertEqual(self.guard.goal_packets,1)
        self.assertTrue(self.guard.latched)

    def test_gripper_opening_follow_wait_from_real_rejection(self):
        self.now[0]+=1_000_000_000
        state=dict(zip(NAMES,(2402,2212,1886,2721,2037,1987)))
        self.guard.observe(state)
        first=self.packet((2397,2194,1860,2723,2034,2220))
        at,raw=self.guard.check_goal_packet(42,2,first)
        self.guard.accepted_transmit(at,raw)
        self.now[0]+=33_000_000
        self.guard.observe(state)
        source_ns=self.guard.source_observation_ns
        self.guard.max_gripper_follow_wait_ns=300_000_000
        self.guard.sleep=lambda seconds:self.now.__setitem__(0,self.now[0]+round(seconds*1e9))
        def refresh():
            state['gripper']+=7
            return state.copy()
        self.guard.refresh_feedback=refresh
        second=self.packet((2398,2194,1867,2720,2037,2248))
        at,raw=self.guard.check_goal_packet(42,2,second)
        self.assertEqual(raw[5],2248)
        self.assertEqual(state['gripper'],2008)
        self.assertEqual(self.guard.source_observation_ns,source_ns)
        self.assertEqual(len([e for e in self.events if e['event']=='legacy_gripper_follow_wait_end']),1)
        self.assertFalse(self.guard.latched)

    def test_gripper_reversal_does_not_wait(self):
        self.now[0]+=1_000_000_000
        state=dict(zip(NAMES,(2402,2212,1886,2721,2037,1987)))
        self.guard.observe(state)
        at,raw=self.guard.check_goal_packet(42,2,self.packet((2397,2194,1860,2723,2034,2220)))
        self.guard.accepted_transmit(at,raw)
        self.now[0]+=33_000_000
        self.guard.observe(state)
        self.guard.max_gripper_follow_wait_ns=300_000_000
        self.guard.refresh_feedback=lambda:state.copy()
        # Closing past the current feedback reverses the last accepted opening direction.
        with self.assertRaisesRegex(LegacyGuardRejected,'target_from_feedback_exceeded:gripper'):
            self.guard.check_goal_packet(42,2,self.packet((2398,2194,1867,2720,2037,1740)))
        self.assertFalse(any(e['event']=='legacy_gripper_follow_wait_start' for e in self.events))

    def _arm_gripper_cancel_case(self):
        self.now[0]+=1_000_000_000
        state=dict(zip(NAMES,(1975,2050,2096,2804,1911,2252)))
        self.guard.observe(state)
        first=self.packet((1968,2032,2087,2802,1905,2027))
        at,raw=self.guard.check_goal_packet(42,2,first)
        self.guard.accepted_transmit(at,raw)
        self.now[0]+=33_000_000
        self.guard.observe(state)
        self.now[0]+=121_000_000
        self.guard.max_gripper_cancel_to_feedback_ticks=60
        fresh=state.copy();fresh['gripper']=2223
        self.guard.refresh_feedback=lambda:fresh.copy()
        return state,fresh

    def test_fresh_feedback_cancel_accepts_recorded_case(self):
        self._arm_gripper_cancel_case()
        candidate=self.packet((1983,2052,2085,2791,1903,2262))
        at,raw=self.guard.check_goal_packet(42,2,candidate)
        self.assertEqual(raw[5],2262)
        events=[e for e in self.events if e['event']=='legacy_gripper_cancel_to_feedback']
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['fresh_feedback'],2223)
        self.assertFalse(self.guard.latched)

    def test_cancel_refuses_target_more_than_sixty_from_fresh_feedback(self):
        self._arm_gripper_cancel_case()
        candidate=self.packet((1983,2052,2085,2791,1903,2284))
        with self.assertRaisesRegex(LegacyGuardRejected,'target_from_previous_goal_exceeded:gripper'):
            self.guard.check_goal_packet(42,2,candidate)

    def test_cancel_refuses_source_older_than_two_hundred_ms(self):
        self._arm_gripper_cancel_case()
        self.now[0]+=200_000_001
        candidate=self.packet((1983,2052,2085,2791,1903,2262))
        with self.assertRaisesRegex(LegacyGuardRejected,'target_from_previous_goal_exceeded:gripper'):
            self.guard.check_goal_packet(42,2,candidate)

    def test_cancel_does_not_override_other_joint_violation(self):
        self._arm_gripper_cancel_case()
        candidate=self.packet((2500,2052,2085,2791,1903,2262))
        with self.assertRaises(LegacyGuardRejected):
            self.guard.check_goal_packet(42,2,candidate)
        self.assertFalse(any(e['event']=='legacy_gripper_cancel_to_feedback' for e in self.events))

    def test_rejects_outside_absolute_envelope(self):
        with self.assertRaisesRegex(LegacyGuardRejected,'target_outside_trial_profile:wrist_flex'):
            self.guard.check_goal_packet(42,2,self.packet((2093,1374,2601,3300,1976,2285)))
    def test_rejects_wrist_feedback_beyond_calibrated_upper(self):
        changed=self.state.copy();changed['wrist_flex']=3190
        self.now[0]+=1_000_000_000
        with self.assertRaisesRegex(LegacyGuardRejected,'feedback_outside_trial_profile:wrist_flex'):
            self.guard.observe(changed)

    def test_rejects_stale_feedback(self):
        self.now[0]+=2_000_000_001
        with self.assertRaisesRegex(LegacyGuardRejected,'missing_or_stale_feedback'):
            self.guard.check_goal_packet(42,2,self.packet(tuple(self.state.values())))
    def test_rejects_non_six_packet(self):
        with self.assertRaisesRegex(LegacyGuardRejected,'not_six_raw_goal_positions'):
            self.guard.check_goal_packet(42,2,{1:2093})
    def test_single_tick_in_submillisecond_read_is_quantization_not_rate(self):
        self.now[0]+=509_564
        changed=self.state.copy();changed['shoulder_pan']+=1
        self.guard.observe(changed)
        self.assertFalse(self.guard.latched)

    def test_multi_tick_submillisecond_jump_still_rejected(self):
        self.now[0]+=509_564
        changed=self.state.copy();changed['shoulder_pan']+=5
        with self.assertRaisesRegex(LegacyGuardRejected,'measured_rate_exceeded:shoulder_pan'):
            self.guard.observe(changed)

    def test_rejects_feedback_rate_exceeded(self):
        self.now[0]+=10_000_000
        changed=self.state.copy();changed['elbow_flex']+=100
        with self.assertRaisesRegex(LegacyGuardRejected,'measured_rate_exceeded:elbow_flex'):
            self.guard.observe(changed)

if __name__=='__main__':unittest.main()
