"""Trial-specific guard for the legacy SO101 sync-write path; no robot imports."""
from dataclasses import dataclass
import math
import time

NAMES=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')


class LegacyGuardRejected(RuntimeError):
    def __init__(self,reasons):
        self.reasons=tuple(reasons)
        super().__init__('; '.join(reasons))


@dataclass(frozen=True)
class LegacyTrialProfile:
    raw_lower:tuple[int,...]
    raw_upper:tuple[int,...]
    max_feedback_raw:tuple[int,...]
    max_from_feedback:tuple[int,...]
    max_from_last_target:tuple[int,...]
    max_measured_rate_ticks_s:tuple[int,...]
    max_observation_age_ns:int
    min_goal_period_ns:int
    max_goal_period_ns:int
    max_policy_elapsed_ns:int
    max_goal_packets:int
    source:str

    def validate(self):
        vectors=(self.raw_lower,self.raw_upper,self.max_feedback_raw,self.max_from_feedback,
                 self.max_from_last_target,self.max_measured_rate_ticks_s)
        if any(len(v)!=6 for v in vectors):raise ValueError('six joint profile required')
        if any(type(x) is not int or x<=0 for v in vectors[3:] for x in v):
            raise ValueError('positive joint guards required')
        if any(type(x) is not int or not 0<=x<self.max_feedback_raw[i]<=self.raw_upper[i]<=4095
               for i,x in enumerate(self.raw_lower)):
            raise ValueError('invalid raw envelopes')
        if any(type(x) is not int or x<=0 for x in (self.max_observation_age_ns,self.min_goal_period_ns,
                                                     self.max_goal_period_ns,self.max_policy_elapsed_ns)):
            raise ValueError('positive clock guards required')
        if type(self.max_goal_packets) is not int or self.max_goal_packets<1 or self.min_goal_period_ns>self.max_goal_period_ns or not self.source:
            raise ValueError('invalid period/provenance')
        return self


class LegacySyncGuard:
    """Validate raw sync packet against the most recent existing robot observation."""
    def __init__(self,profile,record,clock=time.perf_counter_ns,sleep=time.sleep,
                 refresh_feedback=None,max_gripper_follow_wait_ns=0,
                 max_gripper_cancel_to_feedback_ticks=0):
        self.profile=profile.validate();self.record=record;self.clock=clock;self.sleep=sleep
        if type(max_gripper_follow_wait_ns) is not int or not 0<=max_gripper_follow_wait_ns<=500_000_000:
            raise ValueError('invalid gripper follow wait')
        if type(max_gripper_cancel_to_feedback_ticks) is not int or not 0<=max_gripper_cancel_to_feedback_ticks<=60:
            raise ValueError('invalid gripper feedback-near cancel limit')
        self.refresh_feedback=refresh_feedback;self.max_gripper_follow_wait_ns=max_gripper_follow_wait_ns
        self.max_gripper_cancel_to_feedback_ticks=max_gripper_cancel_to_feedback_ticks
        self.feedback=None;self.feedback_ns=None;self.source_observation_ns=None
        self.last_target=None;self.last_goal_ns=None;self.first_goal_ns=None;self.goal_packets=0;self.latched=False

    def observe(self,raw,observed_ns=None,source_observation=True):
        if set(raw)!=set(NAMES) or any(type(raw[n]) is not int for n in NAMES):
            raise ValueError('six raw feedback values required')
        now=self.clock() if observed_ns is None else observed_ns;reasons=[]
        for i,n in enumerate(NAMES):
            if not self.profile.raw_lower[i]<=raw[n]<=self.profile.max_feedback_raw[i]:
                reasons.append('feedback_outside_trial_profile:'+n)
            if self.feedback is not None:
                elapsed=(now-self.feedback_ns)/1e9
                if elapsed<=0 or abs(raw[n]-self.feedback[n])>1+self.profile.max_measured_rate_ticks_s[i]*elapsed:
                    reasons.append('measured_rate_exceeded:'+n)
        self.record({'event':'legacy_raw_feedback','raw':dict(raw),'host_ns':now,
                     'source_observation':source_observation,'reasons':reasons})
        if reasons:self.reject(reasons)
        self.feedback=dict(raw);self.feedback_ns=now
        if source_observation:self.source_observation_ns=now

    def reject(self,reasons):
        self.latched=True;self.record({'event':'legacy_trial_guard_rejected','reasons':list(reasons),
                                      'host_ns':self.clock(),'hardware_dispatched':False})
        raise LegacyGuardRejected(reasons)

    def _goal_reasons(self,now,addr,length,ids_values):
        p=self.profile;reasons=[];raw=None
        if self.latched:reasons.append('guard_latched')
        if (addr,length)!=(42,2) or set(ids_values)!=set(range(1,7)) or \
           any(type(v) is not int for v in ids_values.values()):
            reasons.append('not_six_raw_goal_positions')
        if self.feedback is None or self.source_observation_ns is None or \
           now-self.source_observation_ns>p.max_observation_age_ns:
            reasons.append('missing_or_stale_feedback')
        if self.last_goal_ns is not None and not p.min_goal_period_ns<=now-self.last_goal_ns<=p.max_goal_period_ns:
            reasons.append('goal_period_outside_trial_profile')
        if self.first_goal_ns is not None and now-self.first_goal_ns>=p.max_policy_elapsed_ns:
            reasons.append('policy_duration_exceeded')
        if self.goal_packets>=p.max_goal_packets:reasons.append('goal_packet_count_exceeded')
        if not reasons:
            raw=tuple(ids_values[i] for i in range(1,7))
            for i,n in enumerate(NAMES):
                if not p.raw_lower[i]<=raw[i]<=p.raw_upper[i]:
                    reasons.append('target_outside_trial_profile:'+n)
                if abs(raw[i]-self.feedback[n])>p.max_from_feedback[i]:
                    reasons.append('target_from_feedback_exceeded:'+n)
                if self.last_target is not None and abs(raw[i]-self.last_target[i])>p.max_from_last_target[i]:
                    reasons.append('target_from_previous_goal_exceeded:'+n)
        return reasons,raw

    def check_goal_packet(self,addr,length,ids_values):
        now=self.clock();p=self.profile
        if self.last_goal_ns is not None and not self.latched:
            gap=now-self.last_goal_ns
            if 0<=gap<p.min_goal_period_ns:
                requested_ns=p.min_goal_period_ns-gap
                self.sleep(requested_ns/1e9)
                now=self.clock()
                self.record({'event':'legacy_goal_paced','requested_ns':requested_ns,
                             'actual_period_ns':now-self.last_goal_ns,'host_ns':now})
        reasons,raw=self._goal_reasons(now,addr,length,ids_values)
        # At a policy chunk boundary, a target-to-target jump can merely
        # cancel a lagging gripper command. Re-read the encoder before allowing
        # this narrow exception; the source frame itself must be recent.
        can_check_cancel=(reasons==['target_from_previous_goal_exceeded:gripper'] and
            self.max_gripper_cancel_to_feedback_ticks>0 and raw is not None and
            self.last_target is not None and self.refresh_feedback is not None and
            0<=now-self.source_observation_ns<=200_000_000)
        if can_check_cancel:
            updated=self.refresh_feedback()
            self.observe(updated,observed_ns=self.clock(),source_observation=False)
            now=self.clock()
            reasons,raw=self._goal_reasons(now,addr,length,ids_values)
            feedback_gripper=self.feedback['gripper']
            previous_gripper=self.last_target[5]
            if (reasons==['target_from_previous_goal_exceeded:gripper'] and
                0<=now-self.source_observation_ns<=200_000_000 and
                abs(raw[5]-feedback_gripper)<=self.max_gripper_cancel_to_feedback_ticks and
                abs(previous_gripper-feedback_gripper)<=p.max_from_feedback[5]):
                self.record({'event':'legacy_gripper_cancel_to_feedback','host_ns':now,
                             'raw_candidate':raw[5],'fresh_feedback':feedback_gripper,
                             'previous_target':previous_gripper,
                             'source_age_ns':now-self.source_observation_ns,
                             'maximum_target_feedback_ticks':self.max_gripper_cancel_to_feedback_ticks})
                reasons=[]
        only_gripper_gap=reasons==['target_from_feedback_exceeded:gripper']
        # Motor feedback can lag an accepted target while opening or closing.
        # Wait only when the new target continues in the same direction and
        # the previous target still satisfies the feedback-gap limit.
        previous_gripper=self.last_target[5] if self.last_target is not None else None
        feedback_gripper=self.feedback['gripper'] if self.feedback is not None else None
        same_direction=(raw is not None and previous_gripper is not None and feedback_gripper is not None and
                        (raw[5]-previous_gripper)*(previous_gripper-feedback_gripper)>0)
        can_wait=(only_gripper_gap and self.max_gripper_follow_wait_ns>0 and
                  self.refresh_feedback is not None and same_direction and
                  abs(previous_gripper-feedback_gripper)<=p.max_from_feedback[5])
        if can_wait:
            deadline=now+self.max_gripper_follow_wait_ns
            self.record({'event':'legacy_gripper_follow_wait_start','host_ns':now,
                         'raw_candidate':raw,'feedback':dict(self.feedback),
                         'previous_target':self.last_target[5],'deadline_ns':deadline})
            reads=0
            while reasons==['target_from_feedback_exceeded:gripper'] and now<deadline:
                self.sleep(min(20_000_000,deadline-now)/1e9)
                updated=self.refresh_feedback()
                self.observe(updated,observed_ns=self.clock(),source_observation=False)
                reads+=1;now=self.clock()
                reasons,raw=self._goal_reasons(now,addr,length,ids_values)
            if reasons==['target_from_feedback_exceeded:gripper']:
                reasons=['gripper_follow_wait_timeout']
            self.record({'event':'legacy_gripper_follow_wait_end','host_ns':now,
                         'reads':reads,'reasons':list(reasons),'feedback':dict(self.feedback)})
        self.record({'event':'legacy_goal_candidate','raw_ids_values':dict(ids_values),'host_ns':now,
                     'feedback':self.feedback,'reasons':reasons,'hardware_dispatched':False})
        if reasons:self.reject(reasons)
        return now,raw

    def accepted_transmit(self,now,raw):
        completed_ns=self.clock()
        if self.first_goal_ns is None:self.first_goal_ns=completed_ns
        self.last_goal_ns=completed_ns;self.last_target=raw;self.goal_packets+=1
        self.record({'event':'legacy_goal_packet_transmitted','raw':raw,'host_ns':completed_ns,
                     'transport_returned':True,'per_motor_acknowledged':False})
