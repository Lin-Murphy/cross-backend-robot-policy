"""SO101 bounded raw-goal dispatcher. Requires explicit physical profile; no implicit defaults."""
from dataclasses import dataclass
import math
import time

NAMES=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')


class PartialDispatchError(RuntimeError):
    def __init__(self,message,acknowledged_motors):
        self.acknowledged_motors=tuple(acknowledged_motors)
        super().__init__(message)


class DispatchRejected(RuntimeError):
    def __init__(self,reasons):
        self.reasons=tuple(reasons)
        super().__init__('; '.join(reasons))


@dataclass(frozen=True)
class PhysicalDispatchLimits:
    raw_lower:tuple[int,...]
    raw_upper:tuple[int,...]
    max_target_step_ticks:tuple[int,...]
    max_target_rate_ticks_s:tuple[float,...]
    max_measured_rate_ticks_s:tuple[float,...]
    max_feedback_gap_ticks:tuple[int,...]
    goal_velocity_raw:tuple[int,...]
    min_period_ns:int
    max_period_ns:int
    max_host_observation_age_ns:int
    source:str
    trial_kind:str

    def validate(self,calibration):
        fields=(self.raw_lower,self.raw_upper,self.max_target_step_ticks,
                self.max_target_rate_ticks_s,self.max_measured_rate_ticks_s,
                self.max_feedback_gap_ticks,self.goal_velocity_raw)
        if any(len(x)!=6 for x in fields) or set(calibration)!=set(NAMES):
            raise ValueError('six named calibrated joints required')
        if self.trial_kind not in ('development_pilot','formal_trial') or not self.source:
            raise ValueError('physical limit provenance/trial kind required')
        if any(type(x) is not int or x<=0 for x in (self.min_period_ns,self.max_period_ns,self.max_host_observation_age_ns)):
            raise ValueError('positive timing limits required')
        if self.min_period_ns>self.max_period_ns:
            raise ValueError('invalid period limits')
        for i,n in enumerate(NAMES):
            lo,hi,step,rate,measured_rate,gap,velocity=(f[i] for f in fields)
            c=calibration[n]
            if any(type(x) is not int for x in (lo,hi,step,gap,velocity)) or \
               not c['range_min']<=lo<hi<=c['range_max'] or step<1 or gap<1 or not 1<=velocity<=4095 or \
               not isinstance(rate,(int,float)) or not math.isfinite(rate) or rate<=0 or \
               not isinstance(measured_rate,(int,float)) or not math.isfinite(measured_rate) or measured_rate<=0:
                raise ValueError('invalid physical bounds: '+n)
        return self


def calibrated_to_raw(action,calibration):
    try:
        action=tuple(float(v) for v in action)
    except (TypeError,ValueError,OverflowError):
        raise DispatchRejected(['invalid_six_dimensional_finite_action'])
    if len(action)!=6 or any(not math.isfinite(v) for v in action):
        raise DispatchRejected(['invalid_six_dimensional_finite_action'])
    raw=[]
    for i,n in enumerate(NAMES):
        c=calibration[n]
        value=action[i]
        if i<5:
            mid=(c['range_min']+c['range_max'])/2
            unrounded=value*4095/360+mid
        else:unrounded=value/100*(c['range_max']-c['range_min'])+c['range_min']
        if not c['range_min']<=unrounded<=c['range_max']:
            raise DispatchRejected(['action_outside_calibration:'+n])
        raw.append(int(unrounded))
    return tuple(raw)


class SO101BoundedDispatcher:
    """Separates pre-dispatch rejection, register acknowledgement and measured feedback."""
    def __init__(self,bus,calibration,limits,record,clock=time.monotonic_ns):
        self.bus=bus;self.calibration=calibration;self.limits=limits.validate(calibration)
        self.record=record;self.clock=clock
        self.last_sent=None;self.last_dispatch_ns=None;self.last_feedback=None;self.last_feedback_ns=None
        self.last_goal_write_ns={}
        self.velocity_set=False;self.latched=False

    def _read(self,register):
        return {n:self.bus.read(register,n,normalize=False,num_retry=0) for n in NAMES}

    def _reject(self,reasons,action=None,raw=None):
        self.latched=True
        self.record({'event':'candidate_rejected','reasons':reasons,'action':action,'raw':raw,
                     'hardware_dispatched':False,'monotonic_ns':self.clock()})
        raise DispatchRejected(reasons)

    def configure_velocity(self):
        """Motor-affecting write; call only inside a separately approved real trial."""
        if self.latched or self.velocity_set:raise DispatchRejected(['velocity_configuration_phase_invalid'])
        torque=self._read('Torque_Enable')
        if any(v!=1 for v in torque.values()):self._reject(['torque_not_enabled'])
        try:
            for i,n in enumerate(NAMES):
                value=self.limits.goal_velocity_raw[i]
                self.record({'event':'velocity_write_attempt','motor':n,'raw':value,'monotonic_ns':self.clock()})
                self.bus.write('Goal_Velocity',n,value,normalize=False,num_retry=0)
                actual=self.bus.read('Goal_Velocity',n,normalize=False,num_retry=0)
                self.record({'event':'velocity_write_readback','motor':n,'expected':value,'actual':actual,
                             'monotonic_ns':self.clock()})
                if actual!=value:self._reject(['velocity_readback_mismatch:'+n])
        except BaseException as exc:
            self.latched=True
            self.record({'event':'velocity_configuration_failed','error':repr(exc),'monotonic_ns':self.clock()})
            raise
        self.velocity_set=True

    def dispatch(self,action,host_observation_end_ns):
        now=self.clock()
        if self.latched:raise DispatchRejected(['dispatcher_latched'])
        reasons=[]
        if not self.velocity_set:reasons.append('velocity_limit_not_configured')
        if type(host_observation_end_ns) is not int or host_observation_end_ns>now or \
           now-host_observation_end_ns>self.limits.max_host_observation_age_ns:
            reasons.append('host_observation_age_unknown_or_exceeded')
        if self.last_dispatch_ns is not None:
            period=now-self.last_dispatch_ns
            if not self.limits.min_period_ns<=period<=self.limits.max_period_ns:
                reasons.append('dispatch_period_outside_limits')
        else:period=None
        try:raw=calibrated_to_raw(action,self.calibration)
        except DispatchRejected as exc:self._reject(reasons+list(exc.reasons),list(action),None)
        feedback=self._read('Present_Position')
        goals=self._read('Goal_Position')
        torque=self._read('Torque_Enable')
        speeds=self._read('Goal_Velocity')
        for i,n in enumerate(NAMES):
            c=self.calibration[n];value=feedback[n]
            if torque[n]!=1:reasons.append('torque_changed:'+n)
            if speeds[n]!=self.limits.goal_velocity_raw[i]:reasons.append('velocity_changed:'+n)
            if not c['range_min']<=value<=c['range_max']:reasons.append('feedback_outside_calibration:'+n)
            if not self.limits.raw_lower[i]<=raw[i]<=self.limits.raw_upper[i]:reasons.append('target_outside_physical_profile:'+n)
            if not self.limits.raw_lower[i]<=value<=self.limits.raw_upper[i]:reasons.append('feedback_outside_physical_profile:'+n)
            if abs(goals[n]-value)>self.limits.max_feedback_gap_ticks[i]:reasons.append('previous_goal_tracking_gap:'+n)
            if n in self.last_goal_write_ns and now-self.last_goal_write_ns[n]<self.limits.min_period_ns:
                reasons.append('per_motor_goal_write_too_soon:'+n)
            if self.last_feedback is not None:
                measured_dt=(now-self.last_feedback_ns)/1e9
                if measured_dt<=0 or abs(value-self.last_feedback[n])>self.limits.max_measured_rate_ticks_s[i]*measured_dt:
                    reasons.append('measured_rate_exceeded:'+n)
            if abs(raw[i]-value)>self.limits.max_target_step_ticks[i]:reasons.append('target_from_feedback_step:'+n)
            if self.last_sent is not None and abs(raw[i]-self.last_sent[i])>self.limits.max_target_step_ticks[i]:
                reasons.append('target_from_previous_goal_step:'+n)
            if period is not None and self.last_sent is not None and \
               abs(raw[i]-self.last_sent[i])>self.limits.max_target_rate_ticks_s[i]*period/1e9:
                reasons.append('target_rate_exceeded:'+n)
        if reasons:self._reject(reasons,list(action),raw)
        self.record({'event':'candidate_accepted','action':list(action),'raw':raw,'feedback_before':feedback,
                     'goal_before':goals,'host_observation_end_ns':host_observation_end_ns,'monotonic_ns':now,
                     'hardware_dispatched':False})
        acknowledged=[]
        try:
            for i,n in enumerate(NAMES):
                goal_write_ns=self.clock()
                self.record({'event':'goal_write_attempt','motor':n,'raw':raw[i],'monotonic_ns':goal_write_ns})
                self.last_goal_write_ns[n]=goal_write_ns
                self.bus.write('Goal_Position',n,raw[i],normalize=False,num_retry=0)
                acknowledged.append(n)
                actual=self.bus.read('Goal_Position',n,normalize=False,num_retry=0)
                self.record({'event':'goal_write_readback','motor':n,'expected':raw[i],'actual':actual,
                             'monotonic_ns':self.clock()})
                if actual!=raw[i]:raise RuntimeError('goal_readback_mismatch:'+n)
        except BaseException as exc:
            self.latched=True
            self.record({'event':'partial_dispatch_failure','acknowledged_motors':acknowledged,
                         'error':repr(exc),'monotonic_ns':self.clock()})
            raise PartialDispatchError(str(exc),acknowledged) from exc
        self.last_sent=raw;self.last_dispatch_ns=self.clock()
        after=self._read('Present_Position')
        self.last_feedback=after;self.last_feedback_ns=self.clock()
        self.record({'event':'dispatch_complete','raw':raw,'feedback_after':after,
                     'dispatch_monotonic_ns':self.last_dispatch_ns,'hardware_dispatched':True})
        return {'raw':raw,'feedback_after':after,'dispatch_monotonic_ns':self.last_dispatch_ns}

    def stop_hold(self):
        """Best-effort hold at measured position; torque stays on to support the arm."""
        self.latched=True
        positions=self._read('Present_Position')
        torque=self._read('Torque_Enable')
        written=[]
        for i,n in enumerate(NAMES):
            value=positions[n]
            if torque[n]!=1 or not self.limits.raw_lower[i]<=value<=self.limits.raw_upper[i]:
                continue
            self.record({'event':'stop_hold_attempt','motor':n,'raw':value,'monotonic_ns':self.clock()})
            self.bus.write('Goal_Position',n,value,normalize=False,num_retry=0)
            actual=self.bus.read('Goal_Position',n,normalize=False,num_retry=0)
            self.record({'event':'stop_hold_readback','motor':n,'expected':value,'actual':actual,
                         'monotonic_ns':self.clock()})
            if actual!=value:raise RuntimeError('stop_hold_readback_mismatch:'+n)
            written.append(n)
        return {'held_motors':written,'positions':positions,'torque':torque}


def restricted_policy_bus_class(base,limits):
    """Allow only this pilot's bounded raw position and speed register writes."""
    class RestrictedBus(base):
        _permit=None
        def forbidden(self,*args,**kwargs):raise DispatchRejected(['motor_write_outside_policy_allowlist'])
        sync_write=_sync_write=forbidden
        write_calibration=configure_motors=enable_torque=disable_torque=forbidden
        _enable_torque=_disable_torque=forbidden
        def write(self,register,name,value,normalize=False,num_retry=0):
            if name not in NAMES or type(value) is not int or normalize or num_retry!=0:
                self.forbidden()
            i=NAMES.index(name)
            if register=='Goal_Position':
                valid=limits.raw_lower[i]<=value<=limits.raw_upper[i]
            elif register=='Goal_Velocity':
                valid=value==limits.goal_velocity_raw[i]
            else:valid=False
            if not valid:self.forbidden()
            self._permit=(register,name,value)
            try:return super().write(register,name,value,normalize=False,num_retry=0)
            finally:self._permit=None
        def _write(self,addr,length,motor_id,value,*,num_retry=0,raise_on_error=True,err_msg=''):
            permit=self._permit
            valid=False
            if permit is not None:
                register,name,expected=permit
                valid=(addr,length,motor_id,value)==(42 if register=='Goal_Position' else 46,2,
                                                         self.motors[name].id,expected)
            if not valid or num_retry!=0 or not raise_on_error:self.forbidden()
            self._permit=None
            return super()._write(addr,length,motor_id,value,num_retry=0,raise_on_error=True,err_msg=err_msg)
        def disconnect(self,disable_torque=False):
            if disable_torque:self.forbidden()
            return super().disconnect(disable_torque=False)
    return RestrictedBus
