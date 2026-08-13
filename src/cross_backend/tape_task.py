"""Development-only task evaluator. Ground truth stays outside policy observations."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class TaskLimits:
    lift_clearance_m: float = .01
    release_clearance_m: float = .005
    placement_speed_m_s: float = .03
    timeout_ns: int = 20_000_000_000
    def __post_init__(self):
        values=(self.lift_clearance_m,self.release_clearance_m,self.placement_speed_m_s)
        if not all(math.isfinite(v) and v>0 for v in values) or self.release_clearance_m>=self.lift_clearance_m:
            raise ValueError('Invalid development task limits')
        if type(self.timeout_ns) is not int or self.timeout_ns<=0:raise ValueError('Invalid timeout')

@dataclass(frozen=True)
class TapeFacts:
    sim_ns: int
    zone: str  # inside, outside, boundary (full conservative object footprint)
    bottom_clearance_m: float
    speed_m_s: float
    supported_on_mat: bool
    supported_on_table: bool
    both_jaws_contact: bool
    any_jaw_contact: bool
    forbidden_contact: bool = False
    numerical_error: bool = False

class TapeTaskEvaluator:
    def __init__(self,limits=None):
        self.limits=limits or TaskLimits();self.reset()
    def reset(self):
        self.start_ns=None;self.last_ns=None;self.lifted=False;self.unsafe_release=False
        self.status='not_started';self.events=[];self.uncertain_placement=False
    def _event(self,name,stamp):self.events.append({'event':name,'sim_ns':stamp})
    def _stop(self,status,stamp):
        self.status=status;self._event(status,stamp);return status
    def update(self,f):
        if self.status not in ('not_started','running'):return self.status
        valid=(type(f.sim_ns) is int and f.sim_ns>=0 and f.zone in ('inside','outside','boundary')
               and math.isfinite(f.bottom_clearance_m) and math.isfinite(f.speed_m_s) and f.speed_m_s>=0
               and all(type(getattr(f,n)) is bool for n in ('supported_on_mat','supported_on_table','both_jaws_contact','any_jaw_contact','forbidden_contact','numerical_error'))
               and (not f.both_jaws_contact or f.any_jaw_contact))
        if not valid or (self.last_ns is not None and f.sim_ns<=self.last_ns):return self._stop('input_error',self.last_ns)
        self.last_ns=f.sim_ns
        if f.numerical_error:return self._stop('simulation_error',f.sim_ns)
        if f.forbidden_contact:return self._stop('collision',f.sim_ns)
        if self.start_ns is None:
            self.start_ns=f.sim_ns
            if f.zone!='outside' or not f.supported_on_table or f.any_jaw_contact:
                return self._stop('invalid_initial_state',f.sim_ns)
            self.status='running';self._event('started',f.sim_ns)
        if f.sim_ns-self.start_ns>=self.limits.timeout_ns:return self._stop('timeout',f.sim_ns)
        supported=f.supported_on_mat or f.supported_on_table
        self.uncertain_placement=bool(self.lifted and not f.any_jaw_contact and supported and f.zone=='boundary')
        if not self.lifted and f.both_jaws_contact and not supported and f.bottom_clearance_m>=self.limits.lift_clearance_m:
            self.lifted=True;self._event('lift_confirmed',f.sim_ns)
        if self.lifted and not f.any_jaw_contact:
            if not supported and f.bottom_clearance_m>self.limits.release_clearance_m and not self.unsafe_release:
                self.unsafe_release=True;self._event('premature_release',f.sim_ns)
            if supported:
                if self.unsafe_release:return self._stop('drop',f.sim_ns)
                if f.zone=='outside':return self._stop('placement_outside',f.sim_ns)
                if f.zone=='inside' and f.supported_on_mat and f.speed_m_s<=self.limits.placement_speed_m_s:
                    return self._stop('success',f.sim_ns)
        return self.status
    def finish(self):
        if self.status in ('not_started','running'):
            return self._stop('not_started' if self.start_ns is None else 'uncertain' if self.uncertain_placement else 'incomplete',self.last_ns)
        return self.status
