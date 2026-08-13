"""Read-only tracking wait before an SO101 policy candidate is offered for dispatch."""
import time
from .so101_dispatch_backend import NAMES,DispatchRejected,calibrated_to_raw


def wait_for_feedback_window(bus, dispatcher, action, source_end_ns, *, max_wait_ns,
                             poll_ns=25_000_000, record, clock=time.monotonic_ns,
                             sleep=time.sleep):
    """Wait for servo feedback; never transform, skip, or send a policy target.

    A live dispatch still repeats every check and may reject if the state changes.
    """
    if type(max_wait_ns) is not int or max_wait_ns<=0 or type(poll_ns) is not int or poll_ns<=0:
        raise ValueError('positive feedback wait and poll periods required')
    raw=calibrated_to_raw(action,dispatcher.calibration)
    limits=dispatcher.limits
    started=clock();previous=None;previous_ns=None
    while True:
        now=clock()
        reasons=[]
        if type(source_end_ns) is not int or source_end_ns>now or \
           now-source_end_ns>limits.max_host_observation_age_ns:
            reasons.append('host_observation_age_unknown_or_exceeded')
        if now-started>max_wait_ns:reasons.append('feedback_window_timeout')
        if dispatcher.last_dispatch_ns is not None and now-dispatcher.last_dispatch_ns>limits.max_period_ns:
            reasons.append('dispatch_period_outside_limits')
        positions={n:bus.read('Present_Position',n,normalize=False,num_retry=0) for n in NAMES}
        goals={n:bus.read('Goal_Position',n,normalize=False,num_retry=0) for n in NAMES}
        torque={n:bus.read('Torque_Enable',n,normalize=False,num_retry=0) for n in NAMES}
        velocity={n:bus.read('Goal_Velocity',n,normalize=False,num_retry=0) for n in NAMES}
        for i,n in enumerate(NAMES):
            if torque[n]!=1:reasons.append('torque_changed:'+n)
            if velocity[n]!=limits.goal_velocity_raw[i]:reasons.append('velocity_changed:'+n)
            if not limits.raw_lower[i]<=positions[n]<=limits.raw_upper[i]:
                reasons.append('feedback_outside_physical_profile:'+n)
            if not limits.raw_lower[i]<=raw[i]<=limits.raw_upper[i]:
                reasons.append('target_outside_physical_profile:'+n)
            if abs(goals[n]-positions[n])>limits.max_feedback_gap_ticks[i]:
                reasons.append('previous_goal_tracking_gap:'+n)
            if previous is not None:
                elapsed_s=(now-previous_ns)/1e9
                if elapsed_s<=0 or abs(positions[n]-previous[n])>limits.max_measured_rate_ticks_s[i]*elapsed_s:
                    reasons.append('measured_rate_exceeded:'+n)
        gaps={n:abs(raw[i]-positions[n]) for i,n in enumerate(NAMES)}
        ready=all(gaps[n]<=limits.max_target_step_ticks[i] for i,n in enumerate(NAMES)) and \
            (dispatcher.last_dispatch_ns is None or now-dispatcher.last_dispatch_ns>=limits.min_period_ns)
        record({'event':'feedback_wait_sample','monotonic_ns':now,'wait_elapsed_ms':(now-started)/1e6,
                'raw_target':raw,'feedback':positions,'previous_goals':goals,'target_feedback_gap':gaps,
                'ready':ready and not reasons,'reasons':reasons,'hardware_dispatched':False})
        if reasons:raise DispatchRejected(reasons)
        if ready:return {'wait_elapsed_ns':now-started,'feedback':positions,'raw_target':raw}
        previous=positions;previous_ns=now
        sleep(poll_ns/1e9)
