"""Shared degree-unit *candidate* checks. No hardware backend or dispatch API.

All limits are explicit, per joint; synthetic profiles are only process fixtures.
Accepted candidates form a virtual command history, never a sent-command history.
Timestamps require one named clock session; persisted clocks cannot be rebased.
"""
from dataclasses import dataclass
import numpy as np
from .move_pot_policy import CAMERAS, JOINTS, UNITS
from .safety import ActionSafetyError, NoDispatch


class DegreeGateRejected(ActionSafetyError):
    def __init__(self, reasons):
        self.reasons = reasons
        super().__init__('; '.join(reasons))


@dataclass(frozen=True)
class DegreeLimits:
    lower: tuple
    upper: tuple
    max_step: tuple
    max_speed: tuple  # deg/s for first five, calibrated units/s for gripper
    min_period_ns: int
    max_period_ns: int
    max_state_age_ns: int
    max_camera_age_ns: int
    max_action_age_ns: int  # age of oldest source capture, not inference end
    provenance: str
    synthetic: bool  # explicit label, no default physical thresholds


def vector(value):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (6,) or not np.isfinite(a).all():
        raise ValueError('expected_six_finite_values')
    return a.copy()


def stamp(value):
    return type(value) is int and value >= 0


class DegreeCandidateGate:
    """Reject, latch, and never clip. reset requires a new timing baseline.

    First candidate is referenced to current measured state and arm time. Later
    candidates are bounded against BOTH current state and last virtual candidate.
    Actual period is checked before using it as a velocity budget.
    """
    def __init__(self, limits, clock_session, clock_kind):
        self.limits = limits
        self.clock_session = clock_session
        self.clock_kind = clock_kind
        self.sink = NoDispatch()
        self.reset(None)

    def reset(self, armed_ns):
        self.armed_ns = armed_ns
        self.last_ns = None
        self.last_action = None
        self.last_captures = {}
        self.latched = False

    def check(self, action, source, current, start_ns, end_ns, now_ns):
        reasons = []
        def require(ok, reason):
            if not ok:
                reasons.append(reason)
        if self.latched:
            raise DegreeGateRejected(['gate_latched_reset_required'])
        cfg = self.limits
        try:
            require(cfg is not None, 'missing_limits')
            if cfg is not None:
                low, high, step, speed = map(vector, (cfg.lower, cfg.upper, cfg.max_step, cfg.max_speed))
                require(bool(np.all(low < high) and np.all(step > 0) and np.all(speed > 0)), 'invalid_limits')
                require(low[5] >= 0 and high[5] <= 100, 'invalid_gripper_bounds')
                periods = [cfg.min_period_ns, cfg.max_period_ns, cfg.max_state_age_ns,
                           cfg.max_camera_age_ns, cfg.max_action_age_ns]
                require(all(stamp(x) and x > 0 for x in periods) and cfg.min_period_ns <= cfg.max_period_ns,
                        'invalid_time_limits')
                require(bool(cfg.provenance) and type(cfg.synthetic) is bool, 'missing_limit_provenance')
            target, measured, original_state = map(vector, (action, current['state'], source['state']))
            for label, data in [('source', source), ('current', current)]:
                require(tuple(data.get('joint_names', ())) == JOINTS, label + '_joint_order')
                require(tuple(data.get('joint_units', ())) == UNITS, label + '_units')
                require(bool(self.clock_session) and data.get('clock_session') == self.clock_session,
                        label + '_clock_session_unknown_or_mismatch')
                require(self.clock_kind in ('synthetic', 'monotonic') and data.get('clock_kind') == self.clock_kind,
                        label + '_clock_kind_unknown_or_mismatch')
            require(bool(source.get('observation_id')), 'missing_observation_id')
            frames = source.get('frame_ids', {})
            require(set(frames) == set(CAMERAS) and all(frames.values()), 'missing_camera_frame_identity')
            cameras = source.get('camera_capture_ns', {})
            require(set(cameras) == set(CAMERAS), 'missing_camera_timestamps')
            captures = {'source_state': source.get('state_capture_ns'),
                        'current_state': current.get('state_capture_ns'),
                        **{key: cameras.get(key) for key in CAMERAS}}
            for key, value in captures.items():
                require(stamp(value), key + '_capture_unknown_or_invalid')
            previous_ns = self.last_ns if self.last_ns is not None else self.armed_ns
            require(all(stamp(v) for v in (now_ns, start_ns, end_ns, previous_ns)), 'unknown_or_invalid_control_time')
            # Do not subtract unknown timestamps, even when other checks failed.
            if all(stamp(v) for v in (*captures.values(), now_ns, start_ns, end_ns, previous_ns)):
                require(start_ns <= end_ns <= now_ns, 'prediction_time_future_or_reversed')
                require(all(captures[k] <= start_ns for k in captures if k != 'current_state'),
                        'source_capture_after_prediction_start')
                require(captures['source_state'] <= captures['current_state'] <= now_ns,
                        'current_state_future_or_older_than_source')
                require(all(value >= self.last_captures.get(k, 0) for k, value in captures.items()),
                        'capture_time_reversed')
                if cfg is not None and not reasons:
                    dt = now_ns - previous_ns
                    require(cfg.min_period_ns <= dt <= cfg.max_period_ns, 'control_period_outside_limits')
                    require(now_ns - captures['current_state'] <= cfg.max_state_age_ns, 'current_state_stale')
                    require(all(now_ns - captures[k] <= cfg.max_camera_age_ns for k in CAMERAS), 'camera_source_stale')
                    age = now_ns - min(v for k, v in captures.items() if k != 'current_state')
                    require(age <= cfg.max_action_age_ns, 'action_source_stale')
                    if not reasons:
                        for label, value in [('target', target), ('current_state', measured), ('source_state', original_state)]:
                            require(bool(np.all((value >= low) & (value <= high))), label + '_outside_position_limits')
                        refs = [('current_state', measured)]
                        if self.last_action is not None:
                            refs.append(('previous_candidate', self.last_action))
                        for label, ref in refs:
                            delta = np.abs(target - ref)
                            require(bool(np.all(delta <= step)), label + '_step_exceeded')
                            require(bool(np.all(delta <= speed * (dt / 1e9))), label + '_speed_exceeded')
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as exc:
            reasons.append('malformed_input_or_limits:' + str(exc))
        if reasons:
            self.latched = True
            raise DegreeGateRejected(reasons)
        self.last_action = target.copy()
        self.last_ns = now_ns
        self.last_captures = captures.copy()
        return {'status': 'candidate_checks_passed_no_dispatch', 'period_ns': dt,
                'current_state_age_ns': now_ns - captures['current_state'],
                'camera_ages_ns': {k: now_ns - captures[k] for k in CAMERAS},
                'action_source_age_ns': age, 'synthetic_limits': cfg.synthetic,
                'limit_provenance': cfg.provenance, 'clock_kind': self.clock_kind,
                'clock_session': self.clock_session, 'evaluation_ns': now_ns,
                'reference': 'current_state_and_previous_candidate' if len(refs) == 2 else 'current_state_first_candidate'}


def pop_checked(queue, gate, current, now_ns):
    """Persist evaluation context before gate exceptions; queue owns rejection latch."""
    def validate(action, event):
        event['evaluation_context'] = {'evaluation_ns': now_ns, 'clock_kind': gate.clock_kind,
                                       'clock_session': gate.clock_session, 'current': current,
                                       'armed_ns': gate.armed_ns, 'previous_candidate_ns': gate.last_ns,
                                       'synthetic_limits': getattr(gate.limits, 'synthetic', None),
                                       'limit_provenance': getattr(gate.limits, 'provenance', None)}
        event['degree_gate'] = gate.check(action, event['source'], current,
                                         event['prediction_start_ns'], event['prediction_end_ns'], now_ns)
    return queue.pop(context_validator=validate)
