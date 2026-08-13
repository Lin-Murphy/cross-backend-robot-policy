"""Single-use gripper torque recovery and 16-tick probe. No hardware imports."""

from cross_backend.gripper64_microtrial import NAMES, TrialStop

OFFSETS = tuple(range(1, 17)) + tuple(range(15, -1, -1))


def restricted_bus_class(base):
    class RestrictedBus(base):
        _permit = None

        def forbidden(self, *args, **kwargs):
            raise TrialStop('Forbidden motor write/configuration')

        write = sync_write = _sync_write = forbidden
        write_calibration = configure_motors = enable_torque = disable_torque = forbidden
        _enable_torque = _disable_torque = forbidden

        def arm(self, start, clock):
            if hasattr(self, '_start'):
                raise TrialStop('Single-use bus already armed')
            self._start = start
            self._clock = clock
            self._deadline = clock() + 8
            self._index = 0
            self._last_goal = None
            self._phase = 'off'

        def _allowed_write(self, name, value, phase):
            self._permit = (name, value)
            try:
                return super().write(name, 'gripper', value, normalize=False, num_retry=0)
            finally:
                self._permit = None

        def set_initial_goal(self):
            if self._phase != 'off' or self._clock() >= self._deadline:
                raise TrialStop('Initial goal outside phase/deadline')
            self._phase = 'initial_goal_attempted'
            return self._allowed_write('Goal_Position', self._start, self._phase)

        def enable_gripper_only(self):
            if self._phase != 'initial_goal_confirmed' or self._clock() >= self._deadline:
                raise TrialStop('Torque enable outside phase/deadline')
            self._phase = 'enable_attempted'
            return self._allowed_write('Torque_Enable', 1, self._phase)

        def acknowledge_auto_enabled(self):
            if self._phase != 'initial_goal_confirmed' or self._clock() >= self._deadline:
                raise TrialStop('Unexpected automatic torque phase')
            self._phase = 'enabled'

        def write_gripper_goal(self, value):
            if self._phase != 'enabled' or self._index >= len(OFFSETS) or self._clock() >= self._deadline:
                raise TrialStop('Goal outside phase/deadline')
            if self._last_goal is not None and self._clock() - self._last_goal < .1:
                raise TrialStop('Goal too early')
            if type(value) is not int or value != self._start + OFFSETS[self._index]:
                raise TrialStop('Unexpected goal sequence')
            self._index += 1
            self._last_goal = self._clock()
            try:
                return self._allowed_write('Goal_Position', value, self._phase)
            except BaseException:
                self._phase = 'fault'
                raise

        def disable_gripper_only(self):
            if self._phase == 'disabled':
                raise TrialStop('Already disabled')
            self._phase = 'disable_attempted'
            result = self._allowed_write('Torque_Enable', 0, self._phase)
            self._phase = 'disabled'
            return result

        def _write(self, addr, length, motor_id, value, *, num_retry=0, raise_on_error=True, err_msg=''):
            allowed = {(42, 2, 6, self._permit[1]) if self._permit and self._permit[0] == 'Goal_Position' else None,
                       (40, 1, 6, self._permit[1]) if self._permit and self._permit[0] == 'Torque_Enable' else None}
            if (addr, length, motor_id, value) not in allowed or num_retry != 0 or not raise_on_error:
                raise TrialStop('Low-level write outside allowlist')
            self._permit = None
            return super()._write(addr, length, motor_id, value, num_retry=0,
                                  raise_on_error=True, err_msg=err_msg)

        def disconnect(self, disable_torque=False):
            if disable_torque:
                self.forbidden()
            return super().disconnect(disable_torque=False)

    return RestrictedBus


def run_trial(bus, reference, calibration, clock, sleep, record):
    """Return to the start only while feedback and bus checks remain healthy."""
    begin = clock()
    deadline = begin + 8
    observed = []
    acknowledged = []

    def sample():
        start = clock()
        state = {n: bus.read('Present_Position', n, normalize=False, num_retry=0) for n in NAMES}
        end = clock()
        record({'event': 'state', 'read_start_s': start, 'read_end_s': end, 'raw': state})
        if end - start > .05 or end >= deadline:
            raise TrialStop('Slow read/deadline')
        for n, value in state.items():
            if type(value) is not int or not calibration[n]['range_min'] <= value <= calibration[n]['range_max']:
                raise TrialStop('Raw state outside calibration: ' + n)
            if abs(value - reference[n]) > (20 if n == 'gripper' else 4):
                raise TrialStop('Excess displacement: ' + n)
        observed.append(state)
        return state

    def hold_until(target):
        while clock() < target:
            sleep(min(.02, target - clock()))
            sample()

    if sample() != reference:
        raise TrialStop('Posture changed since fresh reference')
    bus.arm(reference['gripper'], clock)
    record({'event': 'write_attempt', 'register': 'Goal_Position', 'raw': reference['gripper'], 'motor_id': 6})
    bus.set_initial_goal()
    record({'event': 'write_acknowledged', 'register': 'Goal_Position', 'raw': reference['gripper'], 'motor_id': 6})
    if bus.read('Goal_Position', 'gripper', normalize=False, num_retry=0) != reference['gripper']:
        raise TrialStop('Initial goal readback mismatch')
    bus._phase = 'initial_goal_confirmed'
    if sample() != reference:
        raise TrialStop('Posture changed before torque enable')
    torque = {n: bus.read('Torque_Enable', n, normalize=False, num_retry=0) for n in NAMES}
    record({'event': 'torque_after_initial_goal', 'raw': torque})
    if any(torque[n] != 0 for n in NAMES[:-1]) or torque['gripper'] not in (0, 1):
        raise TrialStop('Unexpected torque state after initial goal')
    if torque['gripper'] == 1:
        bus.acknowledge_auto_enabled()
        record({'event': 'gripper_torque_enabled_observed_after_goal', 'explicit_enable_write': False})
    else:
        record({'event': 'write_attempt', 'register': 'Torque_Enable', 'raw': 1, 'motor_id': 6})
        bus.enable_gripper_only()
        record({'event': 'write_acknowledged', 'register': 'Torque_Enable', 'raw': 1, 'motor_id': 6})
        bus._phase = 'enabled'
    torque = {n: bus.read('Torque_Enable', n, normalize=False, num_retry=0) for n in NAMES}
    record({'event': 'torque_before_movement', 'raw': torque})
    if any(torque[n] != 0 for n in NAMES[:-1]) or torque['gripper'] != 1:
        raise TrialStop('Torque mismatch before movement')
    if sample() != reference:
        raise TrialStop('Posture changed before movement')
    last = None
    for index, offset in enumerate(OFFSETS):
        if last is not None:
            hold_until(last + .101)
        sample()
        goal = reference['gripper'] + offset
        if not calibration['gripper']['range_min'] <= goal <= calibration['gripper']['range_max']:
            raise TrialStop('Goal outside calibration')
        record({'event': 'write_attempt', 'register': 'Goal_Position', 'raw': goal, 'motor_id': 6,
                'monotonic_s': clock()})
        bus.write_gripper_goal(goal)
        last = clock()
        record({'event': 'write_acknowledged', 'register': 'Goal_Position', 'raw': goal,
                'motor_id': 6, 'monotonic_s': last})
        start = clock()
        actual = bus.read('Goal_Position', 'gripper', normalize=False, num_retry=0)
        record({'event': 'goal_readback', 'expected': goal, 'raw': actual})
        if clock() - start > .05 or actual != goal:
            raise TrialStop('Goal readback slow/mismatch')
        acknowledged.append(goal)
        sample()
        if index == 15:
            hold_until(last + 1)
    hold_until(last + .5)
    final = sample()
    if abs(final['gripper'] - reference['gripper']) > 2:
        raise TrialStop('Return error greater than 2 ticks')
    return {'acknowledged_goals': acknowledged, 'start_raw': reference, 'final_raw': final,
            'gripper_max_observed_excursion_ticks': max(abs(x['gripper'] - reference['gripper']) for x in observed),
            'elapsed_s': clock() - begin, 'motion_response_detected': any(x['gripper'] != reference['gripper'] for x in observed)}


def verify_disabled_preflight(bus, reference, calibration, sleep):
    """Inspect stable power-cycle goals while all torque remains off."""
    previous_goals = None
    for _ in range(3):
        positions = {n: bus.read('Present_Position', n, normalize=False, num_retry=0) for n in NAMES}
        goals = {n: bus.read('Goal_Position', n, normalize=False, num_retry=0) for n in NAMES}
        torque = {n: bus.read('Torque_Enable', n, normalize=False, num_retry=0) for n in NAMES}
        if positions != reference or any(value != 0 for value in torque.values()):
            raise TrialStop('Position or torque changed during disabled preflight')
        if previous_goals is not None and goals != previous_goals:
            raise TrialStop('Existing goals changed during disabled preflight')
        if any(type(positions[n]) is not int or not calibration[n]['range_min'] <= positions[n] <= calibration[n]['range_max'] for n in NAMES):
            raise TrialStop('Present position outside calibration')
        if any(type(goals[n]) is not int for n in NAMES):
            raise TrialStop('Invalid existing goal readback')
        previous_goals = goals
        sleep(.1)
    return {'positions': positions, 'existing_goals': goals, 'torque': torque,
            'stale_goal_outside_calibration': [n for n in NAMES if not calibration[n]['range_min'] <= goals[n] <= calibration[n]['range_max']]}
