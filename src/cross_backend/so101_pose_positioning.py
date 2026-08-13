"""Fixed, bounded SO101 pose alignment; no policy calls or implicit robot configuration."""

NAMES = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper')
REFERENCE = (2015, 888, 2393, 3051, 2021, 2302)
TARGET = (2082, 1402, 2634, 3126, 1963, 2284)
PERIOD_S = 0.1
STEP_COUNT = max(abs(b-a) for a,b in zip(REFERENCE,TARGET))


class PositioningStop(RuntimeError):
    pass


def waypoint(index):
    if type(index) is not int or not 0 <= index <= STEP_COUNT:
        raise PositioningStop('invalid waypoint index')
    return tuple(a + round((b-a)*index/STEP_COUNT) for a,b in zip(REFERENCE,TARGET))


def verify_plan(calibration):
    if set(calibration) != set(NAMES):
        raise PositioningStop('calibration names changed')
    previous = waypoint(0)
    if previous != REFERENCE or waypoint(STEP_COUNT) != TARGET:
        raise PositioningStop('endpoints changed')
    for index in range(STEP_COUNT+1):
        goal = waypoint(index)
        for name, value, old in zip(NAMES,goal,previous):
            if type(value) is not int or not calibration[name]['range_min'] <= value <= calibration[name]['range_max']:
                raise PositioningStop('waypoint outside calibration: '+name)
            if abs(value-old) > 1:
                raise PositioningStop('waypoint step over 1 tick: '+name)
        previous = goal
    return {'reference_raw':dict(zip(NAMES,REFERENCE)),'target_raw':dict(zip(NAMES,TARGET)),
            'steps':STEP_COUNT,'minimum_period_s':PERIOD_S,'minimum_duration_s':STEP_COUNT*PERIOD_S,
            'max_command_step_raw_ticks_per_joint':1,'max_command_rate_raw_ticks_per_s':10,
            'actual_motor_speed_limit_verified':False}


def restricted_bus_class(base):
    class RestrictedBus(base):
        _permit = None

        def forbidden(self,*args,**kwargs):
            raise PositioningStop('motor write outside fixed allowlist')

        write = sync_write = _sync_write = forbidden
        write_calibration = configure_motors = enable_torque = disable_torque = forbidden
        _enable_torque = _disable_torque = forbidden

        def allowed_write(self,register,name,value):
            if register not in ('Goal_Position','Torque_Enable') or name not in NAMES or type(value) is not int:
                raise PositioningStop('invalid allowed write')
            if register=='Goal_Position' and not self.calibration[name].range_min <= value <= self.calibration[name].range_max:
                raise PositioningStop('goal outside calibrated range')
            if register=='Torque_Enable' and value not in (0,1):
                raise PositioningStop('invalid torque value')
            self._permit=(register,name,value)
            try:
                return super().write(register,name,value,normalize=False,num_retry=0)
            finally:
                self._permit=None

        def _write(self,addr,length,motor_id,value,*,num_retry=0,raise_on_error=True,err_msg=''):
            permit=self._permit
            valid=False
            if permit is not None:
                register,name,expected=permit
                expected_address,expected_length=((42,2) if register=='Goal_Position' else (40,1))
                valid=(addr,length,motor_id,value)==(expected_address,expected_length,self.motors[name].id,expected)
            if not valid or num_retry != 0 or not raise_on_error:
                raise PositioningStop('low-level motor write outside allowlist')
            self._permit=None
            return super()._write(addr,length,motor_id,value,num_retry=0,raise_on_error=True,err_msg=err_msg)

        def disconnect(self,disable_torque=False):
            if disable_torque:self.forbidden()
            return super().disconnect(disable_torque=False)
    return RestrictedBus
