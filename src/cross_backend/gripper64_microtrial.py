"""Exact single-gripper microtrial, independent of policy execution. No hardware imports."""
NAMES=('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')
OFFSETS=tuple(range(1,65))+tuple(range(63,-1,-1))


class TrialStop(RuntimeError): pass


def restricted_bus_class(base):
    class RestrictedBus(base):
        _permit=None
        def forbidden(self,*args,**kwargs): raise TrialStop('Forbidden motor write/configuration')
        write=sync_write=_sync_write=forbidden
        write_calibration=configure_motors=enable_torque=disable_torque=forbidden
        _enable_torque=_disable_torque=forbidden
        def arm_trial(self,start,clock):
            if hasattr(self,'_trial_start'):raise TrialStop('Single-use bus already armed')
            self._trial_start=start;self._clock=clock;self._deadline=clock()+20
            self._index=0;self._last_write=None
        def write_gripper_goal(self,value):
            if not hasattr(self,'_trial_start') or self._index>=len(OFFSETS):raise TrialStop('Not armed/exhausted')
            now=self._clock()
            if now>=self._deadline:raise TrialStop('Write deadline exceeded')
            if self._last_write is not None and now-self._last_write<.1:raise TrialStop('Write too early')
            if type(value) is not int or value!=self._trial_start+OFFSETS[self._index]:raise TrialStop('Unexpected goal')
            self._index+=1;self._last_write=now;self._permit=value
            try:return super().write('Goal_Position','gripper',value,normalize=False,num_retry=0)
            except BaseException:
                self._index=len(OFFSETS);raise
            finally:self._permit=None
        def _write(self,addr,length,motor_id,value,*,num_retry=0,raise_on_error=True,err_msg=''):
            if self._permit is None or (addr,length,motor_id,value)!=(42,2,6,self._permit) or num_retry!=0 or not raise_on_error:
                raise TrialStop('Low-level write outside allowlist')
            self._permit=None
            return super()._write(addr,length,motor_id,value,num_retry=0,raise_on_error=True,err_msg=err_msg)
        def disconnect(self,disable_torque=False):
            if disable_torque:self.forbidden()
            return super().disconnect(disable_torque=False)
    return RestrictedBus


def run_trial(bus,reference,calibration,clock,sleep,record):
    """Stops on every fault, with no recovery writes or torque changes."""
    begin=clock();deadline=begin+20;observations=[];writes=[]
    def check_time():
        if clock()>=deadline:raise TrialStop('Trial exceeded twenty seconds')
    def sample():
        check_time();start=clock()
        state={n:bus.read('Present_Position',n,normalize=False,num_retry=0) for n in NAMES}
        end=clock();record({'event':'state','read_start_s':start,'read_end_s':end,'raw':state})
        observations.append(state)
        if end-start>.05:raise TrialStop('State read exceeded 50 ms')
        check_time()
        for n,v in state.items():
            if type(v) is not int or not calibration[n]['range_min']<=v<=calibration[n]['range_max']:raise TrialStop('Raw state out of range: '+n)
            if abs(v-reference[n])>(68 if n=='gripper' else 4):raise TrialStop('Excess displacement: '+n)
        return state
    initial=sample()
    if initial!=reference:raise TrialStop('Posture changed since approved reference')
    bus.arm_trial(initial['gripper'],clock)
    last=None
    def monitor_until(target):
        while clock()<target:
            sleep(min(.02,target-clock()));sample()
    for index,offset in enumerate(OFFSETS):
        if last is not None:monitor_until(last+.101)
        before=sample();check_time()
        goal=initial['gripper']+offset
        if not calibration['gripper']['range_min']<=goal<=calibration['gripper']['range_max']:raise TrialStop('Goal outside calibration')
        sent=clock();record({'event':'write_attempt','motor':'gripper','motor_id':6,'register':'Goal_Position','raw':goal,'monotonic_s':sent})
        bus.write_gripper_goal(goal)
        last=clock();writes.append(goal)
        record({'event':'write_acknowledged','raw':goal,'monotonic_s':last})
        read_start=clock()
        readback=bus.read('Goal_Position','gripper',normalize=False,num_retry=0)
        record({'event':'goal_readback','expected':goal,'raw':readback,'monotonic_s':clock()})
        if clock()-read_start>.05:raise TrialStop('Goal read exceeded 50 ms')
        if type(readback) is not int or readback!=goal:raise TrialStop('Goal register mismatch')
        sample()
        if index==63:monitor_until(last+1.)
    monitor_until(last+.5)
    final=sample();error=abs(final['gripper']-initial['gripper'])
    if error>2:raise TrialStop('Return error greater than two ticks')
    return {'status':'passed_bounded_return_check','acknowledged_goals':writes,'start_raw':initial,'final_raw':final,
            'gripper_max_observed_excursion_ticks':max(abs(r['gripper']-initial['gripper']) for r in observations),
            'return_error_ticks':error,'other_joint_max_drift_ticks':{n:max(abs(r[n]-initial[n]) for r in observations) for n in NAMES[:-1]},
            'elapsed_s':clock()-begin,'scope':'single gripper, single pose, not six-joint safety or policy acceptance'}


def verify_stationary_preflight(bus, reference, calibration, sleep):
    """Static tracking error is recorded, not mistaken for observed joint drift."""
    previous_goals=None
    for _ in range(3):
        positions={n:bus.read('Present_Position',n,normalize=False,num_retry=0) for n in NAMES}
        goals={n:bus.read('Goal_Position',n,normalize=False,num_retry=0) for n in NAMES}
        if positions!=reference:raise TrialStop('Actual posture changed')
        if previous_goals is not None and goals!=previous_goals:raise TrialStop('Existing targets changed during preflight')
        for n,g in goals.items():
            if type(g) is not int or not calibration[n]['range_min']<=g<=calibration[n]['range_max']:raise TrialStop('Existing goal outside calibration')
        previous_goals=goals
        sleep(.1)
    return {'positions':positions,'existing_goals':goals,'static_goal_minus_position':{n:goals[n]-positions[n] for n in NAMES}}
