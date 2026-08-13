"""Discrete-event simulation scheduling around the existing provenance queue.

Predictor host compute and modeled simulation availability are separate clocks.
Only one completed-but-unavailable job is stored; no physics/hardware access here.
"""
from collections import deque
import time
import numpy as np
from .move_pot_policy import MovePotChunk

class SimChunkSchedule:
    def __init__(self,queue,predictor,*,latency_ns,observation_delay_ns=0,observation_delay_frames=None,execute_steps=50,prefetch_remaining=5,clock_id='sim-episode'):
        if latency_ns!='measured_host' and (type(latency_ns) is not int or latency_ns<0):raise ValueError('Invalid latency mode')
        for value in (observation_delay_ns,execute_steps,prefetch_remaining):
            if type(value) is not int or value<0:raise ValueError('Nonnegative integer configuration required')
        if execute_steps<1 or prefetch_remaining>=execute_steps or not clock_id:raise ValueError('Invalid horizon/clock')
        if observation_delay_frames is not None and (type(observation_delay_frames) is not int or observation_delay_frames<0 or observation_delay_ns!=0):raise ValueError('Frame delay must be a nonnegative integer, exclusive with time delay')
        self.delay_frames=observation_delay_frames;self.capture_count=0
        self.queue=queue;self.predictor=predictor;self.latency_ns=latency_ns;self.delay=observation_delay_ns
        self.horizon=execute_steps;self.prefetch_remaining=prefetch_remaining;self.clock_id=clock_id
        self.buffer=deque();self.job=None;self.last_tick=None;self.next_job=0;self.events=[];self.failed=False
    def capture(self,stamp_ns,payload):
        if type(stamp_ns) is not int or stamp_ns<0 or (self.buffer and stamp_ns<=self.buffer[-1][0]):raise ValueError('Invalid/out-of-order sim capture')
        self.buffer.append((stamp_ns,payload,self.capture_count));self.capture_count+=1
    def _submit(self,now):
        cutoff=now-self.delay
        eligible=[item for item in self.buffer if item[0]<=cutoff]
        if not eligible or (self.delay_frames is not None and len(eligible)<=self.delay_frames):
            self.events.append({'event':'waiting_for_observation_history','sim_ns':now});return
        selected=eligible[-1] if self.delay_frames is None else eligible[-1-self.delay_frames]
        stamp,payload,sequence=selected
        latest=[item for item in self.buffer if item[0]<=now][-1]
        while len(self.buffer)>1 and self.buffer[1][0]<=stamp:self.buffer.popleft()
        start=time.perf_counter_ns()
        predicted=self.predictor(payload)
        original_chunk=predicted if isinstance(predicted,MovePotChunk) else None
        actions=np.asarray(original_chunk.actions if original_chunk is not None else predicted,dtype=float)
        end=time.perf_counter_ns()
        if actions.ndim!=2 or actions.shape[1]!=6 or len(actions)<self.horizon or not np.isfinite(actions).all():
            self.failed=True;self.events.append({'event':'prediction_rejected','sim_ns':now,'reason':'invalid_chunk'});raise ValueError('Invalid predicted chunk')
        cost=end-start if self.latency_ns=='measured_host' else self.latency_ns
        source={'observation_id':payload['observation_id'],'clock_domain':'simulation','clock_id':self.clock_id,'capture_sim_ns':stamp,'request_sim_ns':now,'available_sim_ns':now+cost,'job_id':self.next_job,'observation_sequence':sequence,'latest_observation_sequence':latest[2],'realized_delay_frames':latest[2]-sequence,'delay_mode':'minimum_age_ns' if self.delay_frames is None else 'preceding_frames','configured_delay_frames':self.delay_frames}
        self.next_job+=1
        if original_chunk is not None:
            source['model_source']=original_chunk.source
            chunk=MovePotChunk(actions,source,original_chunk.prediction_start_ns,original_chunk.prediction_end_ns,original_chunk.forward_ms,original_chunk.processing_and_forward_ms)
        else:
            source['prediction_timing_scope']='callback_envelope_not_model_forward'
            chunk=MovePotChunk(actions,source,start,end,(end-start)/1e6,(end-start)/1e6)
        self.job=chunk
        self.events.append({'event':'prediction_scheduled',**source,'modeled_latency_ns':cost,'latency_mode':self.latency_ns,'host_compute_ns':end-start,'latency_semantics':'measured_host_callback_injected_once' if self.latency_ns=='measured_host' else 'declared_sim_availability; host_compute_not_added_again'})
    def tick(self,now,validator):
        if self.failed:raise ValueError('Scheduler latched after failure')
        if type(now) is not int or now<0 or (self.last_tick is not None and now<=self.last_tick):raise ValueError('Sim ticks must increase')
        self.last_tick=now
        try:
            if not self.queue.pending and self.job is None:self._submit(now)
            if not self.queue.pending and self.job is not None and now>=self.job.source['available_sim_ns']:
                self.queue.enqueue(self.job,self.horizon);self.job=None
            if not self.queue.pending:
                self.events.append({'event':'hold_existing_target','sim_ns':now,'reason':'prediction_unavailable'});return None
            source=dict(self.queue.pending[0][3]);offset=self.queue.pending[0][1]
            def check(action,event):
                if source['clock_id']!=self.clock_id or now<source['available_sim_ns'] or now<source['capture_sim_ns']:raise ValueError('Invalid source clock/availability')
                validator(action,event)
            action=self.queue.pop(context_validator=check)
            self.events.append({'event':'sim_target_selected','sim_ns':now,'chunk_offset':offset,'source':source,'source_age_ns':now-source['capture_sim_ns'],'hardware_dispatched':False})
            if len(self.queue.pending)==self.prefetch_remaining and self.job is None:self._submit(now)
            return action
        except Exception:
            self.failed=True;raise
    def reset(self):
        self.events.append({'event':'scheduler_reset','pending_job_discarded':self.job.source if self.job else None})
        self.queue.reset();self.buffer.clear();self.job=None;self.last_tick=None;self.next_job=0;self.capture_count=0;self.failed=False
