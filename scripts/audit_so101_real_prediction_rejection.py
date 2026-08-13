"""Record actual saved SmolVLA prediction through shared queue/gate, with no motor imports."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from cross_backend.move_pot_policy import MovePotObservation,MovePotChunk,ReplayChunkQueue,CAMERAS,JOINTS,UNITS
from cross_backend.degree_execution import DegreeCandidateGate,DegreeGateRejected,pop_checked


def main(capture,prediction,out):
    out.mkdir(parents=True,exist_ok=False)
    obs_summary=json.loads((capture/'summary.json').read_text())
    pred_summary=json.loads((prediction/'summary.json').read_text())
    assert obs_summary['status']=='passed' and obs_summary['dispatches']==0
    assert pred_summary['status']=='passed' and pred_summary['hardware_dispatches']==0
    with np.load(capture/'sample-00.npz',allow_pickle=False) as data:
        obs=MovePotObservation({k:data[k].copy() for k in CAMERAS},data['state'].copy(),
            capture.name+':0',{k:capture.name+':0:'+k for k in CAMERAS},
            {k:None for k in CAMERAS},None,task='move pot')
    obs.validate()
    with np.load(prediction/'prediction.npz',allow_pickle=False) as data:
        actions=data['actions'].copy();np.testing.assert_array_equal(data['source_state'],obs.state)
    assert actions.shape==(50,6) and np.isfinite(actions).all()
    source={'observation_id':obs.observation_id,'frame_ids':dict(obs.frame_ids),
        'camera_capture_ns':dict(obs.camera_capture_ns),'state_capture_ns':None,
        'state':obs.state.tolist(),'joint_names':list(JOINTS),'joint_units':list(UNITS),
        'task':obs.task,'policy':'SmolVLA','backend':'so101',
        'source_capture_sha256':hashlib.sha256((capture/'sample-00.npz').read_bytes()).hexdigest(),
        'source_prediction_sha256':hashlib.sha256((prediction/'prediction.npz').read_bytes()).hexdigest(),
        'clock_kind':None,'clock_session':None}
    now=time.perf_counter_ns()
    queue=ReplayChunkQueue(out/'events.jsonl','so101-real-observation-smolvla-rejection')
    gate=DegreeCandidateGate(None,None,None)
    try:
        queue.enqueue(MovePotChunk(actions,source,now,now,pred_summary['inference_seconds']*1000,
                                    pred_summary['inference_seconds']*1000),50)
        try:pop_checked(queue,gate,source,time.perf_counter_ns())
        except DegreeGateRejected as error:
            assert 'missing_limits' in error.reasons and gate.latched and queue.failed
            reasons=error.reasons
        else:raise AssertionError('Expected real hardware candidate to be rejected without physical limits and exposure timestamps')
        queue.reset();gate.reset(None)
    finally:queue.close()
    events=[json.loads(s) for s in (out/'events.jsonl').read_text().splitlines()]
    assert [x['event'] for x in events]==['prediction','candidate','reset']
    assert events[1]['gate_status']=='rejected' and all(not x.get('hardware_dispatched',False) for x in events)
    result={'status':'passed','backend':'so101','task':'move pot','policy':'SmolVLA',
        'evaluation_scope':'saved_real_observation_prediction_rejection_not_task_trial',
        'source_observation':str(capture),'source_prediction':str(prediction),
        'candidate_rejection_reasons':reasons,'recorded_actions':len(actions),
        'discarded_after_rejection':events[2]['discarded_actions'],
        'hardware_dispatches':0,'task_outcome':None,'formal_trial_count':0,
        'events_sha256':hashlib.sha256((out/'events.jsonl').read_bytes()).hexdigest()}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True)
    p.add_argument('--prediction',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();main(a.capture,a.prediction,a.output)
