"""Real-clock scheduled GPU replay of saved inputs; all queues are diagnostic only.

Never grants gate acceptance or connects hardware. Exposure timestamps stay None.
An independent strict gate queue rejects each prediction before any schedule-only
inspection. Its latch is never bypassed to drive hardware or continue that queue.
"""
import os
for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    os.environ[key] = '1'
os.environ['WANDB_MODE'] = 'disabled'
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
from pathlib import Path
import socket
import sys
import time
import traceback
import uuid
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
PERIOD_NS = 33_333_333


def sleep_until(deadline_ns):
    while True:
        remaining = deadline_ns - time.perf_counter_ns()
        if remaining <= 0:
            return
        time.sleep(remaining / 1e9)


def statistics(values):
    import numpy as np
    return {'count': len(values), 'median': float(np.median(values)),
            'p95': float(np.percentile(values, 95)), 'max': float(max(values))}


def main(out, prefetch_steps=0):
    original_connect = socket.socket.connect
    def offline(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError('Network disabled in R3 timing run')
        return original_connect(sock, address)
    socket.socket.connect = offline
    import numpy as np
    import torch
    from cross_backend.move_pot_policy import ACTMovePotAdapter, MovePotObservation, ReplayChunkQueue, CAMERAS, JOINTS, UNITS, file_sha256
    from cross_backend.smolvla_move_pot import SmolVLAMovePotAdapter
    from cross_backend.degree_execution import DegreeCandidateGate, DegreeGateRejected, pop_checked
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for bounded timing measurement')
    fixture = ROOT/'artifacts/r2-live-readonly-20260924-attempt01/sample-00.npz'
    with np.load(fixture, allow_pickle=False) as data:
        obs = MovePotObservation({k: data[k].copy() for k in CAMERAS}, data['state'].copy(),
                                 'saved-live:sample-00', {k:'saved-live:sample-00:'+k for k in CAMERAS},
                                 {k:None for k in CAMERAS}, None)
    protected = [fixture, *sorted((ROOT/'artifacts/move-pot-evaluation-v1-records').glob('*.json'))]
    hashes = {str(p):file_sha256(p) for p in protected}
    result = {'status':'running', 'clock_kind':'monotonic', 'clock_session':str(uuid.uuid4()),
              'scope':'saved-input GPU scheduling only; not physical control or exposure-age test',
              'hardware_dispatches':0, 'training_updates':0, 'capture_included':False,
              'task_success_evaluated':False, 'period_ns':PERIOD_NS, 'warmups_per_model':2,
              'chunks_per_condition':3, 'prefetch_steps':prefetch_steps, 'conditions':['baseline', 'host_hold_33ms'],
              'condition_semantics':'measured saved-input host hold; NOT added camera exposure age',
              'device':torch.cuda.get_device_name(), 'torch':torch.__version__, 'models':{}}
    specs = [('act',ACTMovePotAdapter,ROOT/'artifacts/r1-act-formal-40000-run02/train/checkpoints/040000/pretrained_model'),
             ('smolvla',SmolVLAMovePotAdapter,Path('/home/murphy/project/lerobot/outputs/train/smolvla_move_pot_dualcam_20260923_v3/checkpoints/040000/pretrained_model'))]
    for name,factory,cp in specs:
        print('Loading '+name,flush=True)
        adapter = factory(cp,'cuda')
        manifest = adapter.checkpoint_manifest.copy()
        for _ in range(2): adapter.predict_chunk(obs)
        model = {'checkpoint':str(cp), 'manifest':manifest, 'conditions':{}}
        torch.cuda.reset_peak_memory_stats()
        for condition in result['conditions']:
            adapter.reset()
            q = ReplayChunkQueue(out/f'{name}-{condition}-diagnostic-events.jsonl',name+'-'+condition+'-schedule-only')
            strict = ReplayChunkQueue(out/f'{name}-{condition}-strict-events.jsonl',name+'-'+condition+'-strict-gate')
            gate = DegreeCandidateGate(None,result['clock_session'],'monotonic')
            current = {'state':obs.state.tolist(), 'state_capture_ns':None,
                       'joint_names':list(JOINTS),'joint_units':list(UNITS)}
            ticks=[]; blocks=[]; all_actions=[]; previous=None
            pool=ThreadPoolExecutor(max_workers=1) if prefetch_steps else None
            pending=None
            def predict():
                host_start = time.perf_counter_ns()
                hold = PERIOD_NS if condition=='host_hold_33ms' else 0
                sleep_until(host_start+hold)
                available = time.perf_counter_ns()
                chunk = adapter.predict_chunk(obs)
                return host_start,hold,available,chunk,time.perf_counter_ns()
            try:
                for block in range(3):
                    if pending is None:
                        host_start,hold,available,chunk,returned=predict()
                    else:
                        host_start,hold,available,chunk,returned=pending.result(timeout=15)
                        pending=None
                    future_collected_ns=time.perf_counter_ns()
                    chunk.source.update({'observation_npz_sha256':hashes[str(fixture)],
                                         'timing_experiment':{'clock_session':result['clock_session'],
                                         'clock_kind':'monotonic','host_hold_start_ns':host_start,
                                         'host_payload_available_ns':available, 'requested_host_hold_ns':hold,
                                         'saved_payload_not_new_capture':True}})
                    # Run strict gate, retain rejection, reset it; never consume its rejected remainder.
                    gate.reset(host_start); strict.enqueue(chunk,50)
                    try: pop_checked(strict,gate,current,time.perf_counter_ns())
                    except DegreeGateRejected as exc:
                        reasons=exc.reasons
                        assert 'missing_limits' in reasons and any('capture_unknown' in r for r in reasons)
                    else: raise AssertionError('Saved inputs cannot pass strict gate')
                    strict.reset(); gate.reset(None)
                    q.enqueue(chunk,50)
                    queue_ready=time.perf_counter_ns()
                    first=None
                    for offset in range(50):
                        deadline = queue_ready if previous is None else previous+PERIOD_NS
                        sleep_until(deadline)
                        selected=time.perf_counter_ns()
                        action=q.pop()  # deliberately not_checked; diagnostic inspection, no acceptance
                        np.testing.assert_array_equal(action,chunk.actions[offset])
                        ticks.append({'block':block,'offset':offset,'selected_ns':selected,
                                      'previous_selected_ns':previous,'deadline_ns':deadline,
                                      'interval_ns':None if previous is None else selected-previous,
                                      'lateness_ns':max(0,selected-deadline),
                                      'prediction_residence_ns':selected-chunk.prediction_end_ns,
                                      'camera_age_ns':None,'state_age_ns':None,'action_source_age_ns':None,
                                      'hardware_dispatched':False,'gate_status':'not_checked_diagnostic_only'})
                        if first is None: first=selected
                        previous=selected
                        if pool is not None and block<2 and offset==50-prefetch_steps:
                            assert pending is None
                            pending=pool.submit(predict)
                    blocks.append({'block':block,'host_start_ns':host_start,'host_available_ns':available,
                                   'requested_hold_ns':hold,'measured_hold_ns':available-host_start,
                                   'prediction_start_ns':chunk.prediction_start_ns,'prediction_end_ns':chunk.prediction_end_ns,
                                   'adapter_return_ns':returned,'future_collected_ns':future_collected_ns,'queue_ready_ns':queue_ready,'first_candidate_ns':first,
                                   'forward_ms':chunk.forward_ms,'processing_and_forward_ms':chunk.processing_and_forward_ms,
                                   'strict_rejection_reasons':reasons})
                    all_actions.append(chunk.actions)
            finally:
                if pool is not None: pool.shutdown(wait=True, cancel_futures=True)
                q.reset();q.close();strict.close()
            boundaries=[r['interval_ns']/1e6 for r in ticks if r['offset']==0 and r['interval_ns'] is not None]
            within=[r['interval_ns']/1e6 for r in ticks if r['offset']!=0]
            measurement={'blocks':blocks,'ticks':ticks,'boundary_intervals_ms':boundaries,
                         'within_chunk_interval_ms':statistics(within),
                         'candidate_intervals_over_40ms':sum(r['interval_ns'] is not None and r['interval_ns']>40_000_000 for r in ticks),
                         'diagnostic_threshold_note':'40ms is a synthetic diagnostic reference, not authorized limit',
                         'forward_ms':statistics([b['forward_ms'] for b in blocks]),
                         'processing_and_forward_ms':statistics([b['processing_and_forward_ms'] for b in blocks]),
                         'max_prediction_residence_ms':max(r['prediction_residence_ns'] for r in ticks)/1e6,
                         'strict_rejected_chunks':len(blocks),'candidates_inspected':len(ticks)}
            model['conditions'][condition]=measurement
            np.savez_compressed(out/f'{name}-{condition}-actions.npz',actions=np.stack(all_actions))
            (out/f'{name}-{condition}-timing.json').write_text(json.dumps(measurement,indent=2)+'\n')
            print(name+' '+condition+': '+json.dumps({'boundary_intervals_ms':boundaries,
                'within_chunk_median_ms':measurement['within_chunk_interval_ms']['median'],
                'strict_rejected_chunks':3}),flush=True)
        assert all(file_sha256(cp/f)==h for f,h in manifest.items())
        model.update(checkpoint_unchanged=True,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
        result['models'][name]=model
        (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        del adapter,chunk
        gc.collect();torch.cuda.empty_cache()
    assert all(file_sha256(p)==h for p,h in hashes.items())
    result.update(status='passed',protected_inputs_sha256=hashes,protected_inputs_unchanged=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('R3 scheduled timing complete: no dispatch, no exposure-age claims.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--prefetch-steps',type=int,choices=(0,6),default=0)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    try: main(args.output,args.prefetch_steps)
    except BaseException:
        (args.output/'failure.txt').write_text(traceback.format_exc());raise
