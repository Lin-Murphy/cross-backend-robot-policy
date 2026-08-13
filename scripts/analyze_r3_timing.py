"""Validate retained timing evidence and model lookahead timing (not a live executor)."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def stats(values):
    return {'count':len(values),'median':float(np.median(values)),
            'p95':float(np.percentile(values,95)),'min':float(min(values)),'max':float(max(values))}


def main(root, run_name="run01"):
    run=root/run_name;summary=json.loads((run/'summary.json').read_text())
    assert summary['status']=='passed'
    period=summary['period_ns'];paired={};event_counts={'strict_rejected':0,'diagnostic_candidates':0}
    for model,data in summary['models'].items():
        paired[model]={}
        for condition,measurement in data['conditions'].items():
            ticks=measurement['ticks'];blocks=measurement['blocks']
            assert len(ticks)==150 and len(blocks)==3
            assert all(t['camera_age_ns'] is None and t['state_age_ns'] is None and t['action_source_age_ns'] is None for t in ticks)
            assert all(t['selected_ns']>=t['deadline_ns'] and t['prediction_residence_ns']>=0 for t in ticks)
            assert all(b['host_start_ns']<=b['host_available_ns']<=b['prediction_start_ns']<=b['prediction_end_ns']<=b['adapter_return_ns']<=b['queue_ready_ns']<=b['first_candidate_ns'] for b in blocks)
            for kind in ('strict','diagnostic'):
                events=[json.loads(x) for x in (run/f'{model}-{condition}-{kind}-events.jsonl').read_text().splitlines()]
                candidates=[e for e in events if e['event']=='candidate']
                predictions=[e for e in events if e['event']=='prediction']
                assert len(predictions)==3
                assert all(e.get('dispatch_monotonic_ns') is None and not e.get('hardware_dispatched',False) for e in events)
                if kind=='strict':
                    assert len(candidates)==3 and all(e['gate_status']=='rejected' for e in candidates)
                    assert sum(e['discarded_actions'] for e in events if e['event']=='reset')==147
                    event_counts['strict_rejected']+=len(candidates)
                else:
                    assert len(candidates)==150 and all(e['gate_status']=='not_checked' for e in candidates)
                    event_counts['diagnostic_candidates']+=len(candidates)
                with np.load(run/f'{model}-{condition}-actions.npz',allow_pickle=False) as arrays:
                    np.testing.assert_array_equal(arrays['actions'],[p['actions'] for p in predictions])
            service=[b['queue_ready_ns']-b['host_start_ns'] for b in blocks[1:]]
            # Trace model: begin preparing L periods before the nominal swap.
            # Any overrun increases interval; no live threading or dynamic-state claim.
            alternatives={}
            for lead in (() if summary.get("prefetch_steps",0) else (1,4,5,6)):
                simulated=[period+max(0,s-lead*period) for s in service]
                alternatives[str(lead)]={'clock_kind':'trace_simulation',
                    'simulated_boundary_intervals_ms':[v/1e6 for v in simulated],
                    'all_meet_40ms_diagnostic':all(v<=40_000_000 for v in simulated),
                    'prepared_payload_age_at_tail_ms':(49+lead)*period/1e6,
                    'physical_source_age_ns':None}
            paired[model][condition]={
                'measured_boundary_intervals_ms':measurement['boundary_intervals_ms'],
                'within_chunk_ms':measurement['within_chunk_interval_ms'],
                'service_to_ready_ms':[s/1e6 for s in service],
                'minimum_lookahead_periods_for_no_extra_gap_in_this_trace':math.ceil(max(service)/period),
                'lookahead_counterfactuals':alternatives,
                'processing_and_forward_ms':measurement['processing_and_forward_ms'],
                'measured_host_hold_ms':[ (b['host_available_ns']-b['host_start_ns'])/1e6 for b in blocks],
                'max_prediction_residence_ms':measurement['max_prediction_residence_ms']}
        with np.load(run/f'{model}-baseline-actions.npz',allow_pickle=False) as a, np.load(run/f'{model}-host_hold_33ms-actions.npz',allow_pickle=False) as b:
            paired[model]['same_saved_input_hold_action_max_abs_difference']=float(np.max(np.abs(a['actions']-b['actions'])))
    camera_rows=[json.loads(x) for x in (root/'camera-probe/frames.jsonl').read_text().splitlines()]
    cameras={}
    import cv2
    for name in ('follower','camera2'):
        rows=[r for r in camera_rows if r['camera']==name];assert len(rows)==60
        raw=(root/'camera-probe'/f'{name}.mjpg').read_bytes()
        cursor=0;bad_decode=[]
        for r in rows:
            assert r['byte_offset']==cursor and r['bytesused']>0
            blob=raw[cursor:cursor+r['bytesused']];cursor+=r['bytesused']
            image=cv2.imdecode(np.frombuffer(blob,dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None or image.shape!=(480,640,3):bad_decode.append(r['index'])
        assert cursor==len(raw) and not bad_decode
        driver=np.array([r['driver_timestamp_ns'] for r in rows],dtype=np.int64)
        lag=[(r['host_dequeue_end_ns']-r['driver_timestamp_ns'])/1e6 for r in rows]
        assert all(r['timestamp_monotonic'] and r['timestamp_source_soe'] and not r['buffer_error'] for r in rows)
        assert np.all(np.diff(driver)>0) and min(lag)>=0
        drops=[{'after_index':i-1,'sequence_gap':rows[i]['sequence']-rows[i-1]['sequence']-1}
               for i in range(1,len(rows)) if rows[i]['sequence']!=rows[i-1]['sequence']+1]
        cameras[name]={'frames':60,'all_decode_640x480':True,'driver_flags':'monotonic + SOE',
            'driver_to_host_dequeue_ms_all':stats(lag),
            'driver_to_host_dequeue_ms_after_first5':stats(lag[5:]),
            'driver_intervals_ms_all':stats((np.diff(driver)/1e6).tolist()),
            'driver_intervals_ms_after_first5':stats((np.diff(driver[5:])/1e6).tolist()),
            'sequence_gaps':drops,'monotonic_and_not_future':True,'optically_validated':False,
            'startup_exclusion_note':'All raw frames retained; after-first5 is explicitly a diagnostic subset, not hidden exclusion'}
    result={'status':'passed','scope':'timing diagnostics, no physical gate authorization',
            'event_counts':event_counts,'models':paired,'cameras':cameras,
            'camera_timestamp_semantics':'driver-reported start-of-exposure estimate in CLOCK_MONOTONIC; optical error unmeasured',
            'clock_boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'hardware_dispatches':0,'counterfactual_warning':'Trace lookahead is simulated; no asynchronous executor implemented or validated'}
    for p,h in summary['protected_inputs_sha256'].items():
        assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
    (root/('analysis.json' if run_name=='run01' else 'analysis-'+run_name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','event_counts':event_counts,
        'model_boundary_intervals_ms':{m:{c:v['measured_boundary_intervals_ms'] for c,v in values.items() if isinstance(v,dict)} for m,values in paired.items()},
        'camera_driver_to_host_ms_after_first5':{k:v['driver_to_host_dequeue_ms_after_first5'] for k,v in cameras.items()},
        'camera_sequence_gaps':{k:v['sequence_gaps'] for k,v in cameras.items()}},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--directory',type=Path,required=True);parser.add_argument('--run-name',default='run01')
    args=parser.parse_args();main(args.directory,args.run_name)
