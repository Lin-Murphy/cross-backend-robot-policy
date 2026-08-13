"""Validate stored pilot traces and videos without running another policy or environment."""
import argparse
import hashlib
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image, ImageDraw


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run',type=Path);args=ap.parse_args()
    p=args.run;r=json.loads((p/'result.json').read_text())
    checks=[];all_forward=[];all_chunk=[]
    assert r['strict_load_passed'] and r['native_queue_verification']['max_abs_error']<=1e-4
    for e in r['episodes']:
        horizon=e.get('execute_steps',r['execute_steps'])
        d=p/f"episode_{e['episode']:03d}";rows=[json.loads(s) for s in (d/'trace.jsonl').read_text().splitlines()]
        assert len(rows)==e['steps']
        assert [x['step'] for x in rows]==list(range(e['steps']))
        for row in rows:
            replan = (row['step']//horizon)*horizon
            source = max(0, replan-e.get('delay_steps', 0))
            assert row['action_source_step']==source
            if r.get('schema_version', 1) >= 2:
                assert row['replan_step']==replan
                assert row['configured_input_delay_steps']==e.get('delay_steps', 0)
                assert row['actual_input_delay_steps']==replan-source
                assert row['history_size']==min(row['step']+1,r['history_capacity'])
                assert row['source_capture_monotonic_ns']==rows[source]['observation_capture_monotonic_ns']
                assert abs(row['action_source_age_wall_ms']-(row['dispatch_monotonic_ns']-row['source_capture_monotonic_ns'])/1e6)<1e-9
            assert row['chunk_offset']==row['step']%horizon
            assert abs(row['action_source_age_sim_s']-(row['step']-source)/10)<1e-10
            assert np.isfinite(row['action']).all() and all(0<=a<=512 for a in row['action'])
            with np.load(d/f"chunk_{row['chunk_id']:03d}.npz") as c:
                assert np.array_equal(c['actions'][row['chunk_offset']],np.array(row['action'],dtype=np.float32))
                assert c['actions'].shape==(r['prediction_steps'],2)
                if r.get('schema_version', 1)>=2:
                    assert int(c['source_step'])==source and int(c['replan_step'])==replan
                if row['chunk_offset']==0:
                    assert hashlib.sha256(c['input_rgb'].tobytes()).hexdigest()==rows[source]['observation_rgb_sha256']
                    # Preserve raw environment precision; adapter casts to float32 only for the model.
                    assert np.array_equal(c['input_state'],np.array(rows[source]['state_before']))
            assert row['success']==(row['coverage']>0.95)
            assert abs(row['reward']-min(1.,max(0.,row['coverage']/0.95)))<1e-12
        assert e['success']==any(x['success'] for x in rows)
        assert e['max_coverage']==max(x['coverage'] for x in rows)
        assert e['final_coverage']==rows[-1]['coverage']
        with av.open(str(d/'rollout.mp4')) as c:
            frames=0;selected=[];targets={0,e['steps']//4,e['steps']//2,3*e['steps']//4,e['steps']}
            for i,f in enumerate(c.decode(video=0)):
                frames+=1
                if i in targets:selected.append((i,f.to_image()))
        assert frames==e['steps']+1==e['video_frames']
        sheet=Image.new('RGB',(1280,285),'white');draw=ImageDraw.Draw(sheet)
        for i,(frame,im) in enumerate(selected):
            im.thumbnail((256,256));sheet.paste(im,(i*256,25));draw.text((i*256+3,4),f'episode {e["episode"]} frame {frame}',fill='black')
        sheet.save(d/'contact.jpg',quality=90)
        forwards=[x['forward_ms'] for x in e['chunks']];totals=[x['processing_and_forward_ms'] for x in e['chunks']]
        all_forward+=forwards;all_chunk+=totals
        checks.append({'episode':e['episode'],'execute_steps':horizon,'delay_steps':e.get('delay_steps', 0),'steps':len(rows),'video_frames':frames,
                       'native_chunk_actions_match':True,'action_origin_and_input_match':True,
                       'success_and_coverage_match':True,'forward_ms':forwards,'chunk_total_ms':totals,
                       'max_action_origin_age_sim_s':max(x['action_source_age_sim_s'] for x in rows),
                       'max_action_origin_age_wall_ms':max(x['action_source_age_wall_ms'] for x in rows),
                       'queue_pop_median_ms':float(np.median([x['queue_pop_ms'] for x in rows]))})
        print('episode',e['episode'],'success',e['success'],'max/final coverage',e['max_coverage'],e['final_coverage'],
              'forwards_ms',forwards,'peak_GPU_MiB',e['gpu_peak_allocated_bytes']/2**20,
              'wall_s',e.get('wall_ms_including_recording',float('nan'))/1000)
    summary={'all_checks_passed':True,'episodes':checks,'full_chunk_calls_measured':len(all_forward),
             'forward_median_ms':float(np.median(all_forward)),'forward_range_ms':[min(all_forward),max(all_forward)],
             'full_chunk_median_ms':float(np.median(all_chunk)),
             'timing_limit':f'{len(all_forward)} post-warmup chunk calls; descriptive exploratory values, not independent-trial sample size or hardware deadline evidence',
             'no_extra_environment_rollouts':True,'checkpoint_unchanged':r['checkpoint_files_before']==r['checkpoint_files_after']}
    assert summary['checkpoint_unchanged']
    (p/'verification.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
