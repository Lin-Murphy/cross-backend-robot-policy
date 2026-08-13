"""Read-only local dataset QA; writes only a new output directory. No model/hardware imports.
Run using the existing LeRobot venv; --output must not already exist.
"""
import argparse, csv, hashlib, importlib.metadata, json, platform, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def dump(p,obj):
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x)))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--root',type=Path,default=Path('/home/murphy/.cache/huggingface/lerobot')); args=ap.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False); (out/'qa').mkdir()
    started=datetime.now(timezone.utc).isoformat()
    dump(out/'environment.json',dict(started_utc=started,python=sys.version,executable=sys.executable,platform=platform.platform(),packages={k:importlib.metadata.version(k) for k in ['av','numpy','pyarrow','pillow','torch','torchvision','lerobot']},script_sha256=sha(Path(__file__)),argv=sys.argv,near_black_rule='RGB uint8 maximum <=10 across every pixel; not a general acceptance criterion'))
    roots=sorted(p.parent.parent for p in args.root.glob('*/*/meta/info.json'))
    sources=sorted(set(p for r in roots for p in r.rglob('*') if p.is_file()))
    before=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,sha256=sha(p)) for p in sources]
    dump(out/'input_manifest.json',before)
    results=[]; all_eps=[]; samples=[]
    for root in roots:
        name=root.name; info=json.loads((root/'meta/info.json').read_text()); fps=info['fps']
        r=dict(path=str(root),name=name,declared_episodes=info['total_episodes'],declared_frames=info['total_frames'],features=info['features'],splits=info.get('splits'),errors=[],videos=[])
        epfiles=sorted((root/'meta/episodes').rglob('*.parquet'))
        eps=[row for p in epfiles for row in pq.read_table(p).to_pylist()]
        datafiles=sorted((root/'data').rglob('*.parquet'))
        rows=[row for p in datafiles for row in pq.read_table(p).to_pylist()]
        r['actual_rows']=len(rows); r['actual_episodes']=len(eps)
        r['tasks']=pq.read_table(root/'meta/tasks.parquet').to_pylist() if (root/'meta/tasks.parquet').exists() else []
        if len(rows)!=info['total_frames'] or len(eps)!=info['total_episodes']: r['errors'].append('declared_count_mismatch')
        if [x['index'] for x in rows]!=list(range(len(rows))): r['errors'].append('global_index_noncontiguous')
        erows={int(e['episode_index']):e for e in eps}; audits={}; thumbs={}
        for eid,e in erows.items():
            rr=[x for x in rows if x['episode_index']==eid]; n=len(rr)
            a=dict(dataset=name,episode=eid,length=e['length'],rows=n,tasks=e.get('tasks'),errors=[],videos={},metadata_image_stats={k:v for k,v in e.items() if k.startswith('stats/observation.images')})
            if n!=e['length']: a['errors'].append('length_mismatch')
            if [x['frame_index'] for x in rr]!=list(range(n)): a['errors'].append('frame_index_noncontiguous')
            if n:
                ts=np.array([x['timestamp'] for x in rr]); a['max_timestamp_grid_error_s']=float(np.max(abs(ts-np.arange(n)/fps)))
                if a['max_timestamp_grid_error_s']>1e-4: a['errors'].append('timestamp_grid_mismatch')
                if not np.all(np.diff(ts)>0): a['errors'].append('nonincreasing_data_time')
                if e['dataset_from_index']!=rr[0]['index'] or e['dataset_to_index']!=rr[-1]['index']+1: a['errors'].append('dataset_range_mismatch')
                h=hashlib.sha256()
                for key in ['observation.state','action']:
                    ar=np.asarray([x[key] for x in rr],dtype='<f4'); h.update(ar.tobytes())
                    a[key]=dict(min=ar.min(0).tolist(),max=ar.max(0).tolist(),mean=ar.mean(0).tolist(),nonfinite=int((~np.isfinite(ar)).sum()),max_abs_step=np.abs(np.diff(ar,axis=0)).max(0).tolist() if n>1 else [0]*6)
                    if not np.isfinite(ar).all(): a['errors'].append(key+'_nonfinite')
                a['state_action_sha256']=h.hexdigest()
                ar=np.asarray([x['action'] for x in rr]); st=np.asarray([x['observation.state'] for x in rr]); a['action_state_mean_abs']=np.abs(ar-st).mean(0).tolist()
            audits[eid]=a
        for vp in sorted(root.rglob('*.mp4')):
            rel=str(vp.relative_to(root)); key=vp.parts[-3] if 'videos' in vp.parts else 'observation.images.follower'
            # Actual video paths are resolved from metadata rather than inferred from row index.
            segments=[]
            for eid,e in erows.items():
                prefix='videos/'+key
                if prefix+'/from_timestamp' not in e: continue
                expected=root/info['video_path'].format(video_key=key,chunk_index=e[prefix+'/chunk_index'],file_index=e[prefix+'/file_index'])
                if expected!=vp: continue
                lo=e[prefix+'/from_timestamp']; hi=e[prefix+'/to_timestamp']
                segments.append((eid,lo,hi)); audits[eid]['videos'][key]=dict(from_s=lo,to_s=hi,frames=0,near_black_frames=0,zero_frames=0,min=255,max=0,mean_sum=0,pts=[],digest=hashlib.sha256())
            vr=dict(path=rel,frames=0,near_black_frames=0,zero_frames=0,min=255,max=0,mean_sum=0,pts_monotonic=True,unassigned=0)
            targets={}
            for eid,lo,hi in segments:
                for frac in [0.1,0.5,0.9]: targets[round((lo+(hi-lo)*frac)*fps)]=(eid,frac)
            # Orphan video retains standalone visual samples and is never silently assigned to an episode.
            try:
                container=av.open(str(vp)); stream=container.streams.video[0]; stream.thread_type='AUTO'; stream.codec_context.thread_count=2
                vr.update(codec=stream.codec_context.name,width=stream.width,height=stream.height,time_base=str(stream.time_base),declared_stream_frames=stream.frames)
                prev=None; first=None
                with (out/(name+'__'+key.replace('.','_')+'__'+vp.parent.name+'_'+vp.stem+'_frames.csv')).open('w',newline='') as f:
                    writer=csv.writer(f); writer.writerow(['frame','pts_s','episode','min','max','mean','near_black','rgb_sha256'])
                    for idx,frame in enumerate(container.decode(stream)):
                        ar=frame.to_ndarray(format='rgb24'); ts=float(frame.pts*frame.time_base); amin=int(ar.min()); amax=int(ar.max()); mean=float(ar.mean()); dig=hashlib.sha256(ar.tobytes()).hexdigest()
                        if first is None: first=ar.copy()
                        if prev is not None and ts<=prev: vr['pts_monotonic']=False
                        prev=ts; vr['frames']+=1; vr['near_black_frames']+=amax<=10; vr['zero_frames']+=amax==0; vr['min']=min(vr['min'],amin);vr['max']=max(vr['max'],amax);vr['mean_sum']+=mean
                        assigned=[eid for eid,lo,hi in segments if lo-1e-6<=ts<hi-1e-6]
                        if len(assigned)!=1: vr['unassigned']+=1
                        writer.writerow([idx,ts,';'.join(map(str,assigned)),amin,amax,mean,int(amax<=10),dig])
                        for eid in assigned:
                            v=audits[eid]['videos'][key]; v['frames']+=1; v['near_black_frames']+=amax<=10; v['zero_frames']+=amax==0;v['min']=min(v['min'],amin);v['max']=max(v['max'],amax);v['mean_sum']+=mean;v['pts'].append(ts);v['digest'].update(bytes.fromhex(dig))
                        if idx in targets or (not segments and idx==0):
                            eid,frac=targets.get(idx,(-1,0.0)); im=Image.fromarray(ar);thumb=im.copy();thumb.thumbnail((320,240));thumbs.setdefault((eid,key),[]).append((ts,thumb))
                            sample=dict(dataset=name,video=rel,episode=eid,frame=idx,pts_s=ts,fraction=frac,rgb_sha256=dig,source_video_sha256=next(x['sha256'] for x in before if x['path']==str(vp)),min=amin,max=amax,mean=mean)
                            if eid in [0,15,-1] and (frac==0.5 or eid==-1):
                                fn=f'{name}_e{eid}_{key}_{idx}.png';im.save(out/'qa'/fn);sample['png']='qa/'+fn
                            samples.append(sample)
                container.close()
                vr['mean']=vr.pop('mean_sum')/max(1,vr['frames']);vr['last_pts_s']=prev
                # Separate FFmpeg executable/linked libraries, exact frame 0, no scaling.
                cp=subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(vp),'-frames:v','1','-f','rawvideo','-pix_fmt','rgb24','-threads','2','pipe:1'],capture_output=True)
                vr['ffmpeg_frame0']=dict(returncode=cp.returncode,stderr=cp.stderr.decode(errors='replace')[:2000],bytes=len(cp.stdout))
                if first is not None and len(cp.stdout)==first.size:
                    other=np.frombuffer(cp.stdout,np.uint8).reshape(first.shape);vr['ffmpeg_frame0'].update(min=int(other.min()),max=int(other.max()),mean=float(other.mean()),max_abs_difference=int(np.abs(other.astype(int)-first.astype(int)).max()))
            except Exception as ex:
                vr['error']=repr(ex);r['errors'].append('video_decode_error:'+rel)
            r['videos'].append(vr)
        for eid,a in audits.items():
            for key,v in a['videos'].items():
                pts=np.array(v.pop('pts'));v['decoded_rgb_sequence_sha256']=v.pop('digest').hexdigest();v['mean']=v.pop('mean_sum')/max(1,v['frames'])
                expected=v['from_s']+np.arange(a['length'])/fps
                v['max_pts_grid_error_s']=float(np.max(abs(pts-expected))) if len(pts)==len(expected) and len(pts) else None
                if v['frames']!=a['length']: a['errors'].append('video_frame_count_mismatch')
                if abs((v['to_s']-v['from_s'])*fps-a['length'])>1e-3: a['errors'].append('video_duration_mismatch')
                if v['max_pts_grid_error_s'] is not None and v['max_pts_grid_error_s']>1e-4: a['errors'].append('video_pts_mismatch')
            if not a['videos']: a['errors'].append('missing_video')
            all_eps.append(a)
        for key in sorted({k for _,k in thumbs}):
            entries=sorted((eid,ims) for (eid,k),ims in thumbs.items() if k==key)
            # At most 10 episodes per sheet for readable visual review.
            for page in range(0,len(entries),10):
                batch=entries[page:page+10]; sheet=Image.new('RGB',(960,270*len(batch)),(240,240,240)); draw=ImageDraw.Draw(sheet)
                for row,(eid,ims) in enumerate(batch):
                    for col,(ts,im) in enumerate(ims[:3]): sheet.paste(im,(320*col,270*row+25));draw.text((320*col+4,270*row+4),f'ep {eid} | PTS {ts:.3f}s',(0,0,0))
                sheet.save(out/'qa'/f'{name}_{key}_sheet{page//10}.jpg',quality=88)
        results.append(r);dump(out/'datasets.json',results);dump(out/'episodes.json',all_eps);dump(out/'samples.json',samples)
        print(name,'rows',len(rows),'videos',[(v['frames'],v['near_black_frames'],v.get('error')) for v in r['videos']],flush=True)
    groups={}
    for a in all_eps:
        if 'state_action_sha256' in a: groups.setdefault(a['state_action_sha256'],[]).append(dict(dataset=a['dataset'],episode=a['episode'],length=a['length']))
    dump(out/'duplicate_state_action_groups.json',[dict(sha256=k,members=v) for k,v in groups.items() if len(v)>1])
    after=[dict(path=x['path'],sha256=sha(Path(x['path'])),bytes=Path(x['path']).stat().st_size,mtime_ns=Path(x['path']).stat().st_mtime_ns) for x in before]
    dump(out/'input_integrity_after.json',dict(checked=len(before),changed=[x['path'] for x,y in zip(before,after) if x!=y],files=after,finished_utc=datetime.now(timezone.utc).isoformat()))
    print('FINISHED',out,flush=True)

if __name__=='__main__': main()
