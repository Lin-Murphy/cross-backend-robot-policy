"""Diagnose saved trajectories without inference, downloads or new rollouts."""
from pathlib import Path
import hashlib
import json
import shutil
import numpy as np
import av
from PIL import Image, ImageDraw
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'artifacts/sim-v1-act-queue-20260921/run01'
out=ROOT/'artifacts/sim-v1-act-queue-diagnosis-20260921'
out.mkdir(exist_ok=False)
r=json.loads((source/'result.json').read_text())
results=[]
for seed in range(1000,1005):
    es={e['execute_steps']:e for e in r['episodes'] if e['seed']==seed}
    ds={h:source/f"episode_{e['episode']:03d}" for h,e in es.items()}
    rows={h:[json.loads(s) for s in (d/'trace.jsonl').read_text().splitlines()] for h,d in ds.items()}
    n=min(map(len,rows.values()))
    first=next(i for i in range(n) if rows[100][i]['action']!=rows[10][i]['action'])
    assert first==10
    details={}
    for h,rr in rows.items():
        cov=np.array([x['coverage'] for x in rr]);peak=int(np.argmax(cov))
        jumps=[float(np.linalg.norm(np.array(rr[i]['action'])-rr[i-1]['action'])) for i in range(1,len(rr)) if i%h==0]
        revisions=[]
        for cid in range(1,len(es[h]['chunks'])):
            # A 10-step replan has an unused counterfactual prediction for this same timestep.
            if h==10:
                with np.load(ds[h]/f'chunk_{cid-1:03d}.npz') as prev,np.load(ds[h]/f'chunk_{cid:03d}.npz') as cur:
                    revisions.append(float(np.linalg.norm(cur['actions'][0]-prev['actions'][10])))
        details[str(h)]={'peak_step':peak,'peak_coverage':float(cov[peak]),'final_coverage':float(cov[-1]),
          'loss_after_peak':float(cov[peak]-cov[-1]),'steps_coverage_above_090':int((cov>.9).sum()),
          'replan_action_jump_median_pixels':float(np.median(jumps)),
          'replan_revision_median_pixels':float(np.median(revisions)) if revisions else None,
          'replan_revision_max_pixels':max(revisions) if revisions else None}
    divergence=float(np.linalg.norm(np.array(rows[100][10]['action'])-rows[10][10]['action']))
    results.append({'seed':seed,'first_action_divergence_step':first,'first_divergence_action_distance_pixels':divergence,'conditions':details})
    # Same-time images at first divergence, each condition's peak, and last common step.
    targets=sorted({10,11,details['100']['peak_step']+1,details['10']['peak_step']+1,n})
    sheet=Image.new('RGB',(256*len(targets),560),'white');draw=ImageDraw.Draw(sheet)
    for rowidx,h in enumerate([100,10]):
        selected={}
        with av.open(str(ds[h]/'rollout.mp4')) as c:
            for idx,f in enumerate(c.decode(video=0)):
                if idx in targets:selected[idx]=f.to_image()
        for col,idx in enumerate(targets):
            draw.text((col*256+3,rowidx*280+3),f'seed={seed} execute={h} frame={idx}',fill='black')
            if idx in selected:
                selected[idx].thumbnail((256,256));sheet.paste(selected[idx],(col*256,rowidx*280+22))
            else:draw.text((col*256+4,rowidx*280+100),'Already terminated',fill='black')
    sheet.save(out/f'seed_{seed}_paired.jpg',quality=92)
(out/'diagnosis.json').write_text(json.dumps(results,indent=2)+'\n')
shutil.copy2(__file__,out/Path(__file__).name)
(out/'source_manifest.json').write_text(json.dumps({str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()},indent=2)+'\n')
print(json.dumps(results,indent=2))
