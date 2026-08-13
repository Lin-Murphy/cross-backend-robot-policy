"""Small follow-up checks on immutable run01 outputs and source video."""
from pathlib import Path
import argparse
import csv,json,subprocess
from PIL import Image,ImageDraw
from r0_offline_audit import dump,sha
ap=argparse.ArgumentParser(); ap.add_argument('--audit',type=Path,required=True); args=ap.parse_args()
p=args.audit.resolve()
if (p/'nearblack_exceptions.json').exists(): raise FileExistsError('Use a fresh audit directory; existing results are preserved')
ds=json.loads((p/'datasets.json').read_text());es=json.loads((p/'episodes.json').read_text());pr=json.loads((p/'provenance.json').read_text())
exceptions=[]
for suffix in ['180644','194524']:
 d=next(d for d in ds if d['name'].endswith(suffix));file=next(p.glob(d['name']+'*frames.csv')); rows=list(csv.DictReader(file.open())); selected=[r for r in rows if int(r['max'])>10]
 for row in selected[:12]:
  vp=Path(d['path'])/d['videos'][0]['path'];idx=int(row['frame']);fn=f"exception_{suffix}_f{idx}.png"
  cp=subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(vp),'-vf',f'select=eq(n\\,{idx})','-frames:v','1','-threads','2',str(p/'qa'/fn)],capture_output=True,text=True)
  exceptions.append(dict(dataset=d['name'],source=str(vp),source_sha256=sha(vp),frame=row,qa_png='qa/'+fn,ffmpeg_returncode=cp.returncode,stderr=cp.stderr))
dump(p/'nearblack_exceptions.json',exceptions)
# Verify merged episodes in numeric and decoded-pixel domains independently.
lookup={(e['dataset'],e['episode']):e for e in es};pairs=[]
combined='lerobot101_dataset_a_combined_20260919'
for i in range(25):
 name='lerobot101_dataset_a_20260919_114052' if i<15 else 'lerobot101_dataset_a_more20_20260919_180644';ep=i if i<15 else i-15
 a=lookup[(name,ep)];b=lookup[(combined,i)]
 pairs.append(dict(source=name,source_episode=ep,combined_episode=i,state_action_equal=a['state_action_sha256']==b['state_action_sha256'],decoded_rgb_equal=a['videos']['observation.images.follower']['decoded_rgb_sequence_sha256']==b['videos']['observation.images.follower']['decoded_rgb_sequence_sha256']))
dump(p/'combined_lineage.json',pairs)
# Assemble existing unaltered thumbnails of earlier episodes; all sample PTS remain visible.
files=[f for f in sorted((p/'qa').glob('*sheet0.jpg')) if any(s in f.name for s in ['111836','112115','112758','113011','113825','113210'])]
images=[(f,Image.open(f).copy()) for f in files]
sheet=Image.new('RGB',(960,sum(im.height+25 for _,im in images)),(240,240,240));draw=ImageDraw.Draw(sheet);y=0
for f,im in images:draw.text((4,y),f.name,fill='black');sheet.paste(im,(0,y+25));y+=im.height+25
sheet.save(p/'qa/early_candidates.jpg',quality=88)
# Summary including duplicate/static frame diagnostics and integrity assertions.
summary=[]
for d in ds:
 eps=[e for e in es if e['dataset']==d['name']];r=dict(dataset=d['name'],episodes=len(eps),rows=d['actual_rows'],errors=d['errors'],episode_errors=[dict(episode=e['episode'],errors=e['errors']) for e in eps if e['errors']],videos=d['videos'])
 summary.append(r)
dump(p/'summary.json',summary)
print('merge equality',sum(x['state_action_equal'] and x['decoded_rgb_equal'] for x in pairs),'of',len(pairs))
print('exceptions',len(exceptions)); print('episode errors',[(e['dataset'],e['episode'],e['errors']) for e in es if e['errors']]);print('total decoded frames',sum(v['frames'] for d in ds for v in d['videos']))
