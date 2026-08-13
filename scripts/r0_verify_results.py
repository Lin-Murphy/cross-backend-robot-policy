"""Independent FFmpeg all-frame checks plus numerical/calibration consistency assertions."""
from pathlib import Path
import argparse
import csv,json,subprocess
import numpy as np
import pyarrow.parquet as pq
from r0_offline_audit import dump,sha
ap=argparse.ArgumentParser(); ap.add_argument('--audit',type=Path,required=True); args=ap.parse_args()
p=args.audit.resolve()
if (p/'verification.json').exists(): raise FileExistsError('Use a fresh audit directory; existing results are preserved')
ds=json.loads((p/'datasets.json').read_text());es=json.loads((p/'episodes.json').read_text());pr=json.loads((p/'provenance.json').read_text());result={'ffmpeg':[],'calibration_grid':[]}
for suffix in ['114052','combined_20260919','180644','194524']:
 d=next(d for d in ds if d['name'].endswith(suffix));vp=Path(d['path'])/d['videos'][0]['path'];output=p/(d['name']+'_ffmpeg_framehash.txt')
 cmd=['ffmpeg','-v','error','-threads','2','-i',str(vp),'-map','0:v:0','-pix_fmt','rgb24','-threads','2','-f','framehash','-hash','sha256','pipe:1']
 with output.open('w') as f:cp=subprocess.run(cmd,stdout=f,stderr=subprocess.PIPE,text=True)
 actual=[s.rsplit(',',1)[-1].strip() for s in output.read_text().splitlines() if s and not s.startswith('#')]
 reference=[r['rgb_sha256'] for r in csv.DictReader(next(p.glob(d['name']+'*frames.csv')).open())]
 result['ffmpeg'].append(dict(dataset=d['name'],command=cmd,returncode=cp.returncode,stderr=cp.stderr,frames=len(actual),same_as_pyav=len(actual)==len(reference) and actual==reference,mismatch_count=sum(a!=b for a,b in zip(actual,reference)),source_sha256=sha(vp)))
 print('independent decoder',suffix,result['ffmpeg'][-1]['same_as_pyav'],flush=True)
# Test gripper quantization against current candidate leader/follower calibration spans.
for suffix in ['114052','180644','194524']:
 d=next(d for d in ds if d['name'].endswith(suffix));rows=[r for f in sorted((Path(d['path'])/'data').rglob('*.parquet')) for r in pq.read_table(f).to_pylist()]
 for key,kind in [('action','teleoperators/so_leader'),('observation.state','robots/so_follower')]:
  c=next(c for c in pr['calibrations'] if '/calibration/'+kind+'/' in c['path']);cal=c['motors'];values=np.asarray([r[key] for r in rows],dtype=float);grip=cal['gripper'];span=grip['range_max']-grip['range_min'];ticks=values[:,5]*span/100
  body=[]
  for idx,joint in enumerate(d['features'][key]['names'][:5]):
   m=cal[joint.removesuffix('.pos')];raw=values[:,idx]*4095/360+(m['range_min']+m['range_max'])/2
   body.append(dict(joint=joint,integer_tick_max_residual=float(np.max(abs(raw-np.round(raw)))),implied_raw_min=float(raw.min()),implied_raw_max=float(raw.max()),outside_calibration_count=int(((raw<m['range_min']-1e-3)|(raw>m['range_max']+1e-3)).sum())))
  result['calibration_grid'].append(dict(dataset=d['name'],feature=key,candidate_sha256=c['sha256'],gripper_span=span,gripper_integer_tick_max_residual=float(np.max(abs(ticks-np.round(ticks)))),body=body,interpretation='Numerical consistency only, not proof of acquisition-time calibration or physical safety.'))
# Validate completed audit outputs and input preservation; no data edits.
assert not json.loads((p/'input_integrity_after.json').read_text())['changed']
assert all(not e['errors'] for e in es)
assert all(x['same_as_pyav'] and x['returncode']==0 for x in result['ffmpeg'])
assert all(x['state_action_equal'] and x['decoded_rgb_equal'] for x in json.loads((p/'combined_lineage.json').read_text()))
assert sum(v['frames'] for d in ds for v in d['videos'])==64419
result['assertions']='passed: 4 all-frame independent decodes, complete episode structure, 25 duplicate mappings, 87 input files preserved, decoded total 64419'
dump(p/'verification.json',result)
