"""Supplement R0 with allowlisted source/config evidence. Never execute collection commands."""
import json,shlex,subprocess
from pathlib import Path
import argparse
from datetime import datetime,timezone
import numpy as np
import pyarrow.parquet as pq
from r0_offline_audit import sha,dump

project=Path(__file__).resolve().parents[1]
ap=argparse.ArgumentParser(); ap.add_argument('--audit',type=Path,required=True); args=ap.parse_args()
out=args.audit.resolve()
if (out/'provenance.json').exists(): raise FileExistsError('Use a fresh audit directory; existing provenance is preserved')
repo=Path('/home/murphy/project/lerobot');base=Path('/home/murphy/.cache/huggingface/lerobot')
evidence=dict(utc=datetime.now(timezone.utc).isoformat(),project_git=(project/'.git').exists(),alternate_project_exists=Path('/home/murphy/project/cross-backend-lerobot-policy').exists(),lerobot_git_head=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip(),lerobot_git_status=subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True),history=[])
allow={'robot.type','robot.id','robot.cameras','robot.use_degrees','robot.calibration_dir','robot.max_relative_target','teleop.type','teleop.id','teleop.use_degrees','teleop.calibration_dir','dataset.repo_id','dataset.single_task','dataset.num_episodes','dataset.episode_time_s','dataset.fps','dataset.streaming_encoding','dataset.video','display_data','display_mode','resume'}
history=Path('/home/murphy/.bash_history')
for lineno,line in enumerate(history.read_text(errors='replace').splitlines(),1):
 if 'uv run lerobot-record ' not in line:continue
 try: tokens=shlex.split(line)
 except ValueError:continue
 cfg={}
 for token in tokens:
  if token.startswith('--') and '=' in token:
   key,value=token[2:].split('=',1)
   if key in allow:cfg[key]=value
 if cfg:evidence['history'].append(dict(line=lineno,flags=cfg))
evidence['history_scope']='Only allowlisted flags from lerobot-record lines; no execution success or timestamp linkage implied; full history not copied.'
source_names=['scripts/lerobot_record.py','robots/so_follower/config_so_follower.py','robots/so_follower/so_follower.py','teleoperators/so_leader/config_so_leader.py','teleoperators/so_leader/so_leader.py','motors/motors_bus.py','robots/robot.py','datasets/image_writer.py','datasets/dataset_writer.py','datasets/video_utils.py','datasets/compute_stats.py','cameras/opencv/camera_opencv.py','utils/visualization_utils.py','utils/rerun_visualization.py']
evidence['sources']=[]
(out/'source_snapshots').mkdir(exist_ok=True)
for name in source_names:
 p=repo/'src/lerobot'/name
 target=out/'source_snapshots'/name.replace('/','__');target.write_bytes(p.read_bytes())
 evidence['sources'].append(dict(path=str(p),sha256=sha(p),snapshot=str(target.relative_to(out))))
calibs=sorted(set(base.glob('calibration/**/*.json'))|set(project.glob('so101*.json'))|set((project/'artifacts/safety').glob('so101*.json')))
evidence['calibrations']=[]
for p in calibs:
 j=json.loads(p.read_text());evidence['calibrations'].append(dict(path=str(p),sha256=sha(p),mtime_utc=datetime.fromtimestamp(p.stat().st_mtime,timezone.utc).isoformat(),motors=j))
evidence['checkpoints']=[]
for name,step in [('act_orange_to_box','020000'),('smolvla_move_orange_v7','040000')]:
 cp=repo/'outputs/train'/name/'checkpoints'/step/'pretrained_model';cfg=json.loads((cp/'config.json').read_text());tc=json.loads((cp/'train_config.json').read_text())
 evidence['checkpoints'].append(dict(path=str(cp),files=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(cp.iterdir()) if p.is_file()],policy={k:cfg.get(k) for k in ['type','chunk_size','n_action_steps','input_features','output_features','normalization_mapping','pretrained_path']},training={k:tc.get(k) for k in ['dataset','steps','batch_size','seed','eval']},processors={p.name:json.loads(p.read_text()) for p in cp.glob('*processor.json')}))
# Grid consistency is supporting evidence, never substitutes for acquisition-time config/hash.
evidence['unit_grid_checks']=[]
for p in base.glob('*/*/meta/info.json'):
 root=p.parent.parent;files=sorted((root/'data').rglob('*.parquet'))
 if not files:continue
 rows=[x for f in files for x in pq.read_table(f).to_pylist()]
 result=dict(dataset=root.name)
 for key in ['observation.state','action']:
  a=np.asarray([x[key] for x in rows],dtype=np.float64)
  ticks2=a[:,:5]*4095/180
  result[key]=dict(body_half_tick_grid_max_residual=np.abs(ticks2-np.round(ticks2)).max(0).tolist(),min=a.min(0).tolist(),max=a.max(0).tolist())
 evidence['unit_grid_checks'].append(result)
evidence['raw_search']={}
for path in [project,Path('/home/murphy/project'),Path('/home/murphy/Desktop'),Path('/home/murphy/Downloads'),base]:
 files=subprocess.run(['rg','--files','--hidden',str(path),'-g','*.mp4','-g','*.avi','-g','*.mkv','-g','*.png','-g','*.jpg','-g','*.jpeg','-g','!**/.venv/**','-g','!**/.git/**','-g','!**/r0-audit-*/**'],capture_output=True,text=True)
 evidence['raw_search'][str(path)]=files.stdout.splitlines()
dump(out/'provenance.json',evidence)
print('saved provenance; history candidates',len(evidence['history']),'calibrations',len(calibs))
