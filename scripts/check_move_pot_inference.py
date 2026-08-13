"""Bounded offline inference from recorded data; no hardware interfaces."""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
import json
import time
from pathlib import Path
import av
import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

root = Path('/home/murphy/.cache/huggingface/lerobot/local/move_pot_20260923_142914')
ckpt = Path('/home/murphy/project/lerobot/outputs/train/smolvla_move_pot_dualcam_20260923_v3/checkpoints/040000/pretrained_model')
out = Path('artifacts/move-pot-offline-inference-040000')
out.mkdir(parents=True, exist_ok=False)
rows = [r for p in sorted((root/'data').rglob('*.parquet')) for r in pq.read_table(p).to_pylist()]
episodes = [r for p in sorted((root/'meta/episodes').rglob('*.parquet')) for r in pq.read_table(p).to_pylist()]
tasks = {r['task_index']:r['task'] for r in pq.read_table(root/'meta/tasks.parquet').to_pylist()}
samples = []
for eid in (0,14,29):
    rr = sorted([r for r in rows if r['episode_index']==eid],key=lambda r:r['frame_index'])
    row = rr[len(rr)//2]
    ep = next(e for e in episodes if e['episode_index']==eid)
    frames = {}
    for cam in ('observation.images.follower','observation.images.camera2'):
        key = 'videos/'+cam+'/'
        path = root/'videos'/cam/f"chunk-{ep[key+'chunk_index']:03d}"/f"file-{ep[key+'file_index']:03d}.mp4"
        target = ep[key+'from_timestamp']+row['timestamp']
        with av.open(str(path)) as container:
            for frame in container.decode(video=0):
                if float(frame.time) >= target-1/60:
                    frames[cam] = frame.to_ndarray(format='rgb24')
                    assert abs(float(frame.time)-target)<1/30, (frame.time,target)
                    break
        assert cam in frames
    samples.append((row,frames))
print('Three dual-camera observations decoded',flush=True)
policy = SmolVLAPolicy.from_pretrained(str(ckpt),local_files_only=True,strict=True)
policy.eval()
pre,post = make_pre_post_processors(policy.config,pretrained_path=str(ckpt))
print('Checkpoint strictly loaded; saved processors loaded',flush=True)
captured = {}
original = policy.prepare_images
def inspect_images(batch):
    images,masks = original(batch)
    captured['present_keys'] = [k for k in policy.config.image_features if k in batch]
    captured['image_shapes'] = [list(x.shape) for x in images]
    captured['masks'] = [m.detach().cpu().tolist() for m in masks]
    return images,masks
policy.prepare_images = inspect_images
train_actions = np.asarray([r['action'] for r in rows])
lo,hi = train_actions.min(0),train_actions.max(0)
results=[]
with torch.inference_mode():
    for row,frames in samples:
        policy.reset()
        torch.manual_seed(1000)
        batch = {'observation.state':torch.tensor(row['observation.state'],dtype=torch.float32),'task':tasks[row['task_index']]}
        batch.update({k:torch.from_numpy(v.copy()).permute(2,0,1).float()/255 for k,v in frames.items()})
        processed=pre(batch)
        assert all(k in processed for k in ('observation.images.camera1','observation.images.camera2'))
        torch.cuda.synchronize()
        t=time.perf_counter()
        chunk=policy.predict_action_chunk(processed)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-t
        action=post(chunk).detach().cpu().numpy()
        assert list(action.shape)==[1,50,6], action.shape
        assert np.isfinite(action).all()
        np.save(out/f"episode-{row['episode_index']:02d}-actions.npy",action)
        result={'episode':row['episode_index'],'row_index':row['index'],'frame_index':row['frame_index'],'task':tasks[row['task_index']],'state':row['observation.state'],'action_shape':list(action.shape),'finite':bool(np.isfinite(action).all()),'inference_seconds':elapsed,'min':action.min(axis=(0,1)).tolist(),'max':action.max(axis=(0,1)).tolist(),'first_action':action[0,0].tolist(),'outside_training_range_by_joint':((action<lo)|(action>hi)).sum(axis=(0,1)).tolist(),'camera_inputs':dict(captured)}
        results.append(result)
        print(json.dumps(result),flush=True)
summary={'checkpoint':str(ckpt),'dataset':str(root),'strict_load':True,'offline_only':True,'training_data_samples_not_independent_evaluation':True,'empirical_action_min':lo.tolist(),'empirical_action_max':hi.tolist(),'results':results}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print('Offline checks finished',flush=True)
