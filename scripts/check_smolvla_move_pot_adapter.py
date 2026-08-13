"""Bounded CPU-only replay of the local SmolVLA checkpoint, no downloads or training."""
import os
for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[key]='1'
os.environ['WANDB_MODE']='disabled'
import argparse
import json
from pathlib import Path
import socket
import sys
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main(out):
    import numpy as np
    import torch
    from cross_backend.smolvla_move_pot import SmolVLAMovePotAdapter
    from cross_backend.move_pot_policy import MovePotObservation,ReplayChunkQueue,CAMERAS,file_sha256
    torch.set_num_threads(2)
    original=socket.socket.connect
    def offline(sock,address):
        if sock.family in (socket.AF_INET,socket.AF_INET6):raise RuntimeError('No network permitted in replay')
        return original(sock,address)
    socket.socket.connect=offline
    cp=Path('/home/murphy/project/lerobot/outputs/train/smolvla_move_pot_dualcam_20260923_v3/checkpoints/040000/pretrained_model')
    before={str(p):file_sha256(p) for p in cp.iterdir() if p.is_file()}
    print('Loading cached SmolVLA strictly on CPU',flush=True)
    adapter=SmolVLAMovePotAdapter(cp,'cpu',seed=1000)
    fixture=ROOT/'artifacts/r1-parallel-preparation-20260924/pilot-acceptance-cpu/sample-00000.npz'
    arrays=np.load(fixture)
    obs=MovePotObservation({k:arrays[k] for k in CAMERAS},arrays['state'],'dataset:0',
          {k:'dataset:0:'+k for k in CAMERAS},{k:None for k in CAMERAS},None,task='move pot')
    processed=adapter._prepare(obs)
    with torch.inference_mode():images,masks=adapter.policy.prepare_images(processed)
    assert [m.tolist() for m in masks]==[[True],[True],[False],[False]]
    assert all(tuple(i.shape)==(1,3,512,512) for i in images)
    # Saved rename preserves each physical RGB image (visual normalization is IDENTITY).
    for old,new in [(CAMERAS[0],'observation.images.camera1'),(CAMERAS[1],CAMERAS[1])]:
        expected=torch.from_numpy(obs.images_rgb[old].copy()).permute(2,0,1).float().unsqueeze(0)/255
        torch.testing.assert_close(processed[new],expected,atol=0,rtol=0)
    print('PASS saved camera rename, two real/two padded masks, task tokenization',flush=True)
    adapter.reset()
    chunk=adapter.predict_chunk(obs)
    np.savez_compressed(out/'sample-0-actions.npz',actions=chunk.actions)
    q=ReplayChunkQueue(out/'events.jsonl','smolvla_offline')
    q.enqueue(chunk,50)
    np.testing.assert_array_equal(np.stack([q.pop() for _ in range(50)]),chunk.actions)
    q.reset();q.close()
    print('PASS finite 50x6 chunk and shared external queue',flush=True)
    check=adapter.verify_native_queue(obs)
    print('PASS seeded native queue equivalence',flush=True)
    # Reset must clear a partial native queue and restore the first sample's noise/output.
    with torch.inference_mode():
        batch=adapter._prepare(obs);noise=adapter._noise()
        first=adapter.post(adapter.policy.select_action(batch,noise=noise)).cpu().numpy()[0]
        assert len(adapter.policy._queues['action'])==49
        adapter.reset();assert len(adapter.policy._queues['action'])==0
        replay=adapter.predict_chunk(obs)
        np.testing.assert_allclose(replay.actions,chunk.actions,atol=1e-4,rtol=0)
        np.testing.assert_allclose(first,chunk.actions[0],atol=1e-4,rtol=0)
        assert replay.source['noise_sha256']==chunk.source['noise_sha256']
    assert all(file_sha256(p)==h for p,h in before.items())
    result={'status':'passed','checkpoint':str(cp),'checkpoint_manifest':adapter.checkpoint_manifest,
        'task':obs.task,'physical_camera_keys':['observation.images.camera1','observation.images.camera2'],
        'prepared_shapes':[list(i.shape) for i in images],'masks':[m.tolist() for m in masks],
        'language_token_shape':list(processed['observation.language.tokens'].shape),
        'queue_check':check,'partial_queue_reset':'passed','first_forward_ms_cpu':chunk.forward_ms,
        'gripper_out_of_range_offsets':np.flatnonzero((chunk.actions[:,-1]<0)|(chunk.actions[:,-1]>100)).tolist(),
        'sampling_source':chunk.source,'inputs_unchanged':True,'gpu_used':False,'training_updates':0,
        'task_success_evaluated':False,'hardware_dispatches':0}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS reset, reproducible sampling and checkpoint integrity',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    try:main(args.output)
    except BaseException:
        (args.output/'failure.txt').write_text(traceback.format_exc());raise
