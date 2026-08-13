"""Bounded no-dispatch inference on an explicitly nominal simulation observation."""
import os
for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[key]='1'
os.environ['WANDB_MODE']='disabled'
import sys,json,socket,gc,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'artifacts/s1-model-input-diagnostic-20260925'

def main():
    import numpy as np
    import torch
    from cross_backend.move_pot_policy import ACTMovePotAdapter,MovePotObservation,ReplayChunkQueue,CAMERAS,file_sha256
    from cross_backend.smolvla_move_pot import SmolVLAMovePotAdapter
    original=socket.socket.connect
    def offline(sock,address):
        if sock.family in (socket.AF_INET,socket.AF_INET6):raise RuntimeError('Network blocked')
        return original(sock,address)
    socket.socket.connect=offline;torch.set_num_threads(2)
    data=np.load(OUT/'nominal-observation.npz');meta=json.loads((OUT/'input-metadata.json').read_text());assert meta['action_execution_allowed'] is False
    obs=MovePotObservation({k:data[k].copy() for k in CAMERAS},data['state'].copy(),'nominal-sim:0',{k:'nominal-sim:0:'+k for k in CAMERAS},{k:0 for k in CAMERAS},0)
    result={'status':'running','models':{},'hardware_access':False,'simulation_dispatches':0,'task_success_evaluated':False,'input_sha256':file_sha256(OUT/'nominal-observation.npz'),'input_assumptions':meta}
    specs=[('act',ACTMovePotAdapter,ROOT/'artifacts/r1-act-formal-40000-run02/train/checkpoints/040000/pretrained_model'),('smolvla',SmolVLAMovePotAdapter,Path('/home/murphy/project/lerobot/outputs/train/smolvla_move_pot_dualcam_20260923_v3/checkpoints/040000/pretrained_model'))]
    for name,factory,cp in specs:
        print('Loading local '+name,flush=True);adapter=factory(cp,'cuda');adapter.reset();chunk=adapter.predict_chunk(obs)
        chunk.source.update(clock_domain='simulation',capture_sim_ns=0,host_prediction_times_separate=True,physical_alignment_verified=False)
        q=ReplayChunkQueue(OUT/(name+'-events.jsonl'),'s1-nominal-input-'+name)
        try:q.enqueue(chunk,50);q.reset()
        finally:q.close()
        np.savez_compressed(OUT/(name+'-actions.npz'),actions=chunk.actions)
        result['models'][name]={'shape':list(chunk.actions.shape),'finite':bool(np.isfinite(chunk.actions).all()),'first_action':chunk.actions[0].tolist(),'max_abs_first_body_delta_deg':float(np.max(np.abs(chunk.actions[0,:5]-obs.state[:5]))),'gripper_range':[float(chunk.actions[:,-1].min()),float(chunk.actions[:,-1].max())],'forward_ms_first_call_not_benchmark':chunk.forward_ms,'checkpoint_manifest':adapter.checkpoint_manifest,'execution_rejection':'unverified physical coordinates and cameras; no candidate dispatched'}
        assert all(file_sha256(cp/f)==h for f,h in adapter.checkpoint_manifest.items())
        (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
        print(name+': finite prediction saved, all 50 candidates discarded without execution',flush=True)
        del adapter,chunk;gc.collect();torch.cuda.empty_cache()
    result['status']='passed_input_inference_only';(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
    try:main()
    except BaseException:(OUT/'failure.txt').write_text(traceback.format_exc());raise
