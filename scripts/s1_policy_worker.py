"""Offline policy-only subprocess. JSON lines on stdout; model logs on stderr."""
import os
for key in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):os.environ[key]='1'
os.environ['WANDB_MODE']='disabled'
import sys,json,socket,contextlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from cross_backend.move_pot_policy import ACTMovePotAdapter,MovePotObservation,CAMERAS,file_sha256
from cross_backend.sim_prediction_identity import verify_worker_request
from cross_backend.smolvla_move_pot import SmolVLAMovePotAdapter
original=socket.socket.connect
def offline(sock,address):
    if sock.family in (socket.AF_INET,socket.AF_INET6):raise RuntimeError('Network blocked')
    return original(sock,address)
socket.socket.connect=offline;torch.set_num_threads(2)
name=sys.argv[1];out=Path(sys.argv[2]);trial_id=sys.argv[3]
cp=ROOT/'artifacts/r1-act-formal-40000-run02/train/checkpoints/040000/pretrained_model' if name=='act' else Path('/home/murphy/project/lerobot/outputs/train/smolvla_move_pot_dualcam_20260923_v3/checkpoints/040000/pretrained_model')
with contextlib.redirect_stdout(sys.stderr):
    adapter=(ACTMovePotAdapter if name=='act' else SmolVLAMovePotAdapter)(cp,'cuda');adapter.reset()
(out/'checkpoint-manifest.json').write_text(json.dumps(adapter.checkpoint_manifest,indent=2)+'\n')
print(json.dumps({'ready':True,'trial_id':trial_id}),flush=True)
for line in sys.stdin:
    request=json.loads(line)
    if request.get('stop'):
        unchanged=all(file_sha256(cp/f)==h for f,h in adapter.checkpoint_manifest.items())
        print(json.dumps({'stopped':True,'checkpoint_unchanged':unchanged}),flush=True);break
    path=verify_worker_request(request,trial_id=trial_id,output_dir=out);data=np.load(path);stamp=request['capture_sim_ns'];identity=request['observation_id']
    obs=MovePotObservation({k:data[k].copy() for k in CAMERAS},data['state'].copy(),identity,{k:identity+':'+k for k in CAMERAS},{k:stamp for k in CAMERAS},stamp)
    with contextlib.redirect_stdout(sys.stderr):chunk=adapter.predict_chunk(obs)
    prefix=out/f"prediction-{request['index']:03d}"
    actions_path=Path(str(prefix)+'.npz')
    np.savez_compressed(actions_path,actions=chunk.actions)
    (Path(str(prefix)+'.json')).write_text(json.dumps({'trial_id':trial_id,'actions_sha256':file_sha256(actions_path),'source':chunk.source,'forward_ms':chunk.forward_ms,'processing_ms':chunk.processing_and_forward_ms,'prediction_host_start_ns':chunk.prediction_start_ns,'prediction_host_end_ns':chunk.prediction_end_ns,'input_sha256':request['input_sha256']},indent=2)+'\n')
    print(json.dumps({'actions':str(prefix)+'.npz'}),flush=True)
