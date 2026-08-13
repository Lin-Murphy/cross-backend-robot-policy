"""Run cached SmolVLA on saved, post-alignment SO101 capture; never open robot devices."""
import argparse
import json
from pathlib import Path
import socket
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main(capture,out):
    import numpy as np
    import torch
    from cross_backend.smolvla_move_pot import SmolVLAMovePotAdapter
    from cross_backend.move_pot_policy import MovePotObservation,CAMERAS
    original=socket.socket.connect
    def offline(sock,address):
        if sock.family in (socket.AF_INET,socket.AF_INET6):raise RuntimeError('network disabled')
        return original(sock,address)
    socket.socket.connect=offline
    cp=Path('/home/murphy/project/lerobot/outputs/train/smolvla_move_pot_dualcam_20260923_v3/checkpoints/040000/pretrained_model')
    summary=json.loads((capture/'summary.json').read_text())
    assert summary['status']=='passed' and summary['dispatches']==0
    arrays=np.load(capture/'sample-00.npz')
    state=arrays['state'].astype(np.float32)
    obs=MovePotObservation({k:arrays[k].copy() for k in CAMERAS},state,'so101-post-alignment:0',
        {k:'so101-post-alignment:0:'+k for k in CAMERAS},{k:None for k in CAMERAS},None,task='move pot')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    adapter=SmolVLAMovePotAdapter(cp,device,seed=1000)
    start=time.perf_counter();chunk=adapter.predict_chunk(obs);elapsed=time.perf_counter()-start
    actions=chunk.actions
    np.savez_compressed(out/'prediction.npz',actions=actions,source_state=state)
    cal=summary['hardware_calibration'];names=list(cal)
    lo=[];hi=[]
    for i,n in enumerate(names):
        c=cal[n]
        if i<5:
            mid=(c['range_min']+c['range_max'])/2
            lo.append((c['range_min']-mid)*360/4095)
            hi.append((c['range_max']-mid)*360/4095)
        else:lo.append(0);hi.append(100)
    result={'status':'passed','hardware_dispatches':0,'device':device,'checkpoint':str(cp),
        'source_observation':str(capture/'sample-00.npz'),'state':state.tolist(),
        'first_action':actions[0].tolist(),'first_delta_from_state':(actions[0]-state).tolist(),
        'chunk_min':actions.min(axis=0).tolist(),'chunk_max':actions.max(axis=0).tolist(),
        'calibration_lower':lo,'calibration_upper':hi,
        'outside_calibration_per_joint':((actions<lo)|(actions>hi)).sum(axis=0).tolist(),
        'inference_seconds':elapsed,'source':chunk.source}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('status','device','first_delta_from_state','outside_calibration_per_joint','inference_seconds')},indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    try:main(a.capture,a.output)
    except BaseException:(a.output/'failure.txt').write_text(traceback.format_exc());raise
