"""Replay saved historical action/state pairs through a proposed guard; NO robot."""
import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
from cross_backend.legacy_so101_sync_guard import LegacyTrialProfile,LegacySyncGuard,LegacyGuardRejected,NAMES

DATA=Path('/home/murphy/.cache/huggingface/lerobot/local/rollout_eval_move_pot_smolvla_40k_20260923_211017/data/chunk-000/file-000.parquet')
CAL=Path('/home/murphy/.cache/huggingface/lerobot/calibration/robots/so_follower/so101_follower_arm.json')
PROFILE=Path('configs/so101-smolvla-legacy30-audit-proposal-20260927.json')


def to_raw(vector,cal):
    values=[]
    for i,n in enumerate(NAMES):
        c=cal[n];v=float(vector[i])
        if i<5:values.append(int(v*4095/360+(c['range_min']+c['range_max'])/2))
        else:values.append(int(max(0,min(100,v))*(c['range_max']-c['range_min'])/100+c['range_min']))
    return values


def main(out):
    c=json.loads(PROFILE.read_text());cal=json.loads(CAL.read_text())
    keys={f.name for f in fields(LegacyTrialProfile)}
    limits=LegacyTrialProfile(**{k:tuple(c[k]) if isinstance(c[k],list) else c[k] for k in keys}).validate()
    data=pq.read_table(DATA,columns=['action','observation.state','timestamp','episode_index']).to_pylist()
    results=[]
    for episode in (0,1,2,4,6):
        now=[0];events=[];guard=LegacySyncGuard(limits,events.append,lambda:now[0])
        accepted=0;failure=None
        for item in data:
            if item['episode_index']!=episode:continue
            now[0]=1_000_000_000+round(item['timestamp']*1e9)
            state=to_raw(item['observation.state'],cal)
            try:
                guard.observe(dict(zip(NAMES,state)),observed_ns=now[0])
                now[0]+=5_000_000
                target=to_raw(item['action'],cal)
                packet={i:x for i,x in enumerate(target,1)}
                at,raw=guard.check_goal_packet(42,2,packet)
                guard.accepted_transmit(at,raw)
                accepted+=1
            except LegacyGuardRejected as exc:
                failure={'frame_index':accepted,'reasons':list(exc.reasons)}
                break
        results.append({'episode':episode,'accepted_saved_frames':accepted,'first_rejection':failure})
    payload={'status':'offline_replay_only','real_hardware_dispatches':0,
             'saved_action_not_verified_raw_motor_write':True,
             'source_parquet_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
             'profile_sha256':hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
             'results':results}
    (out/'summary.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    main(args.output)
