"""Read-only analysis and checkpoint replay for the completed R1 short pilot."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import time
import traceback

for name in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE'):
    os.environ[name]='1'
os.environ['WANDB_MODE']='disabled'
PROJECT=Path(__file__).resolve().parents[1]
RUN=PROJECT/'artifacts/r1-act-short-pilot-20260923-run01'
OUT=PROJECT/'artifacts/r1-act-short-pilot-review-20260923'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(name,obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n')

def main():
    import draccus
    import numpy as np
    import torch
    from safetensors import safe_open
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies import make_pre_post_processors
    from lerobot.policies.act.processor_act import make_act_pre_post_processors
    torch.set_num_threads(4)
    def deny(*args,**kwargs):raise RuntimeError('Offline checkpoint verification forbids network/download')
    torch.hub.download_url_to_file=deny
    original_connect=socket.socket.connect
    def offline_connect(sock,address):
        if sock.family in (socket.AF_INET,socket.AF_INET6):deny()
        return original_connect(sock,address)
    socket.socket.connect=offline_connect
    inputs={str(f):sha(f) for f in RUN.rglob('*') if f.is_file()}
    save('run_hashes_before.json',inputs)
    for line in (RUN/'train-files.sha256').read_text().splitlines():
        digest,path=line.split('  ',1)
        assert sha(path)==digest,path
    preflight=json.loads((PROJECT/'artifacts/r1-act-preflight-20260923/attempt02/input_hashes_before.json').read_text())
    assert all(sha(p)==h for p,h in preflight.items())
    result=json.loads((RUN/'process-result.json').read_text())
    assert result['exit_code']==0
    metrics=[]
    log=(RUN/'train.log').read_text()
    for line in log.splitlines():
        if 'ot_train.py:641' not in line:continue
        values={k:float(v) for k,v in re.findall(r'(step|loss|grdn|lr|updt_s|data_s|mem_gb|l1_loss|kld_loss):([^\s]+)',line)}
        assert all(math.isfinite(v) for v in values.values())
        metrics.append(values)
    assert [int(m['step']) for m in metrics]==list(range(10,201,10))
    assert 'End of training' in log
    steady=[m for m in metrics if m['step']>20]
    mean_step=sum(m['updt_s']+m['data_s'] for m in steady)/len(steady)
    gpu=list(csv.DictReader((RUN/'gpu.csv').open()))
    used=[float(r[' memory.used [MiB]'].strip().split()[0]) for r in gpu]
    checkpoints={}
    for step in (100,200):
        cp=RUN/f'train/checkpoints/{step:06d}'
        actual_step=json.loads((cp/'training_state/training_step.json').read_text())
        assert actual_step['step']==step
        for name in ('optimizer_state.safetensors','optimizer_param_groups.json','rng_state.safetensors'):
            assert (cp/'training_state'/name).stat().st_size>0
        # Validate serialized model and optimizer tensors rather than relying only on filenames.
        tensors={}
        for relative in ('pretrained_model/model.safetensors','training_state/optimizer_state.safetensors'):
            count=0
            with safe_open(cp/relative,framework='pt',device='cpu') as archive:
                for key in archive.keys():
                    assert torch.isfinite(archive.get_tensor(key)).all(),key
                    count+=1
            tensors[relative]=count
        checkpoints[str(step)]={'training_step':actual_step,'finite_tensor_counts':tensors,
                                'model_sha256':sha(cp/'pretrained_model/model.safetensors')}
    assert checkpoints['100']['model_sha256']!=checkpoints['200']['model_sha256']
    summary={'status':'training_completed','actual_updates':200,'batch_size':8,'sampled_anchors':1600,
             'wall_s_rounded':result['wall_s'],'wall_s_time_command':45.70,
             'whole_run_updates_per_s':200/45.70,'whole_run_samples_per_s':1600/45.70,
             'steady_scope':'steps21-200 from 18 rounded 10-step log windows; excludes startup/checkpoint overhead',
             'steady_mean_step_s':mean_step,'steady_updates_per_s':1/mean_step,'steady_samples_per_s':8/mean_step,
             'training_peak_allocated_gib_rounded':max(m['mem_gb'] for m in metrics),
             'gpu_whole_device_peak_mib_sampled':max(used),'gpu_samples':len(gpu),
             'first_loss_window':metrics[0],'last_loss_window':metrics[-1],
             'finite_logged_metrics':True,'metrics_are_window_aggregates_not_per_step_trace':True,
             'linear_40000_update_minutes_excluding_overhead':40000*mean_step/60,
             'checkpoints':checkpoints}
    save('training_summary.json',summary);save('training_windows.json',metrics)
    print(json.dumps(summary,indent=2),flush=True)
    cp=RUN/'train/checkpoints/000200/pretrained_model'
    cfg=ACTConfig.from_pretrained(str(cp),local_files_only=True)
    cfg.device='cuda';cfg.pretrained_backbone_weights=None
    assert cfg.chunk_size==cfg.n_action_steps==50 and cfg.temporal_ensemble_coeff is None
    model=ACTPolicy.from_pretrained(str(cp),config=cfg,local_files_only=True,strict=True).eval()
    assert all(torch.isfinite(p).all() for p in model.parameters())
    with safe_open(cp/'model.safetensors',framework='pt',device='cpu') as archive:
        for key,value in model.state_dict().items():
            torch.testing.assert_close(value.cpu(),archive.get_tensor(key),rtol=0,atol=0)
    pre,post=make_pre_post_processors(cfg,pretrained_path=str(cp),
        preprocessor_overrides={'device_processor':{'device':'cuda'}},
        postprocessor_overrides={'device_processor':{'device':'cpu'}})
    train_cfg=draccus.decode(TrainPipelineConfig,json.loads((cp/'train_config.json').read_text()))
    assert train_cfg.steps==200 and train_cfg.batch_size==8 and not train_cfg.wandb.enable
    train_cfg.dataset.image_transforms.enable=False
    ds=make_dataset(train_cfg)
    fresh_pre,fresh_post=make_act_pre_post_processors(cfg,ds.meta.stats)
    obs_records=[]
    for index in (0,5620,10768):
        item=ds[index]
        raw={k:item[k].float()/255 if k in cfg.image_features else item[k] for k in cfg.input_features}
        batch=pre(raw); expected=fresh_pre(raw)
        for key in raw:torch.testing.assert_close(batch[key],expected[key],rtol=0,atol=0)
        with torch.inference_mode():
            model.reset();pre.reset();post.reset()
            normalized=model.predict_action_chunk(batch)
            actions=post(normalized)
            torch.testing.assert_close(actions,fresh_post(normalized),rtol=0,atol=0)
            model.reset()
            native=torch.stack([post(model.select_action(batch)) for _ in range(50)],dim=1)
            assert len(model._action_queue)==0
            torch.testing.assert_close(native,actions,rtol=0,atol=1e-5)
            model.select_action(batch)
            assert len(model._action_queue)==49
            model.reset();pre.reset();post.reset()
            assert len(model._action_queue)==0
            torch.testing.assert_close(post(model.select_action(batch)),actions[:,0],rtol=0,atol=1e-5)
            model.reset()
        assert actions.shape==(1,50,6) and torch.isfinite(actions).all()
        arrays={k.replace('.','_'):v.numpy() for k,v in raw.items()}
        arrays.update(actions=actions.numpy(),normalized_actions=normalized.cpu().numpy())
        np.savez_compressed(OUT/f'input-output-{index:05d}.npz',**arrays)
        obs_records.append({'dataset_index':index,'episode':int(item['episode_index']),
                            'dataset_timestamp_s':float(item['timestamp']),
                            'queue_max_abs_error':float((native-actions).abs().max()),
                            'action_shape':list(actions.shape),'action_min':actions.min(dim=1).values.tolist(),
                            'action_max':actions.max(dim=1).values.tolist(),
                            'hardware_dispatch':False})
    with torch.inference_mode():
        for _ in range(5):model.predict_action_chunk(batch)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        times=[]
        for _ in range(10):
            start=time.perf_counter_ns();model.predict_action_chunk(batch);torch.cuda.synchronize()
            times.append((time.perf_counter_ns()-start)/1e6)
    save('checkpoint_replay.json',{'status':'passed','checkpoint':str(cp),
         'checkpoint_sha256':sha(cp/'model.safetensors'),'strict_load_and_exact_tensor_match':True,
         'saved_processor_matches_training_statistics':True,'native_queue_and_partial_reset':'passed',
         'observations':obs_records,'warmup_forwards':5,'timed_forwards_ms':times,
         'forward_median_ms':float(np.median(times)),
         'replay_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
         'timing_scope':'GPU synchronized batch1 forward on last recorded sample, no processing/dispatch; short-run checkpoint only',
         'task_success_evaluated':False,'training_updates_during_review':0})
    assert all(sha(p)==h for p,h in inputs.items())
    assert all(sha(p)==h for p,h in preflight.items())
    save('integrity.json',{'run_files_unchanged':len(inputs),'preflight_inputs_unchanged':len(preflight),'all_unchanged':True})
    print('PASS strict checkpoint reload, saved processors, three recorded inputs, queues/reset and integrity',flush=True)

if __name__=='__main__':
    OUT.mkdir(exist_ok=False)
    (OUT/'executed_script.py').write_bytes(Path(__file__).read_bytes())
    try:main()
    except BaseException:
        (OUT/'failure.txt').write_text(traceback.format_exc());raise
