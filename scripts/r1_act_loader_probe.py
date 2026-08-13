"""Two-batch spawn-worker probe; no model, optimizer, network or hardware."""
import argparse
import json
import os
from pathlib import Path
import time

for key in ('HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE'):
    os.environ[key] = '1'

def main():
    import draccus
    import torch
    from torch.utils.data import DataLoader
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.scripts.lerobot_train import _dataloader_worker_kwargs
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = draccus.decode(TrainPipelineConfig, json.loads(Path('configs/act-move-pot-r1-prepared.json').read_text()))
    torch.manual_seed(cfg.seed)
    ds = make_dataset(cfg)
    loader = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers,
                        shuffle=True, pin_memory=True, timeout=90,
                        **_dataloader_worker_kwargs(cfg))
    results = []
    start = time.perf_counter()
    it = iter(loader)
    for _ in range(2):
        batch = next(it)
        assert batch['action'].shape == (8, 50, 6)
        for camera in cfg.policy.image_features:
            assert batch[camera].shape == (8, 3, 480, 640)
            assert batch[camera].dtype == torch.uint8
        assert torch.isfinite(batch['action']).all()
        assert torch.isfinite(batch['observation.state']).all()
        results.append({'indices': batch['index'].tolist(), 'episodes': batch['episode_index'].tolist(),
                        'action_shape': list(batch['action'].shape), 'elapsed_s': time.perf_counter()-start})
    del it, loader
    args.output.write_text(json.dumps({'status': 'passed', 'workers': cfg.num_workers,
        'multiprocessing_context': cfg.dataloader_multiprocessing_context,
        'prefetch_factor': cfg.prefetch_factor, 'persistent_workers': cfg.persistent_workers,
        'augmentation': True, 'batches': results, 'training_started': False}, indent=2)+'\n')
    print(args.output.read_text())

if __name__ == '__main__':
    main()
