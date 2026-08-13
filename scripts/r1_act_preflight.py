"""Bounded, offline ACT checks. No backward, optimizer, hardware or checkpoint writes.

Run with the existing LeRobot environment and a fresh --output directory.
The randomly initialized ACT head is only an interface fixture, never a task policy.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
import traceback

for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ[name] = "1"
os.environ["WANDB_MODE"] = "disabled"

PROJECT = Path(__file__).resolve().parents[1]
LEROBOT = Path("/home/murphy/project/lerobot")
CONFIG = PROJECT / "configs/act-move-pot-r1-prepared.json"
BACKBONE = Path("/home/murphy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def main(out):
    import av
    import draccus
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Subset
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.act.processor_act import make_act_pre_post_processors

    # HF flags do not cover torchvision; explicitly forbid network downloads too.
    def deny_download(*args, **kwargs):
        raise RuntimeError("Preflight forbids downloads")
    torch.hub.download_url_to_file = deny_download
    original_connect = socket.socket.connect
    def offline_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise RuntimeError("Preflight forbids network connections")
        return original_connect(sock, address)
    socket.socket.connect = offline_connect
    torch.set_num_threads(4)
    torch.manual_seed(1000)
    np.random.seed(1000)
    cfg = draccus.decode(TrainPipelineConfig, json.loads(CONFIG.read_text()))
    cfg.validate()
    cfg.policy.validate_features()
    root = Path(cfg.dataset.root)
    prior = json.loads((PROJECT / "artifacts/r0-refresh-20260923/summary.json").read_text())
    previous = next(d for d in prior["datasets"] if d["repo_id"] == cfg.dataset.repo_id)
    paths = [Path(f["path"]) for f in previous["files"]] + [BACKBONE, CONFIG]
    before = {str(p): sha(p) for p in paths}
    save(out / "input_hashes_before.json", before)
    assert all(before[f["path"]] == f["sha256"] for f in previous["files"])
    assert before[str(BACKBONE)] == "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
    delivery = json.loads((PROJECT / "artifacts/r0-finalization-20260923/delivery-manifest.json").read_text())
    assert all(sha(PROJECT / p) == h for p, h in delivery.items())
    revision = subprocess.check_output(["git", "-C", str(LEROBOT), "rev-parse", "HEAD"], text=True).strip()
    assert revision == "7e241bd630a3719a56157a497ce5d08f244784f1"
    save(out / "environment.json", {"python": platform.python_version(), "torch": torch.__version__,
         "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "lerobot_revision": revision,
         "lerobot_status": subprocess.check_output(["git", "-C", str(LEROBOT), "status", "--porcelain"], text=True),
         "network": "HF offline flags, socket guard and torch hub download guard", "seed": 1000})
    print("PASS identities, R0 manifest and environment", flush=True)

    # Keep real training factory/statistics, disabling augmentation only for exact alignment checks.
    plain_cfg = copy.deepcopy(cfg)
    plain_cfg.dataset.image_transforms.enable = False
    ds = make_dataset(plain_cfg)
    rows = sorted([r for p in sorted((root / "data").rglob("*.parquet"))
                   for r in pq.read_table(p).to_pylist()], key=lambda r: r["index"])
    assert len(ds) == len(rows) == 10769 and ds.num_episodes == 30
    actions = torch.tensor([r["action"] for r in rows], dtype=torch.float32)
    states = torch.tensor([r["observation.state"] for r in rows], dtype=torch.float32)
    assert torch.isfinite(actions).all() and torch.isfinite(states).all()
    assert list(cfg.policy.image_features) == ["observation.images.follower", "observation.images.camera2"]
    names = ds.features["action"]["names"]
    assert names == ds.features["observation.state"]["names"] and len(names) == 6
    cameras = list(cfg.policy.image_features)
    episodes, samples = [], []
    timestamps_max_error = 0.0
    # All stored table rows and all possible chunk index/mask boundaries, no full video rescan.
    for eid in range(30):
        indices = [i for i, r in enumerate(rows) if r["episode_index"] == eid]
        start, end = indices[0], indices[-1] + 1
        assert indices == list(range(start, end))
        for local, i in enumerate(indices):
            assert rows[i]["index"] == i and rows[i]["frame_index"] == local
            error = abs(rows[i]["timestamp"] - local / ds.fps)
            timestamps_max_error = max(timestamps_max_error, error)
            assert error < cfg.tolerance_s
            query, pad = ds.reader._get_query_indices(i, eid)
            expected = [min(i + d, end - 1) for d in range(50)]
            assert query["action"] == expected
            assert pad["action_is_pad"].tolist() == [i + d >= end for d in range(50)]
        selected = sorted({start, (start + end - 1) // 2, end - 50, end - 2, end - 1})
        for i in selected:
            item = ds[i]
            q = [min(i + d, end - 1) for d in range(50)]
            torch.testing.assert_close(item["action"], actions[q], rtol=0, atol=0)
            torch.testing.assert_close(item["observation.state"], states[i], rtol=0, atol=0)
            assert item["action_is_pad"].tolist() == [i + d >= end for d in range(50)]
            assert int(item["episode_index"]) == eid and int(item["index"]) == i
            for cam in cameras:
                assert item[cam].shape == (3, 480, 640) and item[cam].dtype == torch.uint8
            samples.append({"episode": eid, "index": i, "frame": i - start,
                            "timestamp": float(item["timestamp"]), "valid_actions": int((~item["action_is_pad"]).sum()),
                            "rgb_sha256": {cam: hashlib.sha256(item[cam].numpy().tobytes()).hexdigest() for cam in cameras}})
        episodes.append({"episode": eid, "start": start, "end_exclusive": end, "sample_indices": selected})
    save(out / "alignment.json", {"episodes": episodes, "samples": samples,
         "all_chunk_boundaries_checked": len(rows), "timestamp_max_error_s": timestamps_max_error,
         "joint_names": names, "scope": "stored alignment, not physical camera capture synchronization"})
    print(f"PASS all {len(rows)} chunk boundaries; {len(samples)} decoded dual-camera samples", flush=True)

    # Independent RGB decoder, first/middle/last episode. Record codec numerical differences.
    rgb_checks = []
    for eid in (0, 15, 29):
        i = episodes[eid]["start"]
        item = ds[i]
        ep = ds.meta.episodes[eid]
        for cam in cameras:
            target = ep[f"videos/{cam}/from_timestamp"] + rows[i]["timestamp"]
            path = root / ds.meta.get_video_file_path(eid, cam)
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                container.seek(max(0, int(target / stream.time_base)), stream=stream)
                frame = next(f for f in container.decode(stream) if abs(float(f.pts * f.time_base) - target) < 1e-4)
                rgb = frame.to_ndarray(format="rgb24")
            actual = item[cam].permute(1, 2, 0).numpy()
            delta = np.abs(actual.astype(int) - rgb.astype(int))
            bgr_delta = np.abs(actual.astype(int) - rgb[:, :, ::-1].astype(int))
            assert delta.mean() < 2 and delta.mean() < bgr_delta.mean()
            Image.fromarray(actual).save(out / f"ep{eid:02d}-{cam.rsplit('.', 1)[-1]}.png")
            rgb_checks.append({"episode": eid, "camera": cam, "video_timestamp": target,
                               "rgb_mean_abs_error": float(delta.mean()), "rgb_max_error": int(delta.max()),
                               "bgr_mean_abs_error": float(bgr_delta.mean())})
    save(out / "rgb_decoder_check.json", rgb_checks)

    pre, post = make_act_pre_post_processors(cfg.policy, ds.meta.stats)
    batch_indices = [0, 1, 2, 3] + list(range(episodes[0]["end_exclusive"] - 4, episodes[0]["end_exclusive"]))
    batch = next(iter(DataLoader(Subset(ds, batch_indices), batch_size=8, num_workers=0, shuffle=False)))
    for cam in cameras:  # Exact lerobot_train.py conversion before the processor.
        batch[cam] = batch[cam].float() / 255.0
    processed = pre(batch)
    normalization = {}
    for key in cameras + ["observation.state", "action"]:
        stats = ds.meta.stats[key]
        mean = torch.as_tensor(stats["mean"], dtype=batch[key].dtype)
        std = torch.as_tensor(stats["std"], dtype=batch[key].dtype)
        assert torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()
        expected = (batch[key].cpu() - mean.cpu()) / (std.cpu() + 1e-8)
        torch.testing.assert_close(processed[key].cpu(), expected, rtol=1e-5, atol=1e-5)
        normalization[key] = {"mean": mean.tolist(), "std": std.tolist(),
                              "max_abs_error": float((processed[key].cpu() - expected).abs().max())}
    restored = post(processed["action"])
    torch.testing.assert_close(restored, batch["action"], rtol=1e-5, atol=1e-5)
    save(out / "normalization.json", {"features": normalization,
         "action_roundtrip_max_error": float((restored - batch["action"]).abs().max()),
         "statistics_scope": "all 30 train/development episodes; ImageNet visual mean/std"})
    augmented = make_dataset(cfg)
    aug_batch = next(iter(DataLoader(Subset(augmented, batch_indices), batch_size=8, num_workers=0, shuffle=False)))
    aug_dtypes = {cam: str(aug_batch[cam].dtype) for cam in cameras}
    for cam in cameras:
        if aug_batch[cam].dtype == torch.uint8:
            aug_batch[cam] = aug_batch[cam].float() / 255
        assert torch.isfinite(aug_batch[cam]).all() and aug_batch[cam].min() >= 0 and aug_batch[cam].max() <= 1
    torch.testing.assert_close(aug_batch["action"], batch["action"], rtol=0, atol=0)
    torch.testing.assert_close(aug_batch["observation.state"], batch["observation.state"], rtol=0, atol=0)
    aug_processed = pre(aug_batch)
    print("PASS independent RGB decoding, normalization, action roundtrip and augmented batch", flush=True)

    torch.cuda.reset_peak_memory_stats()
    policy = ACTPolicy(cfg.policy).to(cfg.policy.device).eval()
    state_before = {key: value.detach().clone() for key, value in policy.state_dict().items()}
    obs = {key: processed[key][:1] for key in cfg.policy.input_features}
    with torch.inference_mode():
        torch.cuda.synchronize()
        started = time.perf_counter()
        chunk = policy.predict_action_chunk(obs)
        torch.cuda.synchronize()
        first_forward_ms = (time.perf_counter() - started) * 1000
        assert chunk.shape == (1, 50, 6) and torch.isfinite(chunk).all()
        policy.reset()
        native = torch.stack([policy.select_action(obs) for _ in range(50)], dim=1)
        torch.testing.assert_close(native, chunk, rtol=0, atol=1e-5)
        assert len(policy._action_queue) == 0
        different = {key: processed[key][1:2] for key in cfg.policy.input_features}
        refill = policy.select_action(different)
        assert len(policy._action_queue) == 49
        expected_refill = policy.predict_action_chunk(different)[:, 0]
        torch.testing.assert_close(refill, expected_refill, rtol=0, atol=1e-5)
        policy.reset(); pre.reset(); post.reset()
        assert len(policy._action_queue) == 0
        after_reset = policy.select_action(obs)
        torch.testing.assert_close(after_reset, chunk[:, 0], rtol=0, atol=1e-5)
        policy.reset()
        # Training-mode forward without autograd/updates: exercises VAE and padded loss.
        policy.train()
        loss, loss_dict = policy(aug_processed)
        assert torch.isfinite(loss)
    # eval-mode inference must not mutate learned state; training-mode BatchNorm uses FrozenBN here.
    assert all(torch.equal(state_before[k], v) for k, v in policy.state_dict().items())
    assert all(p.grad is None for p in policy.parameters())
    np.savez_compressed(out / "interface_fixture_NOT_TRAINED.npz",
                        normalized_chunk=chunk.cpu().numpy(), native_queue=native.cpu().numpy(),
                        state=batch["observation.state"][:1].numpy())
    save(out / "model_checks.json", {"identity": "random ACT head with cached ImageNet ResNet18; NOT a task checkpoint",
         "parameters": sum(p.numel() for p in policy.parameters()), "chunk_shape": list(chunk.shape),
         "queue_max_error": float((native-chunk).abs().max()), "refill_and_partial_reset": "passed",
         "first_forward_ms": first_forward_ms, "batch8_no_grad_train_mode_loss": float(loss), "loss_components": loss_dict,
         "augmented_batch_dtypes": aug_dtypes, "batch_indices": batch_indices,
         "batch_valid_action_counts": (~batch["action_is_pad"]).sum(dim=1).tolist(), "peak_allocated_mib": torch.cuda.max_memory_allocated()/2**20,
         "peak_reserved_mib": torch.cuda.max_memory_reserved()/2**20,
         "resource_scope": "preflight only; includes diagnostic state clone, no backward or optimizer; not training throughput",
         "optimizer_steps": 0, "backward_calls": 0, "model_state_unchanged": True})
    after = {p: sha(p) for p in before}
    save(out / "input_integrity.json", {"all_unchanged": before == after, "files": after})
    assert before == after
    print("PASS model forward, native queue, refill/reset, loss; zero optimizer steps; inputs unchanged", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "executed_script.py").write_bytes(Path(__file__).read_bytes())
    start = time.time()
    try:
        main(args.output)
    except BaseException:
        (args.output / "failure.txt").write_text(traceback.format_exc())
        save(args.output / "result.json", {"status": "failed", "elapsed_s": time.time()-start})
        raise
    save(args.output / "result.json", {"status": "passed", "elapsed_s": time.time()-start,
         "training_started": False, "hardware_access": False, "downloads": False})
