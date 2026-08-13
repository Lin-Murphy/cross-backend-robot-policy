"""Build the archived SO101-only bootstrap of the pure-simulation v1 directory.

The current release/v1 also includes ALOHA evidence added after this bootstrap.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "release" / "v1"
FILES = [
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "requirements-sim-v1.txt",
    "scripts/run_sim_cube_v1.sh",
    "scripts/check_sim_cube_scene.py",
    "scripts/s1_fixed_grasp.py",
    "scripts/verify_sim_cube_v1.py",
    "src/cross_backend/__init__.py",
    "src/cross_backend/png_write.py",
    "src/cross_backend/tape_sim_backend.py",
    "src/cross_backend/tape_task.py",
    "src/cross_backend/tape_task_observer.py",
    "artifacts/sim-cube-v1-20260925/scene-cube-reachable.xml",
    "artifacts/sim-cube-v1-20260925/C07-cube-model-start.json",
    "artifacts/sim-cube-v1-20260925/C07-model-preflight/preflight.json",
    "artifacts/sim-cube-v1-20260925/model-evidence/index.html",
    "artifacts/sim-v1-act-pilot-20260921/run01/verification.json",
]
TREES = [
    "artifacts/s0-inventory-20260925/upstream/assets",
    "artifacts/sim-cube-v1-demo-20260925T211024Z-LiPrzX",
    "artifacts/sim-cube-v1-20260925/C07-model-trials",
    "artifacts/sim-v1-act-pilot-20260921/run01/episode_000",
    "artifacts/sim-v1-act-pilot-20260921/run01/episode_001",
]
ASSET_META = [
    "artifacts/s0-inventory-20260925/upstream/LICENSE",
    "artifacts/s0-inventory-20260925/upstream/README.md",
]
EXCLUDE_SUFFIXES = {".pyc", ".safetensors", ".pt", ".pth", ".ckpt"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def copy_one(relative: str, output: Path) -> None:
    source = ROOT / relative
    if not source.is_file() or source.is_symlink():
        raise ValueError(f"missing or symlinked input: {relative}")
    if source.suffix in EXCLUDE_SUFFIXES:
        raise ValueError(f"weight or bytecode input: {relative}")
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def build(output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"release output already exists: {output}")
    output.mkdir(parents=True)
    for relative in FILES + ASSET_META:
        copy_one(relative, output)
    for source_name, target_name in (("README.md", "README.md"), ("EVIDENCE.md", "EVIDENCE.md"), (".gitignore", ".gitignore")):
        source = ROOT / "docs" / "release-v1" / source_name
        shutil.copy2(source, output / target_name)
    for relative in TREES:
        source = ROOT / relative
        if not source.is_dir():
            raise ValueError(f"missing tree: {relative}")
        for item in sorted(source.rglob("*")):
            if item.is_symlink():
                raise ValueError(f"symlink in release source: {item}")
            if item.is_file() and item.suffix not in EXCLUDE_SUFFIXES:
                copy_one(item.relative_to(ROOT).as_posix(), output)
    files = []
    for item in sorted(output.rglob("*")):
        if item.is_file():
            files.append({"path": item.relative_to(output).as_posix(), "bytes": item.stat().st_size, "sha256": sha256(item)})
    manifest = {
        "release": "pure-simulation-v1-legacy-bootstrap",
        "scope": "SO101 cube scripted 3D demo plus saved ACT/SmolVLA cube probes and distinct ACT PushT pilot",
        "files": files,
        "file_count": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "weights_included": False,
        "hardware_access": False,
        "archive_created": False,
    }
    (output / "release-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    args = parser.parse_args()
    result = build(args.output.resolve())
    print(json.dumps({key: result[key] for key in ("release", "file_count", "total_bytes", "weights_included", "archive_created")}))
