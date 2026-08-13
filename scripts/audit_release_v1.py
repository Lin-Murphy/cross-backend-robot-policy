"""Read-only checks for the standalone pure-simulation v1 candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
sys.dont_write_bytecode = True
from html.parser import HTMLParser
from pathlib import Path

DENIED_SUFFIXES = {".safetensors", ".pt", ".pth", ".ckpt", ".pyc"}
VIDEOS = [
    "artifacts/sim-cube-v1-demo-20260925T211024Z-LiPrzX/run/dual-camera.mp4",
    "artifacts/sim-cube-v1-20260925/C07-model-trials/act/dual-camera.mp4",
    "artifacts/sim-cube-v1-20260925/C07-model-trials/smolvla/dual-camera.mp4",
    "artifacts/sim-v1-act-pilot-20260921/run01/episode_000/rollout.mp4",
    "artifacts/sim-v1-act-pilot-20260921/run01/episode_001/rollout.mp4",
]

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.paths = []
    def handle_starttag(self, tag, attrs):
        if tag in {"a", "video", "source"}:
            fields = dict(attrs)
            path = fields.get("href") or fields.get("src")
            if path and not path.startswith(("https:", "http:", "#")):
                self.paths.append(path)

def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def audit(root: Path) -> dict:
    manifest = json.loads((root / "release-manifest.json").read_text())
    declared = {item["path"]: item for item in manifest["files"]}
    actual = {p.relative_to(root).as_posix(): p for p in root.rglob("*") if p.is_file()}
    assert len(declared) == manifest["file_count"]
    assert set(actual) == set(declared) | {"release-manifest.json"}, "manifest does not match release directory"
    assert sum(item["bytes"] for item in declared.values()) == manifest["total_bytes"]
    for name, path in actual.items():
        assert not path.is_symlink(), name
        assert path.suffix not in DENIED_SUFFIXES, name
        if name in declared:
            assert path.stat().st_size == declared[name]["bytes"], name
            assert file_hash(path) == declared[name]["sha256"], name
    page = root / "artifacts/sim-cube-v1-20260925/model-evidence/index.html"
    parser = Links(); parser.feed(page.read_text())
    assert len(parser.paths) == 10
    for link in parser.paths:
        assert (page.parent / link).is_file(), link
    for relative in VIDEOS:
        video = root / relative
        assert video.is_file(), relative
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-f", "null", "-"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    fixed = json.loads((root / "artifacts/sim-cube-v1-demo-20260925T211024Z-LiPrzX/audit.json").read_text())
    act = json.loads((root / "artifacts/sim-cube-v1-20260925/C07-model-trials/act/summary.json").read_text())
    vla = json.loads((root / "artifacts/sim-cube-v1-20260925/C07-model-trials/smolvla/summary.json").read_text())
    pusht_ok = json.loads((root / "artifacts/sim-v1-act-pilot-20260921/run01/episode_000/summary.json").read_text())
    pusht_fail = json.loads((root / "artifacts/sim-v1-act-pilot-20260921/run01/episode_001/summary.json").read_text())
    assert fixed["status"] == "passed_cube_scripted_demo_audit" and fixed["control_steps"] == 799 and not fixed["policy_used"]
    assert act["selected_actions"] == 0 and "development_step_limit" in act["error"]
    assert vla["selected_actions"] == 553 and "policy_target_outside_calibration_bounds" in vla["error"]
    assert pusht_ok["success"] and pusht_ok["steps"] == 134
    assert not pusht_fail["success"] and pusht_fail["steps"] == 300
    real = None
    if manifest.get('release') == 'cross-backend-v1-local-candidate':
        sys.path.insert(0, str(root / 'src'))
        sys.path.insert(0, str(root / 'scripts'))
        from verify_so101_shared_success_offline import verify
        from audit_so101_shared_live_boundary import audit as audit_real_packets
        real = verify(root / 'artifacts/so101-shared-success-freeze-20260927/manifest.json', root)
        if real['real_goal_packets'] != 494 or real['RunRecord'] != 'success':
            raise ValueError('SO101 real success verification failed')
        expected_reasons = (
            ['target_outside_trial_profile:wrist_flex'],
            ['target_from_previous_goal_exceeded:wrist_flex'],
            ['target_from_feedback_exceeded:wrist_flex', 'target_outside_trial_profile:wrist_roll'],
        )
        for index, reasons in enumerate(expected_reasons, 1):
            stem = f'attempt{index:02d}'
            run = root / f'artifacts/so101-act-shared-real-pilot-20260927-{stem}'
            preflight = root / ('artifacts/so101-act-final-scene-20260927/summary.json' if index == 1 else
                                f'artifacts/so101-act-{stem}-final-scene-20260927/summary.json')
            check = audit_real_packets(run, preflight)
            summary = json.loads((run / 'summary.json').read_text())
            evaluation = json.loads((root / f'artifacts/so101-act-{stem}-evaluation-20260927/evaluation.json').read_text())
            if check['paired_observations_actions_packets'] != 50 or not check['stop_hold_after_last_receipt'] or \
               summary['policy'] != 'act' or summary['rejection_reasons'] != reasons or \
               evaluation['record']['policy'] != 'ACT' or evaluation['record']['status'] != 'aborted' or \
               evaluation['record']['outcome'] is not None or evaluation['formal_trial_count'] != 0:
                raise ValueError(f'ACT {stem} evidence mismatch')
    return {"status": "passed_release_v1_audit", "files": len(declared), "bytes": manifest["total_bytes"], "page_links": len(parser.paths), "videos_decoded": len(VIDEOS), "cube_fixed_steps": fixed["control_steps"], "act_cube_accepted": act["selected_actions"], "smolvla_cube_accepted": vla["selected_actions"], "pusht_act_pilot_successes": 1, "pusht_act_pilot_episodes": 2, "weights_included": False, "hardware_access": False,
            "real_success_packets": None if real is None else real["real_goal_packets"],
            "act_real_development_runs": 0 if real is None else 3}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
