"""Audit a completed cube-only scripted simulation and write a local viewer."""
import hashlib
import html
import json
import sys
from pathlib import Path


def main():
    root = Path(sys.argv[1]).resolve()
    preflight = json.loads((root / "preflight/summary.json").read_text())
    summary = json.loads((root / "run/summary.json").read_text())
    events = [json.loads(line) for line in (root / "run/events.jsonl").read_text().splitlines()]
    assert preflight["status"] == "passed_cube_static_preflight"
    assert summary["status"] == "success" and summary["policy_used"] is False
    assert summary["video_exit"] == 0 and summary["control_steps"] == len(events)
    assert [e["event"] for e in summary["task_events"]] == ["started", "lift_confirmed", "success"]
    assert all(not e["facts"]["forbidden_contact"] and not e["facts"]["numerical_error"] for e in events)
    assert any(e["facts"]["both_jaws_contact"] for e in events)
    final = events[-1]["facts"]
    assert final["zone"] == "inside" and final["supported_on_mat"]
    video = root / "run/dual-camera.mp4"
    assert video.is_file() and video.stat().st_size > 0
    audit = {
        "status": "passed_cube_scripted_demo_audit",
        "scene_sha256": preflight["scene_sha256"],
        "control_steps": len(events),
        "max_lift_clearance_m": max(e["facts"]["bottom_clearance_m"] for e in events),
        "both_jaw_contact_steps": sum(e["facts"]["both_jaws_contact"] for e in events),
        "forbidden_contact_steps": 0,
        "numerical_error_steps": 0,
        "final_zone": final["zone"],
        "final_supported_on_mat": final["supported_on_mat"],
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
        "video_decode_exit": 0,
        "policy_used": False,
        "hardware_access": False,
        "model_success_claimed": False,
    }
    (root / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    page = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>小方块纯仿真演示</title>
<style>body{max-width:1100px;margin:2rem auto;padding:0 1rem;font:18px system-ui;background:#f7f7f4;color:#222}video{width:100%;background:#111}small{color:#555}</style>
<h1>小方块抓取放置：固定脚本成功</h1>
<p>机械臂在纯仿真中夹起4厘米方块，放到绿垫内。使用双摄画面；未运行ACT或SmolVLA。</p>
<video controls preload="metadata" src="run/dual-camera.mp4"></video>
<p>控制步数：STEPS。双爪接触：CONTACTS步。终态：方块在绿垫内，由绿垫承托。</p>
<small>这是固定脚本演示，不代表训练模型成功，也不代表真机验证。原始事件与审计文件同目录保留。</small>
</html>""".replace("STEPS", html.escape(str(audit["control_steps"]))).replace("CONTACTS", html.escape(str(audit["both_jaw_contact_steps"])))
    (root / "index.html").write_text(page)
    print(json.dumps({"status": audit["status"], "output": str(root), "control_steps": len(events)}))


if __name__ == "__main__":
    main()
