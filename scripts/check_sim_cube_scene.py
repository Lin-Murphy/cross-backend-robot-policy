"""Read-only-in-simulation preflight for the development cube scene."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cross_backend.png_write import write_png
from cross_backend.tape_sim_backend import TapeSimBackend
from cross_backend.tape_task import TapeTaskEvaluator
from cross_backend.tape_task_observer import observe_task


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / "artifacts/sim-cube-v1-20260925/scene-cube-reachable.xml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene = args.scene.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    backend = TapeSimBackend(scene, 1 / 30)
    try:
        backend.reset()
        cube = backend.model.geom("object_cube")
        assert backend.model.geom_bodyid[cube.id] == backend.tape_id
        assert abs(backend.model.body_mass[backend.tape_id] - .02) < 1e-9
        images = backend.render()
        for key, pixels in images.items():
            write_png(output / (key.rsplit(".", 1)[-1] + ".png"), pixels)
        state = backend.data.qpos[backend.qadr].copy()
        for _ in range(50):
            backend.step_sim_targets(state)
        facts = observe_task(backend)
        status = TapeTaskEvaluator().update(facts)
        if status != "running" or facts.zone != "outside" or not facts.supported_on_table or facts.forbidden_contact:
            raise RuntimeError("Cube initial condition is not task-ready")
        summary = {
            "status": "passed_cube_static_preflight",
            "scene": str(scene),
            "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
            "cube_edge_m": float(backend.model.geom_size[cube.id, 0] * 2),
            "cube_mass_kg": float(backend.model.body_mass[backend.tape_id]),
            "static_control_ticks": 50,
            "cube_zone": facts.zone,
            "supported_on_table": facts.supported_on_table,
            "forbidden_contact": facts.forbidden_contact,
            "hardware_access": False,
            "policy_used": False,
            "grasp_success_claimed": False,
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary))
    finally:
        backend.close()


if __name__ == "__main__":
    main()
