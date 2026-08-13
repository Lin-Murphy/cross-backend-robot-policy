#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${SIM_V1_PYTHON:-$ROOT/.venv-sim/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  printf '找不到纯仿真 Python 环境：%s\n' "$PYTHON_BIN" >&2
  exit 2
fi
if ! command -v ffmpeg >/dev/null 2>&1; then
  printf '找不到 FFmpeg。\n' >&2
  exit 2
fi
OUT="$(mktemp -d "$ROOT/artifacts/sim-cube-v1-demo-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
SCENE="$ROOT/artifacts/sim-cube-v1-20260925/scene-cube-reachable.xml"
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
"$PYTHON_BIN" -B scripts/check_sim_cube_scene.py --scene "$SCENE" --output "$OUT/preflight" | tee "$OUT/preflight.log"
"$PYTHON_BIN" -B scripts/s1_fixed_grasp.py \
  --scene "$SCENE" --initial-wrist-roll-deg 0 \
  --grasp-radial-offset-m -0.015 --grasp-height-m 0.03 \
  --place-height-m 0.036 --close-gripper-rad -0.17 \
  --side-approach-m 0.11 --output "$OUT/run" | tee "$OUT/run.log"
ffmpeg -v error -i "$OUT/run/dual-camera.mp4" -f null -
"$PYTHON_BIN" -B scripts/verify_sim_cube_v1.py "$OUT"
printf '\n方块纯仿真演示完成：%s/index.html\n' "$OUT"
