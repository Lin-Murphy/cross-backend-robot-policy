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
  printf '找不到 FFmpeg；纯仿真双摄视频需要它。\n' >&2
  exit 2
fi
OUT="$(mktemp -d "$ROOT/artifacts/sim-v1-demo-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")"
export PYTHONDONTWRITEBYTECODE=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
"$PYTHON_BIN" -B scripts/s1_fixed_grasp.py --scene "$ROOT/artifacts/sim-v1-scene-20260925/scene.xml" --initial-wrist-roll-deg 0 --output "$OUT/run" 2>&1 | tee "$OUT/run.log"
"$PYTHON_BIN" -B scripts/build_sim_v1_page.py "$OUT"
"$PYTHON_BIN" -B scripts/audit_sim_v1_page.py "$OUT"
printf '\n演示完成。用浏览器打开：\n%s/index.html\n' "$OUT"
