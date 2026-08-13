#!/usr/bin/env bash
# User-invoked formal ACT training: 40000 updates, maximum 7200 seconds.
set -euo pipefail
PROJECT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT"
RUN_DIR="$PROJECT/artifacts/r1-act-formal-40000-run02"
CONFIG_PATH="$PROJECT/configs/act-move-pot-r1-formal-40000.json"
PYTHON_BIN=/home/murphy/project/lerobot/.venv/bin/python
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=online
export PYTHONUNBUFFERED=1
# Refuse overwrite, and fail before launch if the approved cached initialization is absent/changed.
printf '%s  %s\n' f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec /home/murphy/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth | sha256sum --check --status
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
c=json.loads(Path('configs/act-move-pot-r1-formal-40000.json').read_text())
assert c['steps']==40000 and c['batch_size']==8 and c['seed']==1000
assert c['resume'] is False and c['policy']['pretrained_path'] is None
assert c['policy']['type']=='act' and c['policy']['push_to_hub'] is False
assert c['wandb']['enable'] and c['wandb']['mode']=='online'
assert c['wandb']['disable_artifact'] and not c['save_checkpoint_to_hub']
assert c['env'] is None and c['env_eval_freq']==0 and c['eval_steps']==0
assert c['output_dir']==str(Path.cwd()/'artifacts/r1-act-formal-40000-run02/train')
assert not Path(c['output_dir']).exists()
PY
mkdir "$RUN_DIR"
cp "$CONFIG_PATH" "$RUN_DIR/config.json"
cp "$0" "$RUN_DIR/launcher.sh"
git -C /home/murphy/project/lerobot rev-parse HEAD > "$RUN_DIR/lerobot-revision.txt"
nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,utilization.gpu --format=csv -l 1 > "$RUN_DIR/gpu.csv" &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true' EXIT
START_SECONDS=$(date +%s)
date --iso-8601=seconds > "$RUN_DIR/start.txt"
set +e
/usr/bin/time -v -o "$RUN_DIR/time.txt" timeout --signal=INT --kill-after=15s 7200s \
  "$PYTHON_BIN" -m lerobot.scripts.lerobot_train --config_path="$CONFIG_PATH" > "$RUN_DIR/train.log" 2>&1
RESULT=$?
set -e
END_SECONDS=$(date +%s)
date --iso-8601=seconds > "$RUN_DIR/end.txt"
printf '{"exit_code": %s, "wall_s": %s, "requested_steps": 40000}\n' "$RESULT" "$((END_SECONDS-START_SECONDS))" > "$RUN_DIR/process-result.json"
# Preserve partial runs too. Completion requires reading the actual final step/checkpoint.
find "$RUN_DIR/train" -type f -print0 2>/dev/null | sort -z | xargs -0 -r sha256sum > "$RUN_DIR/train-files.sha256" || true
exit "$RESULT"
