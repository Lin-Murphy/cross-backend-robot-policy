#!/usr/bin/env bash
# Offline, simulation-only ACT vs DOT ALOHA TransferCube comparison.
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${LEROBOT_PYTHON:-$(command -v python3)}"
eval_bin="${LEROBOT_EVAL:-$(dirname "$python_bin")/lerobot-eval}"
sim_python="${ALOHA_SIM_PYTHON:-$repo_dir/.venv-aloha/bin/python}"
run_dir="${1:-artifacts/aloha-3d-comparison-$(date -u +%Y%m%dT%H%M%SZ)}"
episodes="${2:-10}"
first_seed="${3:-11}"
mkdir -p "$run_dir"

export MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled
export PYTHONDONTWRITEBYTECODE=1
models_dir="${ALOHA_MODELS_DIR:-$repo_dir/models/aloha-3d}"
act_weights="$models_dir/LeTau__act_aloha_transfer_cube"
dot_weights="$models_dir/IliaLarchenko__dot_transfer_cube"
test -s "$act_weights/model.safetensors"
test -s "$dot_weights/model.safetensors"
test -x "$eval_bin"
test -x "$sim_python"
sim_site="$("$sim_python" -c 'import site; print(site.getsitepackages()[0])')"
device="${ALOHA_DEVICE:-$("$python_bin" -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')}"

act_args=(
  --policy.path="$act_weights" --policy.device="$device" --env.type=aloha
  --env.task=AlohaTransferCube-v0 --env.episode_length=400
  --eval.n_episodes="$episodes" --eval.batch_size=1 --eval.use_async_envs=false
  --output_dir="$run_dir/act" --seed="$first_seed"
)
dot_args=(
  --policy.path="$dot_weights" --env.type=aloha --env.task=AlohaTransferCube-v0
  --env.episode_length=400 --eval.n_episodes="$episodes" --eval.batch_size=1
  --eval.use_async_envs=false --output_dir="$run_dir/dot" --seed="$first_seed"
  --device="$device" --use_amp=false
)

if [[ -n "${ALOHA_DELAY_STEPS+x}" ]]; then
  PYTHONPATH="$sim_site" "$python_bin" -B scripts/aloha_observation_delay_launcher.py \
    --delay-steps="$ALOHA_DELAY_STEPS" --trace-file="$run_dir/act.observation-age.jsonl" \
    -- "$eval_bin" "${act_args[@]}" > "$run_dir/act.log" 2>&1
  PYTHONPATH="$repo_dir/third_party/lerobot-dot-legacy:$sim_site" "$python_bin" -B \
    scripts/aloha_observation_delay_launcher.py \
    --delay-steps="$ALOHA_DELAY_STEPS" --trace-file="$run_dir/dot.observation-age.jsonl" \
    -- third_party/lerobot-dot-legacy/lerobot/scripts/eval.py "${dot_args[@]}" \
    > "$run_dir/dot.log" 2>&1
else
  PYTHONPATH="$sim_site" "$eval_bin" "${act_args[@]}" > "$run_dir/act.log" 2>&1
  PYTHONPATH="$repo_dir/third_party/lerobot-dot-legacy:$sim_site" "$python_bin" -B \
    third_party/lerobot-dot-legacy/lerobot/scripts/eval.py "${dot_args[@]}" \
    > "$run_dir/dot.log" 2>&1
fi

"$python_bin" -B scripts/compare_aloha_3d_eval.py \
  --act "$run_dir/act" --dot "$run_dir/dot" \
  --output "$run_dir/comparison.json" --first-seed "$first_seed"
echo "Results: $run_dir/comparison.json"
