# Reproducible setup and resource boundaries

## Lightweight entry point

Python 3.12 can plan evaluations and generate reports using only the standard library. From the repository root:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p test_evaluation.py -v
python3 scripts/evaluate.py --config configs/eval-mujoco.json \
  --python /path/to/model-environment/bin/python \
  --output artifacts/new-evaluation --dry-run
```

Dry-run checks configuration and prints commands; it does not prove that environments, weights or graphics drivers are installed. Remove `--dry-run` to execute in prepared environments. The CLI first performs preflight checks; failures print diagnostics without creating a run directory. Once checks pass, execution retains `preflight.json`, `plan.json`, `results.json`, `report.md`, logs and native evidence, including runtime model-loading failures. Always use a new output directory.

## Resources

| Resource | Location / configuration |
| --- | --- |
| ALOHA ACT / DOT checkpoints | `models/aloha-3d/LeTau__act_aloha_transfer_cube`, `models/aloha-3d/IliaLarchenko__dot_transfer_cube`; override each `checkpoint` |
| Move-pot ACT / SmolVLA checkpoints | `models/move-pot/act`, `models/move-pot/smolvla`; override each `checkpoint` |
| Model environment | `--python`, config `python`, or `LEROBOT_PYTHON` |
| Simulation environment | config `sim_python`; defaults `.venv-aloha/bin/python` or `.venv-sim/bin/python` |
| MuJoCo scene | Bundled `assets/so101/tape-base-offset.xml` with relative mesh references |
| Nominal simulation joint mapping | Bundled `configs/simulation/nominal-joint-map.json`; simulation reference only, never a physical device calibration |

JSON paths resolve relative to their configuration file. CLI paths resolve relative to the working directory. Checkpoint folders must contain the original model configuration, processor files, normalization state and weights expected by their adapter; copying only `model.safetensors` is insufficient. Weights, environments and raw historical runs are not bundled or automatically downloaded. Preserve the checkpoint identities in frozen study protocols when reproducing those studies.

## Tested environment reference

The local model environment used Python 3.12.3, LeRobot 0.6.1 (checkout `7e241bd630a3719a56157a497ce5d08f244784f1`), PyTorch 2.11.0+cu128, NumPy 2.2.6, Transformers 5.5.4 and Gymnasium 1.3.0. These are observed versions, not a verified clean-install lockfile. ALOHA additionally requires compatible `gym_aloha` and the retained DOT implementation under `third_party/lerobot-dot-legacy`. Rendering needs EGL; video processing needs FFmpeg/ffprobe. CUDA runs need a compatible GPU driver.

MuJoCo uses a separate environment with `requirements-sim.txt` (MuJoCo 3.3.7 and NumPy 2.5.3). Installation and full test commands are in [ARCHITECTURE.md](ARCHITECTURE.md). Offline adapter tests need NumPy, PyTorch and OpenCV; SmolVLA profile metadata is bundled under `tests/fixtures/smolvla-profile`, with no model weights required.

Hardware profiles retain site-specific resources and limits. They are not portable device setup templates; moving them to another machine or robot requires reviewing that integration. This change does not alter physical execution profiles.

## Evidence portability

Keep an entire output directory together: report links to logs and native summaries are relative. The saved plan and JSON provenance preserve original resolved paths and hashes as execution evidence. For formal comparisons, retain frozen protocol/configuration files and source/checkpoint manifests and use `scripts/analyze_model_comparison.py`. A clean-machine end-to-end installation has not been verified.

## Lightweight development checks

Install `requirements-core.txt`, then run `python3 scripts/test_core.py` and `python3 examples/adapter_template/run.py`. The GitHub workflow uses these commands on Python 3.12 and 3.13, independently of local model/robot resources.
