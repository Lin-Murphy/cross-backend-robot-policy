# Architecture and development guide

## Responsibilities

| Responsibility | Implementation |
| --- | --- |
| Shared per-action lifecycle | [execution_session.py](../src/cross_backend/execution_session.py); [extension guide](EXTENDING.md) |
| Unified configuration, launch and result aggregation | [evaluation.py](../src/cross_backend/evaluation.py), [evaluate.py](../scripts/evaluate.py); [usage](EVALUATION.md) |
| Model-specific observations, processors, actions, and reset | [move_pot_policy.py](../src/cross_backend/move_pot_policy.py), [smolvla_move_pot.py](../src/cross_backend/smolvla_move_pot.py) |
| Backend capabilities, observation identity, action mapping, dispatch and stop receipts | [execution_contract.py](../src/cross_backend/execution_contract.py) |
| Action queues and simulation timing | [sim_chunk_schedule.py](../src/cross_backend/sim_chunk_schedule.py), [chunk_policy.py](../src/cross_backend/chunk_policy.py) |
| MuJoCo execution | [offline_backend_adapters.py](../src/cross_backend/offline_backend_adapters.py), [tape_sim_backend.py](../src/cross_backend/tape_sim_backend.py) |
| SO101 shared execution boundary and audited guard | [so101_lerobot_backend_adapter.py](../src/cross_backend/so101_lerobot_backend_adapter.py), [legacy_so101_sync_guard.py](../src/cross_backend/legacy_so101_sync_guard.py), [legacy_bus_audit.py](../src/cross_backend/legacy_bus_audit.py) |
| Task feedback and success/failure rules | [tape_task.py](../src/cross_backend/tape_task.py), [tape_task_observer.py](../src/cross_backend/tape_task_observer.py) |
| Run results and completion milestones | [run_record.py](../src/cross_backend/run_record.py), [completion_record.py](../src/cross_backend/completion_record.py) |
| Model comparisons and paired study analysis | [compare_aloha_3d_eval.py](../scripts/compare_aloha_3d_eval.py), [trial_analysis.py](../src/cross_backend/trial_analysis.py), [sim_study_analysis.py](../src/cross_backend/sim_study_analysis.py) |

Policy adaptation belongs inside the execution workflow. It does not make model processors interchangeable. Named joint mapping requires matching units; conversion between simulator radians and SO101 degrees/calibrated gripper values is explicit.

Action receipts distinguish accepted requests from rejected candidates and physical dispatches. SO101 `sync_transport_return` means the transport call returned, not that every motor acknowledged or reached the target. Verified camera exposure and state-acquisition timestamps are unavailable in the current hardware adapter and remain unknown.

Task records distinguish placement from a full cycle (withdrawal, return to start, normal termination). Unknown outcomes stay unknown. Simulation truth is used by the evaluator, not as privileged model input. Common milestone records enable consistent interpretation but do not establish paired simulation/hardware performance.

## Retained scripts

| Scripts | Purpose and prerequisites |
| --- | --- |
| `evaluate.py` | Common evaluation entry point for ALOHA, MuJoCo and single approved SO101 runs; see [configuration and results](EVALUATION.md). |
| `run_aloha_3d_comparison.sh`, `compare_aloha_3d_eval.py` | Run ACT/DOT and apply the same reward-based success criterion. Require prepared environments and weights. |
| `make_aloha_side_by_side.py` | Create labeled paired videos from recorded model runs. Requires FFmpeg/ffprobe. |
| `aloha_observation_delay_launcher.py`, `analyze_aloha_observation_age.py` | Apply and analyze declared observation-delay conditions for new ALOHA experiments; not a claim of a completed delay study. |
| `run_s1_policy_development.py`, `s1_policy_worker.py` | Shared ACT/SmolVLA MuJoCo development loop and model worker. Uses retained local checkpoints/calibration and a local LeRobot interpreter; nominal mapping is not physically validated. |
| `run_so101_legacy30_audited_pilot.py` | Current SO101 development runner. `--shared-boundary` selects the shared hook; ACT requires it. Physical execution needs a new approved plan. |
| `capture_move_pot_readonly.py` | No-write device capture and device/calibration identities used by the hardware runner. Opening a device is separate from offline tests. |
| `audit_so101_shared_live_boundary.py`, `evaluate_so101_legacy30_pilot_record.py` | Audit ordered observations/actions/packets and build result records from newly supplied run evidence. |
| `calibrate_s1_camera_offline.py`, `capture_s1_checkerboard_readonly.py` | Camera calibration fitting and capture. Offline fitting uses supplied images; capture accesses cameras. |
| `move_pot_trial_records.py`, `analyze_move_pot_trials.py`, `analyze_sim_v_stage.py` | Initialize/validate planned trial records and analyze supplied evidence. Empty trial slots are not results. |
| `compare_completion_records.py` | Compare compatible task milestones across backends without asserting paired performance. |
| `test_command_gate.py` | Offline checks of software action limits and rejection behavior. |

The `s1` and `legacy30` names remain where they identify an implemented integration and its tests. Historical one-shot scene adjustments, gripper microtrials, old bounded/feedback-wait runners, training launchers, release builders, and fixed-trajectory demonstrations are no longer maintained here.

## Running and testing

### ALOHA model comparison

Use a Python environment containing the current LeRobot ACT evaluator and PyTorch. The launcher also needs a compatible `gym_aloha` environment, the vendored DOT code, FFmpeg/ffprobe, and EGL rendering.

| Environment variable | Meaning / default |
| --- | --- |
| `LEROBOT_PYTHON` | Prepared model interpreter; defaults to `python3` |
| `LEROBOT_EVAL` | ACT evaluator; defaults to `lerobot-eval` beside that interpreter |
| `ALOHA_SIM_PYTHON` | ALOHA environment; defaults to `.venv-aloha/bin/python` |
| `ALOHA_MODELS_DIR` | Defaults to `models/aloha-3d`, containing `LeTau__act_aloha_transfer_cube` and `IliaLarchenko__dot_transfer_cube` |
| `ALOHA_DEVICE` | Optional inference device override |
| `ALOHA_DELAY_STEPS` | Optional observation delay for a separately designed experiment |

```bash
bash scripts/run_aloha_3d_comparison.sh artifacts/aloha-new 30 1000
python3 scripts/make_aloha_side_by_side.py \
  --results artifacts/aloha-new --seed 1000 --output artifacts/aloha-new/paired.mp4
```

The launcher enables offline model loading; it does not install dependencies or download weights. Defaults without episode/seed arguments are 10 episodes from seed 11. Fresh outputs should use a new directory. ALOHA evaluators remain separate implementations; the comparison reconciles their native success fields using maximum reward >= 4.

### MuJoCo backend tests

Python 3.12 and [requirements-sim.txt](../requirements-sim.txt) provide the lightweight simulation environment. EGL must work for rendering.

```bash
python3.12 -m venv .venv-sim
.venv-sim/bin/python -m pip install -r requirements-sim.txt
PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl PYTHONPATH=src:scripts:tests:. \
  .venv-sim/bin/python -B -m unittest \
  test_cube_task_observer test_tape_state_clock test_sim_trial_initial -v
```

Scenes live in `assets/so101/` and use relative mesh paths. They are test/development inputs, not fixed-trajectory demonstrations.

### Remaining offline tests

Use an environment with NumPy, PyTorch, and OpenCV. SmolVLA profile tests use bundled metadata fixtures and do not require downloaded weights. See [reproducible setup](REPRODUCIBILITY.md).

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:scripts:tests:. python -B - <<'PYTEST'
from pathlib import Path
import unittest
simulation = {"test_cube_task_observer", "test_tape_state_clock", "test_sim_trial_initial"}
modules = [p.stem for p in sorted(Path("tests").glob("test_*.py"))
           if p.stem not in simulation]
result = unittest.TextTestRunner(verbosity=2).run(
    unittest.defaultTestLoader.loadTestsFromNames(modules))
raise SystemExit(not result.wasSuccessful())
PYTEST
```

Tests exercise offline fixtures and simulation, not new hardware rollouts. The repository retains local reference profiles without treating them as portable physical limits.

## Next engineering work

The unified evaluation entry point now covers the supported native runners. Automatic reports and portable simulation examples are included. Hardware profiles remain site-specific; see [support boundaries](SUPPORT.md). New policy, task, or backend integrations must declare their input/output and feedback capabilities. Formal comparisons need fixed checkpoint identities, paired initial conditions, declared timing, task metrics, and retained raw evidence. The README summarizes current results without a separate versioned results document.
