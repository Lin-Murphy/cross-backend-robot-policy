# Unified evaluation

`scripts/evaluate.py` is the common entry point for configuration, model selection,
launching, evidence retention and result reporting. Backend adapters retain native
model processors and episode scheduling while each policy action passes through the common execution session. This is a unified orchestration layer, not
a claim that ACT, DOT and SmolVLA share identical inference implementations.

## Run

Use an existing model environment with the required LeRobot version and local
weights. No dependencies or checkpoints are downloaded by this entry point.

```bash
python3 scripts/evaluate.py --config configs/eval-aloha.json \
  --python /path/to/lerobot/.venv/bin/python \
  --output artifacts/aloha-evaluation-001
```

`--python` selects the model interpreter. ACT's `lerobot-eval` defaults to the
executable beside it; `act_evaluator` or `LEROBOT_EVAL` can override that path.
The model interpreter can also be set in JSON or through `LEROBOT_PYTHON`.
ALOHA uses `.venv-aloha/bin/python` to locate simulation dependencies; MuJoCo
uses `.venv-sim/bin/python` to run its physics loop. Set `sim_python` in the
configuration to override either. Paths in JSON are relative to the JSON file;
command-line paths are relative to the current working directory. Executable
names without a slash are resolved through PATH. The launch working directory
is always the repository root.

Inspect the resolved commands without launching models, opening devices or
creating an output directory:

```bash
python3 scripts/evaluate.py --config configs/eval-aloha.json \
  --python /path/to/lerobot/.venv/bin/python --output artifacts/preview --dry-run
```

The output directory must be new. Existing evidence is never overwritten.

## Supported configurations

| Backend / task | Policies | Example | Behavior |
| --- | --- | --- | --- |
| `aloha` / `transfer_cube` | `act`, `dot` | `configs/eval-aloha.json` | Select either or both; 400 steps per episode; common maximum-reward >= 4 scoring |
| `mujoco` / `move_pot` | `act`, `smolvla` | `configs/eval-mujoco.json` | Select either or both; existing nominal-map development loop and task evaluator |
| `so101` / `move_pot` | `act`, `smolvla` | `configs/eval-so101.json` | One policy, one episode, existing audited profile and shared execution boundary |

Every configuration has `schema_version: 1`, `backend`, `task`, and a nonempty
`policies` list. A simulation policy has `name` and a local `checkpoint` directory.
SO101 policy checkpoints remain defined by the existing audited profiles.
Unknown fields and unsupported combinations are rejected before launch.

`episodes` defaults to 1 and `first_seed` to 1000. In ALOHA, each policy receives
the same requested environment seed sequence. The legacy DOT output is checked
against that sequence; ACT's native output does not contain per-episode seeds,
so the saved command records its requested sequence. This does not establish
identical evaluator preprocessing.

In MuJoCo, seeds control policy sampling. They do **not** randomize the scene.
Optional `initial_conditions` is a list of C-development initial-state files,
one per episode, reused across selected policies. Without it, episodes use the
scene's default start. This remains nominal simulation without verified physical
alignment. `scene`, `calibration`, `device`, and `observation_delay_frames` (0 or 1)
are explicit inputs. Edit the checkpoint paths in `eval-mujoco.json` to point at
your local move-pot weights; its example paths are not bundled checkpoints.

`device` defaults to `cpu` for simulation; choose `cuda` in JSON when available.

## SO101

Offline configuration validation uses the same entry point:

```bash
python3 scripts/evaluate.py --config configs/eval-so101.json \
  --python /path/to/lerobot/.venv/bin/python \
  --output artifacts/so101-validation-001 --validate-only
```

This invokes the existing no-hardware validation path, not a robot rollout.
Physical execution requires the existing on-site approval for the specific run
and the explicit `--execute-approved-once` flag instead. The entry point does
not batch hardware episodes, select new physical limits, or infer operator
approval from the JSON file. A plain SO101 invocation refuses to start.

Hardware outcomes remain unknown until supported by operator evidence. The
existing `evaluate_so101_legacy30_pilot_record.py` can subsequently build the
richer audited RunRecord from the retained raw directory and operator evidence.

## Results

Each invocation retains:

- `plan.json`: resolved configuration, commands, environment overrides and seeds.
- `logs/<job>.log`: captured native runner output. ALOHA also writes `<job>-execution.jsonl`; MuJoCo writes `execution.jsonl` within each policy raw directory; SO101 retains shared events in its audited event stream.
- `raw/<job>/`: native summaries, action records and any generated videos.
- `results.json`: common execution status, episode outcomes and per-policy totals.
- `report.md`: automatically updated model totals, missing/unknown counts, execution failures, episode reasons and links to raw evidence.

`results.json` is updated after each job and at completion. One failed simulation
job does not discard another policy's results. Checkpoint weight hashes, scene,
calibration and initial-file hashes are recorded when those inputs are available;
normalized episode records reference their native evidence file and its hash.
The file uses a separate evaluation schema from the detailed hardware RunRecord:
missing action counts and physical feedback are not invented.

Execution and task outcomes are separate:

| Field | Meaning |
| --- | --- |
| `execution_status=completed` | The runner produced an evaluable result; a failed task or timeout is allowed |
| `execution_status=stopped` | A control gate or recorded hardware interruption ended execution |
| `execution_status=error` | Loading, configuration, process or evidence validation failed |
| `execution_status=interrupted` | The user interrupted evaluation |
| Episode `outcome=true/false/null` | Task succeeded / task failed / outcome unknown |

The command exits 0 for completed evaluation even if every task failed, 1 for an
execution error or stop, 2 for invalid configuration, and 130 for interruption.
`success_rate_known` uses only known outcomes. Always read it alongside
`expected_episodes`, `missing_episodes`, and `unknown_outcomes`; missing or unknown
evidence is not counted as a failed task. Totals summarize the selected run and
are not automatically a formal experiment or a paired sim-to-real study.

## Tests

The common lifecycle tests need only Python's standard library:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p test_evaluation.py -v
```

Native backend tests still use the environments documented in ARCHITECTURE.md.

## Model comparison and scope

[Completed model comparisons](STUDY_RESULTS.md) use frozen protocols and native-evidence reconstruction. The automatic report is a descriptive summary; `analyze_model_comparison.py` adds verified two-model statistical analysis for a frozen protocol. Simulation/hardware pairing is deferred optional research, not required for either workflow.

See [supported scope](SUPPORT.md) and [setup and resources](REPRODUCIBILITY.md). The MuJoCo example uses a bundled nominal joint mapping for simulation only; it is not a calibration for any physical robot.
