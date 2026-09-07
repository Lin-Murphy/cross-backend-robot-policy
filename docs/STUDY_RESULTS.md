# Model comparisons — 2026-09-28

Two fresh model comparisons completed through the unified evaluation entry point.
A task failure is a valid observed result; it does not imply execution failure.
The prospective simulation/hardware pairing study is not yet complete.

## ALOHA TransferCube

Thirty new requested environment seeds (2000–2029), the same per model, were
frozen before execution. Both models used CUDA, 400-step episodes, and the common
criterion `episode maximum reward >= 4`.

| Model | Successes | Success rate | Wilson 95% interval |
| --- | --- | --- | --- |
| ACT | 14/30 | 46.7% | 30.2–63.9% |
| DOT | 27/30 | 90.0% | 74.4–96.5% |

Across paired outcomes, DOT alone succeeded on 14 cases, ACT alone on 1, both on
13, and neither on 2. DOT minus ACT was **43.3 percentage points**, with a paired
bootstrap 95% interval of **23.3–63.3 points** (10,000 resamples, fixed seed).
The exact two-sided McNemar p-value is 0.0009765625. These are finite-sample
statistics for the two concrete deployments: evaluator versions and
preprocessing differ, so the difference cannot be attributed solely to model
architecture. ACT's native scores omit episode seeds; its saved launch command
provides the requested sequence, while DOT's reported seeds were checked.

These are new results, separate from the historical seeds 1000–1029 (11/30 versus
26/30) and the two-episode README demonstration.

## SO101 scene in MuJoCo

Ten distinct C-development initial geometries were frozen before outcomes were
observed: five object x positions from -0.190 to -0.170 m, crossed with two y
positions (-0.155 and -0.145 m). Both policies used identical initial files,
zero robot joint starts, the same target mat, 30 Hz control, 600 control ticks,
measured inference latency, and CUDA. Seeds 2000–2009 controlled policy sampling,
not random initial geometry.

| Model | Executed / planned | Successes | Recorded task result |
| --- | --- | --- | --- |
| ACT | 10/10 | 0/10 | 10 timeouts |
| SmolVLA | 10/10 | 0/10 | 10 timeouts |

All 20 executions completed and retained their raw evidence. This is a valid
comparison with no observed successes. It does not establish equivalent model
ability: the 95% Wilson interval for each observed 0/10 is 0–27.8%, the sample is
small, and the scene uses a nominal, physically unverified coordinate/camera
mapping. No simulation-to-hardware predictive conclusion follows from these
results. Calibration cases must not later be relabeled as unseen evaluation
cases.

## CPU records and runtime effects

An earlier CPU pass is retained separately. ACT completed all ten move-pot
cases (timeouts); SmolVLA recorded ten observation-age gate stops before task
outcomes could be determined. Those are **unknown task outcomes**, not ten task
failures. ALOHA ACT also completed 30 CPU episodes (14 successes), while the
much slower DOT CPU evaluation was interrupted for runtime reasons.

The subsequent CUDA protocols retained the same models, initial cases, seeds
and metrics. CPU and CUDA results are not pooled. Host load was not isolated
for a dedicated inference-speed benchmark; timing measurements describe these
deployments and are not causal architecture comparisons.

## Simulation/hardware pairing

The new unpaired ACT hardware reference recorded 533 policy dispatches,
verified return to start (maximum absolute error 6 raw ticks), and an
operator-confirmed failed placement. This is valid real-world failure evidence.
It is not one of the prospective paired cases:

- The first real policy state was approximately `[-8.09, -55.56, 73.32, 36.62,
  -24.13, 8.63]` in degrees / calibrated gripper units, while the simulation
  used the declared nominal zero-joint starts.
- The physical cameras, table frame and joint/gripper mapping have not yet been
  measured and reviewed against the simulation scene.
- The real policy budget was 18 s; the completed simulation study used 20 s.
- A corresponding SmolVLA real episode and frozen physical case identity are
  absent.

The prospective baseline pilot has **20 planned pairs** (10 measured cases ×
2 models) and currently **0 eligible completed pairs**. It is separate from the
older 40-slot baseline/added-age study. The pairing analyzer checks measurements,
source hashes, model identities, initial-state binding, timing and raw audit
references; it accepts failed tasks and excludes unknown or unmatched cases.
See [pairing workflow](SIM_REAL_STUDY.md).

## Evidence and reproducibility

[Compact results](../evidence/model-comparison-20260928.json) retain per-pair
outcomes, uncertainty, checkpoint hashes and protocol/source identities.
Local raw studies are under:

- `artifacts/studies/model-comparison-cuda-20260928/`: completed CUDA comparisons,
  frozen protocols, full native scores, model inputs/outputs, logs and reports.
- `artifacts/studies/model-comparison-20260928/`: CPU records and frozen C initial
  conditions shared with the CUDA run.
- `artifacts/studies/sim-real-pairing-20260928/`: prospective records, readiness,
  unpaired hardware evidence index and calibration acquisition.

Native ALOHA evaluators emit at most ten videos per model; all emitted videos
and all 30 episode scores are retained. MuJoCo retains each episode's video,
action events, predictions, inputs, schedule and initial/final state. Execution
source snapshots are archived by SHA256. These raw outputs are ignored by Git;
a source-only checkout includes compact findings, not the full experiment data.

Recompute either report using `scripts/analyze_model_comparison.py --run ...
--protocol ... --output <new-directory>`. The analyzer reconstructs outcomes
from native evidence and checks them against frozen identities before counting.
