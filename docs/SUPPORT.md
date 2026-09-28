# Supported project scope

The project delivers reusable policy deployment, backend execution, task evaluation and retained evidence. Evaluation completion does not require task success. Simulation and hardware are independent evaluation tracks.

| Backend / task | Models | Supported workflow | Boundary |
| --- | --- | --- | --- |
| ALOHA / transfer_cube | ACT, DOT | Unified launch, native scoring, automatic comparison report | Same requested environment seeds and common reward criterion; native processors/evaluators differ |
| MuJoCo / move_pot | ACT, SmolVLA | Unified launch, nominal scene, task outcomes, automatic report | Same declared scene, initial states and timing; seeds affect policy sampling only |
| SO101 / move_pot | ACT, SmolVLA | Offline profile validation; individually approved physical runs | Site-specific audited profiles; operator evidence needed for task outcomes; no batch hardware comparison |

Unsupported model/task/backend combinations are rejected. Adding a model requires an adapter and declared observation/action contracts, not merely a checkpoint path.

Current acceptance criteria: the selected runner executes or reports an actionable error; raw evidence is retained; task outcomes remain distinct from execution status; summaries expose missing and unknown outcomes. A valid timeout is a completed evaluation with a failed task.

Model comparisons belong within one backend/task and declared conditions. The completed studies are documented in [STUDY_RESULTS.md](STUDY_RESULTS.md). Descriptive automatic reports do not establish statistical superiority or architecture-only effects.

Simulation/hardware pairing, measured scene reconstruction and transfer performance are deferred optional research. Their existing tools and records remain available for reference; they do not block delivery. Physical models and the simulated scene are not claimed to match.
