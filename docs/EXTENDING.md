# Model and backend extension interfaces

The models and robots in this repository are examples of the interfaces. A new model requires an adapter; arbitrary model compatibility is not an acceptance criterion.

## Shared execution lifecycle

`ExecutionSession` in `src/cross_backend/execution_session.py` owns each observation → action → receipt transition. It validates observations before policy invocation, checks action identity/order/units, dispatches once, records events and stops on rejection or exception. Normal context-manager exit stops the backend. A closed session cannot execute again; stop is attempted once, and a failed stop is reported without replacing the original inference/dispatch exception.

```python
from cross_backend.execution_session import ExecutionSession

with ExecutionSession(backend, record=events.append) as session:
    for _ in range(control_budget):
        observation, request, receipt, stop = session.step(policy_action)
        if not receipt.accepted:
            break
        # Task evaluation decides when the episode is complete.
```

`policy_action(observation)` returns `ActionRequest`. It may call model inference or select an action from its own chunk queue. Model loading, input normalization, joint conversion and queue reset belong in that adapter. `action_request_from_policy` provides checked named-joint reordering; it deliberately does not silently convert units.

A scheduler that already captured an observation uses `session.step(policy_action, observation=frame)`. This performs no extra device read. Chunk prediction source/age stays in scheduler evidence; the execution request identifies the observation at the dispatch boundary, not necessarily the older observation used to predict a queued action.

## Backend interface

Implement `ExecutionBackend`: declare `BackendCapabilities`, return `BackendObservation` from `observe`, accept `ActionRequest` in `dispatch`, and return `ActionReceipt` / `StopReceipt`. Declare joint names, units, cameras, clock domain, reset and feedback capabilities explicitly. Unknown acquisition timestamps remain `None`. A physical transport return does not imply that a motor reached its target.

Reset/setup, physical limits, scheduling and task success criteria remain backend/task responsibilities. The shared lifecycle supplies no robot limits or implicit hardware reset. Task success/failure is evaluated separately from execution completion.

## Example integrations

| Example | Integration with shared lifecycle | Adapter-specific behavior |
| --- | --- | --- |
| SO101 ACT / SmolVLA | Existing guarded observation/send hook uses a persistent session; policy-loop exit calls its stop | Native LeRobot policy scheduling, audited bus guard, existing current-position hold and full-cycle return |
| MuJoCo ACT / SmolVLA | Every evaluation control tick dispatches through the session and `SimJointBackendAdapter` | Chunk scheduling, nominal coordinate conversion, physics clock and task observer; settling occurs before evaluation |
| ALOHA ACT / DOT | Native evaluators launched through `aloha_execution_launcher.py`; each Gym action uses the session | Original model processors, Gym reset, native rewards/termination and score serialization |

All examples use the same per-action lifecycle. Native outer episode loops remain adapters because their timing and task semantics differ. SO101 setup/return movements are retained audited device operations, outside the policy-action session.

Use the unified entry point to obtain this integration. Directly calling a third-party evaluator bypasses the repository launcher. Tests use fake devices for hardware; integration checks execute simulations only.

## Copyable starter and contract tests

Start with [the two-joint adapter template](../examples/adapter_template/README.md). It runs entirely in memory, deliberately maps a different model joint order into backend order, and records the same execution events as the integrations. Replace the model calculation and backend I/O while keeping the shared lifecycle unchanged.

```bash
python3 -m pip install -r requirements-core.txt
python3 examples/adapter_template/run.py
python3 scripts/test_core.py
```

The portable core suite covers contracts, stopped sessions, fake hardware hooks, evaluation aggregation, preflight diagnostics and the template. GitHub Actions runs this suite and the example on Python 3.12 and 3.13 for pushes and pull requests. It does not download models or access devices. Full model and physics checks remain separate from this lightweight CI job.
