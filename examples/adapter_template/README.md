# Minimal policy/backend integration

From the repository root, with Python 3.12 and NumPy installed:

```bash
python3 examples/adapter_template/run.py
```

This produces three observation/request/receipt transitions and a stop event. It is a deterministic interface example, not a learned policy or task-performance demonstration. No weights, device access or graphics stack are needed.

Copy `adapters.py` as a starting point. Replace `ExamplePolicy.__call__` with model preprocessing, inference and an explicit output mapping. Replace `MemoryBackend.observe/dispatch/stop` with backend I/O and declare its capabilities. This example intentionally uses two joints with different units and a different model joint order; the shared execution core requires no changes.

Reset your model's queues before a new episode. Backend setup/reset and task termination are application responsibilities. Do not reuse this fake backend's behavior as a physical stop implementation.

Run the contract checks with:

```bash
PYTHONPATH=src:tests:. python3 -m unittest test_adapter_template -v
```

This example plugs directly into `ExecutionSession`. Registering a new native runner with `scripts/evaluate.py` also requires declaring its supported configuration, launch plan and result normalization; that application layer is separate from the execution interface.
