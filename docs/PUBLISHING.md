# What belongs on GitHub

The repository presents the reusable interfaces and representative examples. It does not need to contain every experiment artifact.

Keep source, tests, example configurations, interface/setup documentation, a representative demo, and compact results describing model identity, conditions, sample counts and limitations. The README links to these materials. An evaluation automatically produces a readable report, so a selected report can accompany a demo or release.

Keep model weights, environments, raw videos, per-action logs and intermediate runs under ignored local directories (`models/`, `artifacts/`). If a complete experiment needs to be independently audited, publish that specific evidence bundle separately as release assets or a dataset and link it from the result summary. No upload is required for ordinary project delivery.

Claims must match the available evidence. A hash establishes identity when a file is available; it does not substitute for distributing that file. Compact historical findings are explicitly identified where raw evidence is unavailable. Simulation and real-world results are presented separately. Example-model success rates demonstrate particular deployments, not universal interface performance.
