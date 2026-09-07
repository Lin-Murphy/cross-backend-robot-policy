# Third-party notices

The root [MIT license](LICENSE) applies to this project's original code and documentation, except where a file states otherwise. It does not relicense third-party material.

## SO101 mesh assets

The SO101 robot mesh assets and source description under `assets/so101/upstream/` are from **The Robot Studio SO101 Description (MJCF)**, developed by I2RT Robotics and derived from The Robot Studio's SO101 description. They remain under **Apache License 2.0**. Upstream attribution is in the [upstream README](assets/so101/upstream/README.md), and the full license is in [upstream LICENSE](assets/so101/upstream/LICENSE). Project scene XML files use these assets with task-specific modifications.

## DOT compatibility code

The vendored DOT compatibility code in `third_party/lerobot-dot-legacy/lerobot/` comes from [IliaLarchenko/lerobot](https://github.com/IliaLarchenko/lerobot/tree/42cca283322bbb68b6d3f1b4a436ada5b5d3393a) and remains under Apache License 2.0. See its [full license](third_party/lerobot-dot-legacy/LICENSE). Upstream attribution and license text are preserved.

## Models and historical results

Model weights are not distributed in Git. The ALOHA launcher expects locally supplied checkpoints from `LeTau/act_aloha_transfer_cube` and `IliaLarchenko/dot_transfer_cube`; their original repositories govern model terms and provenance. SO101 inference resources are also local and excluded from Git.

Historical raw trial traces and media were intentionally removed during the 2026-09-28 cleanup. The retained [compact results](evidence/results.json) contain historical summaries and source hashes, not model weights or the deleted media. Earlier statements that historical PushT/ACT/SmolVLA raw traces remain available no longer apply.
