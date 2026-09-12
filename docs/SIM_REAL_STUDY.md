> **Deferred optional research.** This workflow is retained for reference. Camera/table calibration and physical matching are not prerequisites for the unified evaluator or within-backend model comparisons.

# Measured simulation/hardware pairing

The study compares the same policy checkpoint on corresponding measured initial
conditions and a shared task rule. Execution success, task success and complete
return-to-start are separate quantities. A failure/failure pair is valid data;
an unknown outcome or unmatched start is not a valid binary outcome pair.

## Current field work

The operator has placed the existing checkerboard on the desk. Initial read-only
640×480 captures found the overhead board cropped and no board in the wrist
camera. No intrinsic/extrinsic calibration was inferred from those rejected
frames. A printed square's actual side length is required, because print scaling
changes metric scale. Both cameras need multiple distinct board positions and
tilts; a single flat, stationary image does not determine reliable intrinsics.

The camera-only capture tool imports no robot or motor modules. Interactive mode
remains available. Automatic mode captures while the operator moves the board:

```bash
python scripts/capture_s1_checkerboard_readonly.py \
  --camera follower --device /dev/videoN --columns 9 --rows 6 \
  --automatic --duration-seconds 60 --interval-seconds 1 \
  --views 20 --min-views 15 --output artifacts/new-board-capture
```

Use the verified V4L2 device identity, board corner count and camera resolution.
The example 9×6 pattern is the existing tool default, not a measurement of a
new print. Keep the robot stationary; move the board, not the camera mounts.
The same process is needed for `camera2`, with the board in that camera's view.
The fitting tool `calibrate_s1_camera_offline.py` uses the measured square size,
checks capture hashes and distinct views, and reports candidate intrinsics with
holdout reprojection errors. Candidate fits do not automatically certify camera
extrinsics or the full scene.

## Freeze correspondence before task trials

1. Measure and review the joint zero/sign and gripper endpoints, camera
   intrinsics/extrinsics, table reference frame, object and mat dimensions.
2. Use calibration/development cases for these measurements. Freeze the resulting
   scene and mapping before selecting independent P cases for evaluation.
3. Record each P case's six-joint state, object/mat pose, two views, units and
   source hashes. Declare numeric pair tolerances before examining task outcomes.
4. Fix checkpoint/processor identities, task success criteria, timing and
   observation/queue behavior. The current prospective baseline draft targets
   30 Hz and an 18 s policy window to match the current hardware reference.
5. Prepare and review one concrete hardware execution plan at a time under the
   existing on-site workflow. No analyzer or study manifest authorizes motion.
6. Preserve raw evidence for success, failure, rejection and interruption.
   Bind the corresponding simulation and real initial observations, record
   measured residuals and review outcomes before computing paired statistics.

The new baseline pilot is 10 cases × ACT/SmolVLA = 20 backend pairs. The historical
40-slot baseline/added-observation-age plan remains separate and uncompleted;
this pilot does not silently replace or fulfill that delay study.

## Aligned simulation entry point

A MuJoCo unified configuration can additionally specify:

```json
{
  "max_control_ticks": 540,
  "alignment_spec": "measured-alignment.json",
  "initial_conditions": ["measured-initial-C01.json"]
}
```

These are additional fields in the normal `mujoco` / `move_pot` configuration,
not a complete configuration. `max_control_ticks` accepts 1–600. At 30 Hz, 540
steps is 18 s; ending before the evaluator's 20 s timeout yields `incomplete`
when no terminal task event occurred, a known unsuccessful task result.

Without `alignment_spec`, the existing nominal mapping remains explicitly
unverified. With it, `load_sim_alignment` requires:

- `schema_version: 1`, the scene-plus-mesh `scene_sha256`, `robot_calibration_sha256`, reviewer and review time.
- `verified_coordinate_mapping` in the existing `CoordinateMapping` schema:
  six-element scale/offset, policy/simulation bounds, provenance and
  `validation: physical_alignment_verified`.
- `joint_and_gripper_mapping`, `camera_intrinsics`, `camera_extrinsics`,
  `table_frame`, `object_and_mat_geometry`, each with
  `status: measured_and_reviewed` and nonempty hashed measurement references.

The runner uses the provided affine mapping in both observation and action
conversion. It checks referenced evidence and scene identity. The measurements
and review remain external empirical work; setting a flag without valid
measurements does not establish scientific alignment.

## Pairing records and analysis

Create a fresh draft with `prepare_sim_real_study.py --model-protocol <frozen
move-pot protocol> --output <new-directory>`. It creates unstarted records and
unset measurement fields; no fabricated outcomes or hardware execution.

`pairs.json` references a frozen protocol by path/hash and retains every planned
case/policy pair. Each completed pair requires:

| Section | Required references |
| --- | --- |
| `simulation` | `summary`, `initial`, `budget`, `checkpoint_manifest`, `events`, `video` |
| `real` | `summary`, `assessment`, `audit`, `profile`, `first_observation`, `events`, `checkpoint`, `follower_video`, `camera2_video` |
| `binding` | Reviewed case/policy identity, simulation initial hash, real first-observation hash, scene hash, shared success rule, measured residuals, measurement references and timing review |

Each reference is `{ "path": "...", "sha256": "..." }`. Relative paths resolve
from the pairing manifest directory. The assessment preserves `task_success`
as true/false/null with `task_evidence`; a return-to-start verdict must not
substitute for placement success. `binding.measured_residuals` contains
`joint_max_error_deg`, `gripper_error_units`, `object_position_error_m`,
`mat_position_error_m`, and `camera_reprojection_error_px`, checked against
predeclared `protocol.pair_tolerances`.

```bash
python3 scripts/analyze_sim_real.py --manifest <pairs.json> \
  --output <new-report.json> --require-complete
```

The report retains excluded pairs and their reasons, and computes agreement,
paired outcome difference and uncertainty only for eligible known outcomes.
`--require-complete` exits nonzero until every planned pair is eligible; that
means the study is incomplete, not that the individual failed tasks are invalid.
Agreement in a small sample, especially all-failure cases, cannot establish
predictive validity beyond those conditions.
