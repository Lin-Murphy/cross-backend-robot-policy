"""Read calibration registers only; requires explicit physical readiness flag."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import traceback


def compare(expected, actual):
    # drive_mode is a host-side field synthesized by read_calibration, not read from hardware.
    fields = ("id", "homing_offset", "range_min", "range_max")
    return [
        {"motor": name, "field": field, "expected": values[field],
         "actual": actual.get(name, {}).get(field)}
        for name, values in expected.items() for field in fields
        if values[field] != actual.get(name, {}).get(field)
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--physical-ready", action="store_true")
    args = parser.parse_args()
    if not args.physical_ready:
        parser.error("Confirm power-cut access, clear workspace and port identity before --physical-ready")
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    raw = args.calibration.read_bytes()
    expected = json.loads(raw)
    names = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    if set(expected) != set(names) or any(expected[n]["id"] != i for i, n in enumerate(names, 1)):
        raise ValueError("Expected six SO101 joints with IDs 1 through 6")

    class ReadOnlyBus(FeetechMotorsBus):
        def write(self, *a, **kw):
            raise RuntimeError("Motor writes forbidden in calibration check")

        def sync_write(self, *a, **kw):
            raise RuntimeError("Motor writes forbidden in calibration check")

    bus = ReadOnlyBus(port=args.port, motors={
        n: Motor(i, "sts3215", MotorNormMode.RANGE_0_100 if n == "gripper" else MotorNormMode.RANGE_M100_100)
        for i, n in enumerate(names, 1)
    })
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report = {"timestamp_utc": stamp, "port": args.port, "dispatch_enabled": False,
              "calibration_path": str(args.calibration.resolve()),
              "calibration_sha256": hashlib.sha256(raw).hexdigest(), "status": "failed"}
    try:
        bus.connect()
        actual = {n: asdict(c) for n, c in bus.read_calibration().items()}
        report["registers"] = actual
        report["present_position_raw"] = bus.sync_read("Present_Position", normalize=False)
        report["torque_enabled"] = {n: bus.read("Torque_Enable", n, normalize=False) for n in names}
        report["operating_mode"] = {n: bus.read("Operating_Mode", n, normalize=False) for n in names}
        report["gripper_control_registers"] = {
            field: bus.read(field, "gripper", normalize=False)
            for field in ("CW_Dead_Zone", "CCW_Dead_Zone", "P_Coefficient", "Goal_Position")
        }
        report["differences"] = compare(expected, actual)
        report["calibration_matches"] = not report["differences"]
        report["position_violations"] = [
            {"motor": n, "position": report["present_position_raw"][n],
             "minimum": expected[n]["range_min"], "maximum": expected[n]["range_max"]}
            for n in names if not expected[n]["range_min"] <= report["present_position_raw"][n] <= expected[n]["range_max"]
        ]
        report["status"] = ("mismatch" if report["differences"] else
                            "position_out_of_range" if report["position_violations"] else "match")
    except Exception:
        report["error_trace"] = traceback.format_exc()
    finally:
        try:
            # Default disconnect disables torque; explicitly avoid that motor write.
            if bus.is_connected:
                bus.disconnect(disable_torque=False)
        except Exception:
            report["close_error_trace"] = traceback.format_exc()
            report["status"] = "failed"
        output = Path("artifacts/safety") / f"live-calibration-{stamp}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"{report['status']}: {output}")
    return 0 if report["status"] == "match" else 1


if __name__ == "__main__":
    raise SystemExit(main())
