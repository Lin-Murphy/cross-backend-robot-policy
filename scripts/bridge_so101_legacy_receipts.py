"""Convert saved SO101 audit into shared transport-scope receipts; no device access."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from cross_backend.legacy_receipt_bridge import bridge_legacy_receipts, bridge_legacy_stop


def main(run, output):
    summary_path = run / 'summary.json'
    events_path = run / 'hardware-events.jsonl'
    summary = json.loads(summary_path.read_text())
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    capabilities, receipts = bridge_legacy_receipts(events, summary['raw_goal_packets_transmitted'])
    stop = bridge_legacy_stop(events)
    result = {'backend_capabilities': capabilities, 'receipts': receipts, 'stop_receipt': stop,
        'accepted': sum(r['accepted'] for r in receipts),
        'rejected': sum(not r['accepted'] for r in receipts),
        'physical_dispatches': sum(r['physical_dispatches'] for r in receipts),
        'per_motor_acknowledgements_verified': False,
        'source_exposure_timestamps_verified': False,
        'evidence_sha256': {'summary': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                            'events': hashlib.sha256(events_path.read_bytes()).hexdigest()}}
    output.mkdir(parents=True, exist_ok=False)
    (output / 'receipts.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('accepted', 'rejected', 'physical_dispatches')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    main(a.run, a.output)
