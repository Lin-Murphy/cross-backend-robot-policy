"""Analyze the frozen 40-slot record package without altering any trial records."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT))
from cross_backend.trial_analysis import analyze_trials
from scripts.move_pot_trial_records import validate


def render(result):
    def percent(v):return 'undefined' if v is None else f'{100*v:.1f}%'
    def difference(v):return 'undefined' if v is None else f'{100*v:+.1f} percentage points'
    title='Synthetic Test Data: Not Experimental Results' if result['synthetic_fixture'] else 'Recorded Move-Pot Trial Analysis'
    lines=[f'# {title}','',
           'All started trials enter the denominator; unstarted trials are listed separately. Rates with running trials are provisional. Success counts use human-entered labels, not tool-proven task outcomes. Real pairs require file-integrity and frozen-identity checks; synthetic pairs only test calculations.','',
           '| Model / condition | Planned | Started | Not run | Running | Successes | Successes / started |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,g in result['groups'].items():
        lines.append(f"| {name} | {g['planned']} | {g['started']} | {g['not_run']} | {g['running']} | {g['confirmed_successes']} | {percent(g['success_rate_among_started'])} |")
    lines+=['','| Paired comparison | Difference direction | Analyzable pairs | Success proportion difference |','|---|---|---:|---:|']
    for c in result['paired_comparisons']:
        lines.append(f"| {c['comparison']} | {c['direction']} | {c['complete_recorded_pairs']} | {difference(c['paired_success_difference'])} |")
    lines+=['','See summary.json for failure classes, terminal durations, successful completion times, per-initial-state differences, and exclusions. Missing results are not replaced with zero success rates or invented completion times.',
            f"Trials passing real-evidence checks: {len(result['verified_trial_ids'])}; checks cover file integrity and frozen identity only, not human outcome judgment.",
            'These are descriptive statistics, not significance claims. Completion times cover successful trials only and must be interpreted alongside failure rates.','']
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    structural=validate(args.records)
    source=args.records/'trials.json';before=hashlib.sha256(source.read_bytes()).hexdigest()
    result=analyze_trials(json.loads(source.read_text()), verified_trial_ids=structural['verified_trial_ids'])
    result['record_validation']=structural;result['source_sha256']=before
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    (args.output/'report.md').write_text(render(result))
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    print('Analysis saved without modifying trial records:',args.output)
