"""Readable development evaluation reports; no model dependencies."""
from pathlib import Path
from urllib.parse import quote


def cell(value):
    return str(value).replace('|', r'\|').replace('\n', ' ').replace('\r', ' ')


def write_report(result, output):
    output = Path(output)
    lines = ['# Evaluation report', '',
             f"Backend: **{result['backend']}** · Task: **{result['task']}**",
             f"Execution: **{result['execution_status']}** · Mode: **{result['mode']}**", '',
             'Task failure and execution failure are separate. Unknown or missing outcomes are not failures.', '',
             '| Policy | Expected | Recorded | Successes / known | Unknown | Missing | Success rate (known only) |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in result['comparison']:
        rate = row['success_rate_known']
        values = [row['policy'], row['expected_episodes'], row['recorded_episodes'],
                  f"{row['successes']} / {row['known_outcomes']}", row['unknown_outcomes'],
                  row['missing_episodes'], 'N/A' if rate is None else f'{rate:.1%}']
        lines.append('| ' + ' | '.join(map(cell, values)) + ' |')
    lines += ['', '## Execution and evidence', '', '| Job | Execution | Reason | Log |', '| --- | --- | --- | --- |']
    for job in result['jobs']:
        log = quote('logs/' + Path(job['log']).name)
        lines.append(f"| {cell(job['id'])} | {cell(job['execution_status'])} | {cell(job.get('error', '—'))} | [log]({log}) |")
    lines += ['', '## Episodes', '', '| Policy | Episode | Task status | Outcome | Reason | Evidence |', '| --- | ---: | --- | --- | --- | --- |']
    for row in result['episodes']:
        evidence = Path(row['evidence']['path'])
        try:
            target = quote(str(evidence.relative_to(output.resolve())))
        except ValueError:
            target = evidence.as_uri()
        metrics = row.get('metrics', {})
        reason = metrics.get('error') or metrics.get('rejection_reasons') or '—'
        outcome = 'unknown' if row['outcome'] is None else 'success' if row['outcome'] else 'failure'
        lines.append('| ' + ' | '.join(map(cell, [row['policy'], row['episode_index'], row['task_status'], outcome, reason])) + f' | [raw]({target}) |')
    lines += ['', '## Interpretation', '',
              'This report summarizes models within this backend and task. It does not rank models across backends or establish simulation-to-hardware transfer.',
              'Read unknown and missing counts before comparing rates. Formal statistical comparisons require a frozen protocol and verified evidence.', '',
              '[Resolved configuration and commands](plan.json) · [Machine-readable results and evidence hashes](results.json)', '']
    lines += ['- ' + note for note in result.get('notes', [])]
    temporary = output / 'report.md.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(output / 'report.md')
