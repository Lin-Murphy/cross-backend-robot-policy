"""Backend-neutral milestone view beside RunRecord; unknown evidence stays unknown."""
from dataclasses import dataclass

STATES = {'yes', 'no', 'unknown'}


def combine_required(stages, required):
    if not required or any(name not in stages for name in required):
        raise ValueError('required milestone missing')
    values = [stages[name]['state'] for name in required]
    if 'no' in values:
        return 'no'
    return 'yes' if all(value == 'yes' for value in values) else 'unknown'


@dataclass(frozen=True)
class CompletionRecord:
    backend: str
    task: str
    run_status: str
    stages: dict
    task_required: tuple[str, ...]
    full_cycle_required: tuple[str, ...]
    evidence_sha256: dict[str, str]

    def to_dict(self):
        if not self.backend or not self.task or not self.run_status or not self.stages:
            raise ValueError('missing completion identity')
        for name, item in self.stages.items():
            if not name or set(item) != {'state', 'source'} or item['state'] not in STATES or not item['source']:
                raise ValueError('invalid milestone evidence')
        if not self.evidence_sha256 or any(not key or len(value) != 64 or
            any(c not in '0123456789abcdef' for c in value)
            for key, value in self.evidence_sha256.items()):
            raise ValueError('invalid evidence digest')
        task_state = combine_required(self.stages, self.task_required)
        cycle_state = combine_required(self.stages, self.full_cycle_required)
        return {'schema_version': 1, 'backend': self.backend, 'task': self.task,
                'run_status': self.run_status, 'stages': self.stages,
                'task_required': list(self.task_required), 'task_state': task_state,
                'full_cycle_required': list(self.full_cycle_required),
                'full_cycle_state': cycle_state, 'evidence_sha256': self.evidence_sha256}
