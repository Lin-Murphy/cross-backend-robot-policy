"""Backend-neutral result envelope; backend-specific evidence remains referenced, not flattened."""
from dataclasses import asdict,dataclass
from typing import Optional

STATUSES={'success','failed','timeout','rejected','aborted','uncertain'}
SCOPES={'development_trial','historical_pilot','saved_candidate_rejection'}


@dataclass(frozen=True)
class RunRecord:
    schema_version:int
    backend:str
    task:str
    policy:str
    scope:str
    status:str
    outcome:Optional[bool]
    outcome_source:Optional[str]
    predicted_actions:int
    applied_actions:int
    physical_dispatches:int
    task_feedback_available:bool
    capture_timestamps_verified:bool
    action_clock_kind:Optional[str]
    evidence_sha256:dict[str,str]

    def validate(self):
        if self.schema_version!=1 or not all(isinstance(x,str) and x for x in (self.backend,self.task,self.policy)):
            raise ValueError('invalid result identity')
        if self.scope not in SCOPES or self.status not in STATUSES:
            raise ValueError('invalid scope/status')
        if any(type(x) is not int or x<0 for x in (self.predicted_actions,self.applied_actions,self.physical_dispatches)):
            raise ValueError('invalid action counts')
        if self.applied_actions>self.predicted_actions:
            raise ValueError('applied actions exceed predicted actions')
        if type(self.task_feedback_available) is not bool or type(self.capture_timestamps_verified) is not bool:
            raise ValueError('invalid capabilities')
        if self.status=='rejected' and (self.applied_actions or self.physical_dispatches or self.outcome is not None):
            raise ValueError('rejected candidate cannot have action or task outcome')
        if self.status=='success' and self.outcome is not True:
            raise ValueError('success must have positive outcome')
        if self.outcome is not None and (type(self.outcome) is not bool or not self.outcome_source):
            raise ValueError('outcome missing valid source')
        if not self.task_feedback_available and self.outcome is not None:
            raise ValueError('task outcome without task feedback')
        if self.action_clock_kind not in (None,'simulation','host_monotonic'):
            raise ValueError('invalid action clock')
        if not self.evidence_sha256 or any(not k or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
                                               for k,v in self.evidence_sha256.items()):
            raise ValueError('invalid evidence hashes')
        return self

    def to_dict(self):
        self.validate()
        return asdict(self)
