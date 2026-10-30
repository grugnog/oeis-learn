"""Append-only workflow policy and fresh-root recovery decisions."""
from dataclasses import dataclass, replace
from .interfaces import EXIT_CODES, SoftwareStatus

@dataclass(frozen=True)
class Workflow:
    study: str
    required: tuple[str,...]
    receipts: tuple[tuple[str,str],...] = ()
    active: str | None = None

    def __post_init__(self):
        if not self.study or not self.required or len(set(self.required))!=len(self.required): raise ValueError('frozen workflow required')
        if len({r[0] for r in self.receipts})!=len(self.receipts) or any(k not in self.required or s not in EXIT_CODES for k,s in self.receipts):
            raise ValueError('invalid terminal receipts')

    def start(self,stage):
        completed=dict(self.receipts)
        if self.active or stage not in self.required or stage in completed: raise ValueError('stage cannot restart in terminal root')
        prior=self.required[:self.required.index(stage)]
        if any(completed.get(k)!='PASS' for k in prior): raise ValueError('missing successful prerequisites')
        return replace(self,active=stage)

    def finish(self,stage,status):
        if stage!=self.active or status not in EXIT_CODES: raise ValueError('invalid terminal transition')
        return replace(self,active=None,receipts=self.receipts+((stage,str(status)),))

    @property
    def status(self):
        statuses={s for _,s in self.receipts}
        for value in ('FAIL','INTERRUPTED','BLOCKED','INCONCLUSIVE'):
            if value in statuses: return value
        return 'PASS' if len(self.receipts)==len(self.required) and statuses=={'PASS'} and not self.active else 'PARTIAL'

    @property
    def exit_code(self): return EXIT_CODES[SoftwareStatus(self.status)]


def restart(workflow,*,old_root,new_root,reason,learner,full_state=None):
    from pathlib import Path
    if Path(old_root).resolve()==Path(new_root).resolve() or Path(new_root).exists(): raise ValueError('fresh output root required')
    if reason not in {'interrupted','invalidated_source','incompatible_qualification','report_failure','failed_stage','missing_payload'}: raise ValueError('precise repeat reason required')
    if learner and (not isinstance(full_state,dict) or full_state.get('kind')!='full_state'): raise ValueError('full-state learner resume required')
    return dict(study=workflow.study,parent_root=str(old_root),output_root=str(new_root),reason=reason,
                immutable_receipts=workflow.receipts,reuse_requires_original_verification=True,
                resume=full_state if learner else None)
