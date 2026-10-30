"""Read-only export adapters. Unknown measurements remain unavailable."""
from dataclasses import asdict
from fractions import Fraction
from oeis_learn.generalization.evidence import reconcile,checkpoint_roster
from .identity import seal,write


def recorded_report(records,*,study,source,roster,checkpoints,available,output=None):
    summary=asdict(reconcile(records,study=study,source=source,roster=roster))
    summary={k:str(v) if isinstance(v,Fraction) else v for k,v in summary.items()}
    report=seal(dict(version='generalization-recorded-report/v1',study=study,source=source,summary=summary,
                     checkpoints=checkpoint_roster(checkpoints,available),scientific_status='UNMEASURED',
                     scope='finite recorded evidence',automatic_run_authorized=False))
    if output is not None: write(output,report)
    return report
