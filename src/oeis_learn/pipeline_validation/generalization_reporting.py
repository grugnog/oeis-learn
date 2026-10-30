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


def measurement_report(study,invocation,attempts,*,output):
    """Export all projections from actual attempts, preserving unavailable timings."""
    from pathlib import Path
    from oeis_learn.generalization.contracts import validate
    from oeis_learn.generalization.reconciliation import reconcile_measurements
    validate('study',study)
    projections=reconcile_measurements(invocation,attempts,study)
    output=Path(output)
    for name,payload in projections.items():write(output/(name+'.json'),payload)
    write(output/'attempts.json',attempts)
    result=seal(dict(version='generalization-measurement-report/v1',study=study['identity'],
        status='PARTIAL' if projections['coverage']['missing_attempts'] else 'PASS',
        scientific_status='UNMEASURED',summary=projections['summary'],scope='recorded finite evidence'))
    write(output/'report.json',result)
    return result
