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


def package_evidence(root,record,*,output):
    """Package only reconciled inventory bytes, preserving original producer paths."""
    import json
    from pathlib import Path
    from oeis_learn.generalization.bundles import verify_bundle,CAP
    from oeis_learn.generalization.contracts import verify_artifact
    from .bundle import pack,public
    from .identity import confined
    from .generalization_preparation import reference
    from .generalization_archive import verify_archive
    root=Path(root);output=Path(output)
    if output.exists() or output.with_suffix('.zip').exists():raise ValueError('fresh report output required')
    study=json.loads(verify_artifact(root,record['study']))
    verdict=verify_bundle(record,expected_study=study['identity'],root=root)
    # All mandatory payloads must be shareable; never silently omit sealed truth.
    if any(not public(Path(ref['path'])) for ref in record['inventory']):
        raise ValueError('private or unsupported evidence inventory member')
    if any(ref['path']=='evidence.json' for ref in record['inventory']):
        raise ValueError('reserved evidence manifest path')
    output.mkdir(parents=True,exist_ok=False);payloads=output/'payloads';payloads.mkdir()
    mandatory=['summary.json','evidence.json','payloads/evidence.json']
    for ref in record['inventory']:
        destination=confined(payloads,ref['path'],exists=False)
        destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as stream:stream.write(verify_artifact(root,ref,max_bytes=CAP))
        mandatory.append(str(destination.relative_to(output)))
    write(payloads/'evidence.json',record)
    write(output/'evidence.json',seal(dict(version='generalization-archive/v1',
        evidence=reference(payloads/'evidence.json',output),payload_root='payloads')))
    write(output/'summary.json',dict(selected_scope=verdict['software_status'],
        scientific_status=verdict['scientific_status'],integrity=verdict['integrity']))
    archive=pack(output,mandatory=mandatory,cap=CAP,member_cap=10000)
    try:verify_archive(archive)
    except BaseException:
        archive.unlink(missing_ok=True)
        raise
    return archive
