"""Reuse registered full-denominator matched reports and descriptive family intervals."""
from oeis_learn.fresh.comparison import compare_runs, family_bootstrap, promotion


def matched_runs(roots,output):
    # The existing implementation retains missing/interrupted arms, the exact
    # 128+128 denominators, all controls and fixed 2,000 family resamples.
    return compare_runs(roots,output)


def study_report(contract,invocation,output,receipts,*,workspace,prefix='',debits=()):
    """Preserve all planned arms; only full completed matches reach the inherited gate."""
    import json
    from pathlib import Path
    from .identity import seal,write,file_hash,reference
    invocation=Path(invocation);output=Path(output);workspace=Path(workspace)
    arms=[];roots=[];reports=[]
    for arm in contract['arms']:
        receipt=receipts.get(prefix+arm['id'],{})
        stage_root=invocation/'stages'/(prefix+arm['id'])
        run=reference(workspace,arm['run']).parent if arm.get('run') else stage_root/'run'
        root=stage_root/'analysis' if arm['operation'] in {'diagnose','evaluate'} else run
        path=root/'comparison.json';report=None;artifact=None;reason=receipt.get('reason','not started')
        if path.is_file() and not path.is_symlink():
            try:
                report=json.loads(path.read_text())
                artifact={'path':str(path.relative_to(workspace)),'sha256':file_hash(path)}
            except (OSError,ValueError) as exc:reason=str(exc)
        complete=receipt.get('state')=='PASS' and report is not None
        arms.append(dict(id=arm['id'],seed=arm['seed'],state=receipt.get('state','UNSTARTED'),
            reason=reason,producer=receipt.get('identity'),artifact=artifact,complete=complete))
        if complete:roots.append(root);reports.append(report)
    outcome='INCONCLUSIVE' if any(a['state'] in {'PASS','FAIL','PARTIAL','INCONCLUSIVE','RUNNING'} for a in arms) else 'NOT_EXECUTED';matched=None;error=None
    if arms and all(row['complete'] for row in arms):
        if contract['kind']=='representation' and len(arms)==6:
            try:
                matched=matched_runs(roots,output/'matched')
                decision=matched.get('decision')
                if decision is not None:
                    outcome='COMPLETE' if decision['promote'] else 'NEGATIVE' if decision['coverage'] else 'INCONCLUSIVE'
            except (ValueError,OSError,KeyError) as exc:error=str(exc)
        elif contract['kind']=='learnability':
            if all(r.get('memorization_gate') is True for r in reports):outcome='COMPLETE'
            elif all('memorization_gate' in r for r in reports):outcome='NEGATIVE'
        elif contract['kind']=='final':outcome='COMPLETE'
    result=seal(dict(version='pipeline-study-report/v1',study=contract['identity'],kind=contract['kind'],
        research=outcome,arms=arms,debits=list(debits),matched=matched,error=error,
        automatic_run_authorized=False,learning_promotion='separate reviewed decision required',
        interpretation='completed reporting is distinct from a positive scientific endpoint'))
    write(output/'report.json',result)
    return result
