"""Scope verdicts never imply research completion or adoption."""
EXIT={'PASS':0,'FAIL':1,'BLOCKED':3,'PARTIAL':3,'INCONCLUSIVE':3,'INTERRUPTED':130}


def summarize(receipts, *, interrupted=False, registry_unavailable=()):
    states={v['state'] for v in receipts.values()}
    verdict='INTERRUPTED' if interrupted else 'FAIL' if 'FAIL' in states else 'PARTIAL' if states & {'PARTIAL','INCONCLUSIVE'} else 'BLOCKED' if states & {'BLOCKED','UNAVAILABLE'} else 'PASS'
    return dict(version='pipeline-summary/v1',selected_scope=verdict,stages=receipts,
        engineering=verdict,learning='NOT_ADMITTED',research='NOT_EXECUTED',portability='PENDING',
        long_run_admission='NOT_ADMITTED',feature_complete=False,
        unavailable=list(registry_unavailable),historical={'008:T093':'DEFERRED','008:T096':'NEGATIVE 7730/14400 seconds'})
