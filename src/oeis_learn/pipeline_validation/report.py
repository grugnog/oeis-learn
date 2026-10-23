"""Scope verdicts never imply research completion or adoption."""
EXIT={'PASS':0,'FAIL':1,'BLOCKED':3,'PARTIAL':3,'INCONCLUSIVE':3,'INTERRUPTED':130}


def summarize(receipts, *, interrupted=False, registry_unavailable=()):
    states={v['state'] for v in receipts.values()}
    verdict='INTERRUPTED' if interrupted else 'FAIL' if 'FAIL' in states else 'PARTIAL' if states & {'PARTIAL','INCONCLUSIVE','PENDING','READY','RUNNING'} else 'BLOCKED' if states & {'BLOCKED','UNAVAILABLE'} else 'PASS'
    covered={key for key,row in receipts.items() if row['state'] in {'PASS','REUSED'}}
    portability='PASS' if {'cpu.docker.semantic','cpu.podman.semantic'}<=covered else 'PENDING'
    qualified=receipts.get('qualification',{}).get('observations',{}).get('result',{}).get('engineering')
    research=receipts.get('study.report',{}).get('observations',{}).get('result',{}).get('research','NOT_EXECUTED')
    return dict(version='pipeline-summary/v1',selected_scope=verdict,stages=receipts,
        engineering=qualified or verdict,learning='NOT_ADMITTED',research=research,portability=portability,
        long_run_admission='NOT_ADMITTED',feature_complete=False,
        unavailable=list(registry_unavailable),historical={'008:T093':'DEFERRED','008:T096':'NEGATIVE 7730/14400 seconds'})
