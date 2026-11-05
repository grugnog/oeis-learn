"""Closed canonical job entry for temporary PAC orchestration; host is stdlib-only."""
import json
from pathlib import Path
from .identity import seal,confined,write,digest
from .generalization_preparation import reference
from .generalization_roles import project_and_invoke

OPERATIONS={'pac-assemble':('preparation',False),'pac-campaign-acceptance':('learner',True),'pac-calibrate':('learner',True)}


def main(args,workspace,host_factory):
    path=confined(workspace,args.job)
    if path.stat().st_size>16<<20:raise ValueError('job cap')
    job=json.loads(path.read_text())
    fields={'version','identity','operation','engine','images','source','artifacts','source_roots','settings','envelope','qualification','native_admission'}
    if set(job)!=fields or seal(job)!=job or job['version']!='pac-role-job/v1' or job['engine']!=args.engine:raise ValueError('closed source-bound canonical job required')
    if job['operation'] not in OPERATIONS:raise ValueError('unregistered PAC job')
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    if job['source']!=digest(collect_snapshot(workspace)):raise ValueError('job source changed')
    role,gpu=OPERATIONS[job['operation']]
    output=confined(workspace,args.output,exists=False)
    if output.exists():raise ValueError('fresh canonical output required')
    host=host_factory(args)
    if host.engine.name!=job['engine']:raise ValueError('actual engine mismatch')
    for image in job['images']:
        if host.engine.image(image)!=image:raise ValueError('actual image mismatch')
    output.mkdir(parents=True)
    try:
        if gpu:
            from oeis_learn.generalization.contracts import verify_artifact
            from oeis_learn.generalization.structural import original_cells
            native=job['native_admission']
            if not native or set(native)!={'root','producer','admission'}:raise ValueError('original native admission before GPU job required')
            native_root=confined(workspace,native['root'])
            producer=json.loads(verify_artifact(native_root,native['producer']))
            admission=json.loads(verify_artifact(native_root,native['admission']))
            if (seal(producer)!=producer or producer['status']!='PASS' or producer['source']!=job['source'] or producer['images']!=job['images'] or native['admission'] not in producer['inventory']
                or seal(admission)!=admission or admission['version'] not in {'pac-admission/v2','pac-admission/v3'} or admission['status']!='PASS' or admission['source']!=job['source'] or admission['images']!=job['images']):raise ValueError('incompatible native admission producer')
            if sorted(({k:v for k,v in x.items() if k!='ref'} for x in admission['originals']),key=lambda x:x['name'])!=sorted(original_cells(),key=lambda x:x['name']):raise ValueError('all original exclusions required')
            from oeis_learn.generalization.metrics import _pac_roster
            if (admission['version']=='pac-admission/v3')!=('structural_scope' in admission):raise ValueError('native admission scope version mismatch')
            _pac_roster(admission['roster'],scope=admission.get('structural_scope'))
            if job['operation']=='pac-calibrate' and admission['training']!=job['settings']['bundle']['identity']:raise ValueError('calibration bundle differs from native admission')
            from .generalization_qualification_io import consume_original
            if job['qualification'] is None:raise ValueError('original qualification required before GPU work')
            qualified=consume_original(host,workspace,confined(workspace,job['qualification']),output/'qualification',
                required_suites=['bigint-cpu','bigint-gpu'],required_capabilities=['weight-load','finite-update','native-exact'],pac_seconds=1800)
            if qualified['authority']['images']!=job['images']:raise ValueError('qualification image mismatch')
        result=project_and_invoke(host,job['images'][1 if gpu else 0],workspace,role,job['artifacts'],source_roots=job['source_roots'],
            output=str((output/'execution').relative_to(workspace)),operation=job['operation'],settings=job['settings'],
            seconds=job['envelope']['outer'],envelope=job['envelope'],gpu=gpu)
    except (Exception,KeyboardInterrupt) as exc:
        from .generalization_failures import public_failure
        result=dict(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL',reason=type(exc).__name__,error=str(exc),failure=public_failure(exc))
    from .generalization_lifecycle import inventory
    record=seal(dict(version='pac-job-result/v1',job=job['identity'],source=job['source'],images=job['images'],engine=job['engine'],
        status=result['status'],result=result,inventory=inventory(output)))
    write(output/'result.json',record)
    from oeis_learn.generalization.interfaces import EXIT_CODES
    return EXIT_CODES[result['status']]
