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
    fields={'version','identity','operation','engine','images','source','artifacts','source_roots','settings','envelope','qualification'}
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
    if gpu:
        from .generalization_qualification_io import consume_original
        if job['qualification'] is None:raise ValueError('original qualification required before GPU work')
        qualified=consume_original(host,workspace,confined(workspace,job['qualification']),output/'qualification',
            required_suites=['bigint-cpu','bigint-gpu'],required_capabilities=['weight-load','finite-update','native-exact'])
        if qualified['authority']['images']!=job['images']:raise ValueError('qualification image mismatch')
    result=project_and_invoke(host,job['images'][1 if gpu else 0],workspace,role,job['artifacts'],source_roots=job['source_roots'],
        output=str((output/'execution').relative_to(workspace)),operation=job['operation'],settings=job['settings'],
        seconds=job['envelope']['outer'],envelope=job['envelope'],gpu=gpu)
    from .generalization_lifecycle import inventory
    record=seal(dict(version='pac-job-result/v1',job=job['identity'],source=job['source'],images=job['images'],engine=job['engine'],
        status=result['status'],result=result,inventory=inventory(output)))
    write(output/'result.json',record)
    from oeis_learn.generalization.interfaces import EXIT_CODES
    return EXIT_CODES[result['status']]
