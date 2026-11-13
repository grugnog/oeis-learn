"""Preserve a diagnostic timing attempt before a fresh planning correction."""
from .identity import confined,seal,digest
from .progressive_workstation_state import Blocked,read
from .progressive_research_inputs import checked_file,_json
from oeis_learn.generalization.progressive.contracts import reference
from oeis_learn.generalization.progressive.research_roster import ROSTER_DOCUMENT_BYTES


def proof(root,receipt,session,journal):
    from .progressive_preparation_owner import EARLY
    folder=confined(root,receipt['recipient']);base=folder/'prepare-research'
    allowed=EARLY|{'base-qualification','matrix','qualification-plan','qualify',
                   'evaluation-plan','evaluation-timing'}
    steps=journal['steps']
    for key,row in steps.items():
        if not key.startswith('prepare-research/'):
            raise Blocked('scientific or other execution cannot receive a planning replacement')
        name=key.split('/',1)[1]
        for attempt in [row,*row.get('history',[])]:
            if attempt['status'] in {'RUNNING','INTERRUPTED'} or (name not in allowed and attempt['status']!='BLOCKED'):
                raise Blocked('started downstream or unresolved preparation cannot be replaced')
    if any((base/n).exists() for n in ('construction','prepared','frozen','acceptance')):
        raise Blocked('constructed campaign cannot receive a planning replacement')
    if steps.get('prepare-research/evaluation-feasibility',{}).get('status')!='BLOCKED':
        raise Blocked('original evaluation-feasibility stop required')
    originals={}
    for name in ('base-qualification','qualify','materials','role-roster','evaluation-plan','evaluation-timing'):
        step=steps.get('prepare-research/'+name,{})
        if step.get('status')!='COMPLETE':raise Blocked('completed diagnostic preparation required: '+name)
        for ref in step['outputs']:
            checked_file(root,ref);originals[ref['path']]=ref
    def owned(path,maximum=32<<20):
        ref=reference(root,path)
        if originals.get(ref['path'])!=ref:raise Blocked('original diagnostic publication is not journal-bound')
        return _json(path,maximum=maximum)
    from .progressive_research_feasibility import VERSION,validate_request,estimates
    timing=base/'evaluation-timing';req=validate_request(owned(timing/'request.json'))
    if (req['version']!=VERSION or req['source']!=receipt['current_source'] or
            req['scope']!='training-only prospective throughput; no panel exposure'):
        raise Blocked('original training-only timing scope required')
    from .progressive_handoff import original_authority
    q,qroot=original_authority(root,req['qualification'])
    if digest(_json(checked_file(qroot,q['source_closure'])))!=req['source']:
        raise Blocked('original timing and qualification source differ')
    materials=owned(base/'materials/materials.json')
    support=_json(checked_file(root,materials['support']))
    if (support['identity']!=req['support'] or seal(support)!=support or
            support.get('diagnostic') is not False or materials.get('assessment_exposed') is not False):
        raise Blocked('original unexposed training support required')
    rows={r['draw_identity']:r for shard in support['shards'] for r in shard}
    for sample in _json(checked_file(timing,req['samples'])):
        row=rows.get(sample['id'])
        if (not row or row.get('membership')!='training' or row.get('native_verified') is not True
                or any(sample[k]!=row[k] for k in ('indices','values'))):
            raise Blocked('diagnostic timing touched non-training members')
    roster=owned(base/'roster/private/role-roster.json',ROSTER_DOCUMENT_BYTES)
    expected=estimates(timing,req,roster,publish=False)
    actual=read(timing/'evaluation-feasibility.json')
    if (actual!=expected or actual['status']!='BLOCKED' or not actual['deficits']
            or any(not d.startswith('evaluation_time:') or not d.endswith('-score') for d in actual['deficits'])):
        raise Blocked('original scoring-only resource deficit required')
    process=_json(confined(root,timing/'process.json'),maximum=1<<20)
    cleanup=_json(confined(root,timing/'cleanup.json'),maximum=1<<20)
    if (type(process.get('returncode'))is not int or process['returncode']!=0 or process.get('cleanup_failure')
            or cleanup.get('acknowledged') is not True or cleanup.get('within_bound') is not True or cleanup.get('failure')):
        raise Blocked('diagnostic timing cleanup is unproven')
    return seal(dict(version='progressive-preparation-owner-proof/v2',predecessor=receipt,
        session=session,journal=journal,qualification=req['qualification'],
        evaluation=reference(root,timing/'evaluation-feasibility.json'),
        process=dict(original=reference(root,timing/'process.json'),value=process),
        cleanup=dict(original=reference(root,timing/'cleanup.json'),value=cleanup),
        reason='training-only scoring feasibility blocked before campaign construction',
        original_costs_retained=True,scientific_allowance_transferred=False,
        historical_qualification_only=True,current_source_requalification_required=True))
