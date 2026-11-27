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
    native_stop=steps.get('prepare-research/evaluation-timing',{}).get('status')=='FAILED'
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
        expected='FAILED' if native_stop and name=='evaluation-timing' else 'COMPLETE'
        if step.get('status')!=expected:raise Blocked('completed diagnostic preparation required: '+name)
        for ref in step.get('outputs',[]):
            checked_file(root,ref);originals[ref['path']]=ref
    def owned(path,maximum=32<<20):
        ref=reference(root,path)
        if originals.get(ref['path'])!=ref:raise Blocked('original diagnostic publication is not journal-bound')
        return _json(path,maximum=maximum)
    from .progressive_research_feasibility import VERSION,validate_request,estimates
    timing=base/'evaluation-timing';req=validate_request(owned(timing/'request.json'))
    version='progressive-research-evaluation-request/v2' if native_stop else VERSION
    if (req['version']!=version or req['source']!=receipt['current_source'] or
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
    if native_stop:
        native_identity_failure(timing,req,support,materials['native'])
        evidence=timing/'evaluation-cases.json'
    else:
        roster=owned(base/'roster/private/role-roster.json',ROSTER_DOCUMENT_BYTES)
        expected=estimates(timing,req,roster,publish=False)
        actual=read(timing/'evaluation-feasibility.json')
        if (actual!=expected or actual['status']!='BLOCKED' or not actual['deficits']
                or any(not d.startswith('evaluation_time:') or not d.endswith('-score') for d in actual['deficits'])):
            raise Blocked('original scoring-only resource deficit required')
        evidence=timing/'evaluation-feasibility.json'
    process=_json(confined(root,timing/'process.json'),maximum=1<<20)
    cleanup=_json(confined(root,timing/'cleanup.json'),maximum=1<<20)
    if (type(process.get('returncode'))is not int or process['returncode']!=(1 if native_stop else 0) or process.get('cleanup_failure')
            or cleanup.get('acknowledged') is not True or cleanup.get('within_bound') is not True or cleanup.get('failure')):
        raise Blocked('diagnostic timing cleanup is unproven')
    return seal(dict(version='progressive-preparation-owner-proof/v2',predecessor=receipt,
        session=session,journal=journal,qualification=req['qualification'],
        evaluation=reference(root,evidence),
        process=dict(original=reference(root,timing/'process.json'),value=process),
        cleanup=dict(original=reference(root,timing/'cleanup.json'),value=cleanup),
        reason='training-only native identity prework failure before campaign construction' if native_stop else 'training-only scoring feasibility blocked before campaign construction',
        original_costs_retained=True,scientific_allowance_transferred=False,
        historical_qualification_only=True,current_source_requalification_required=True))


def native_identity_failure(timing,req,support,native):
    """Only the original v2 CPU/GPU identity wiring failure, before service work."""
    from .progressive_scoring_calibration import request
    from .progressive_research_feasibility import verify_case
    base,service,samples=request(timing,req)
    if service['native']!=native:raise Blocked('original census native identity required')
    rows={r['draw_identity']:r for shard in support['shards'] for r in shard}
    for sample in samples:
        row=rows.get(sample['id'])
        if (not row or row.get('membership')!='training' or row.get('native_verified') is not True
                or sample['training_row']!=row['identity'] or any(sample[k]!=row[k] for k in ('source','indices','values'))):
            raise Blocked('original service witnesses must be training-only')
    result=read(base/'service-cases.json')
    if (result['request']!=service['identity'] or result['status']!='FAIL'
            or [r['index'] for r in result['cases']]!=list(range(8))):
        raise Blocked('complete original identity failure batch required')
    for row,case in zip(result['cases'],service['cases']):
        p=read(checked_file(base,row['producer']))
        if (row['status']!='FAIL' or p['status']!='FAIL' or p['request']!=service['identity']
                or p['case']!=case or p.get('error')!='ValueError'
                or p.get('message')!='original native runtime identity differs'
                or p.get('measurements')!={} or p.get('fit')!={} or 'scoring' in p):
            raise Blocked('only identity failure before fitting or scoring may be replaced')
    batch=read(timing/'evaluation-cases.json');failures=_json(timing/'failures.json')
    if (batch['request']!=req['identity'] or batch['status']!='FAIL' or len(batch['cases'])!=14
            or len(failures)!=1 or failures[0].get('case')!='positive-scoring-services'):
        raise Blocked('independent model timing must complete')
    for i,(row,case) in enumerate(zip(batch['cases'],req['cases'])):
        if row['index']!=i or row['case']!=case['identity'] or row['status']!='PASS':
            raise Blocked('complete ordered model timing required')
        p=verify_case(timing,req,case,row['producer'])
        if p['status']!='PASS' or p['device']!='actual_HIP':raise Blocked('original HIP timing required')
