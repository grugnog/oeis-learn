"""Retain a terminated metadata-only compiler attempt before requalification."""
from .identity import confined,seal,digest
from .progressive_workstation_state import Blocked,read
from .progressive_research_inputs import checked_file,_json
from oeis_learn.generalization.progressive.contracts import reference


def proof(root,receipt,session,journal):
    from .progressive_preparation_owner import EARLY
    from .progressive_handoff import original_authority
    from .progressive_research_feasibility import validate_request,verify_contract
    folder=confined(root,receipt['recipient']);base=folder/'prepare-research'
    allowed=EARLY|{'base-qualification','matrix','qualification-plan','qualify',
                   'evaluation-plan','evaluation-timing','evaluation-feasibility','construction'}
    steps=journal['steps']
    for key,row in steps.items():
        if not key.startswith('prepare-research/'):
            raise Blocked('scientific execution cannot receive construction replacement')
        name=key.split('/',1)[1]
        for attempt in [row,*row.get('history',[])]:
            status=attempt['status']
            if status in {'RUNNING','INTERRUPTED'} or (name not in allowed and status!='BLOCKED'):
                raise Blocked('started downstream or unresolved construction work')
            if name=='construction' and status not in {'FAILED','BLOCKED'}:
                raise Blocked('completed construction cannot be replaced')
    if any((base/name).exists() for name in ('prepared','frozen','acceptance','construction/execution','construction/construction-summary.json')):
        raise Blocked('calibration, campaign or completed construction already exists')
    originals={}
    for name in ('base-qualification','qualify','materials','role-roster','evaluation-plan','evaluation-timing','evaluation-feasibility'):
        row=steps.get('prepare-research/'+name,{})
        if row.get('status')!='COMPLETE':raise Blocked('original completed prerequisite required: '+name)
        for ref in row.get('outputs',[]):checked_file(root,ref);originals[ref['path']]=ref
    def owned(path):
        ref=reference(root,path)
        if originals.get(ref['path'])!=ref:raise Blocked('construction prerequisite is not journal-bound')
        return read(path)
    req=validate_request(owned(base/'evaluation-timing/request.json'))
    if req['source']!=receipt['current_source'] or req['scope']!='training-only prospective throughput; no panel exposure':
        raise Blocked('original unexposed timing scope required')
    q,qroot=original_authority(root,req['qualification'])
    if digest(_json(checked_file(qroot,q['source_closure'])))!=req['source']:
        raise Blocked('construction qualification source differs')
    contract=owned(base/'execution-contract.json')
    verify_contract(root,reference(root,base/'execution-contract.json'),dict(payload=dict(
        qualification=[req['qualification']],scope=dict(images=q['images'],execution_policy=q['runtime']['execution_policy']))))
    materials=owned(base/'materials/materials.json')
    if materials.get('assessment_exposed') is not False or materials.get('diagnostic') is not False:
        raise Blocked('unexposed research materials required')
    support=_json(checked_file(root,materials['support']))
    if support['identity']!=req['support'] or seal(support)!=support:raise Blocked('original support differs')
    rows={r['draw_identity']:r for shard in support['shards'] for r in shard}
    samples=_json(checked_file(base/'evaluation-timing',req['samples']))
    if 'scoring_services' in req:
        from .progressive_scoring_calibration import request
        samples+=request(base/'evaluation-timing',req)[2]
    for sample in samples:
        row=rows.get(sample['id'])
        if (not row or row.get('membership')!='training' or row.get('native_verified') is not True
                or any(sample[k]!=row[k] for k in ('indices','values'))
                or ('source' in sample and (sample['source']!=row['source'] or sample['training_row']!=row['identity']))):
            raise Blocked('construction timing must use original training members')
    logs=base/'logs/construction';command=_json(logs/'command.json');process=_json(logs/'process.json')
    code='import sys;sys.path.insert(0,sys.argv.pop(1));from oeis_learn.pipeline_validation.progressive_research_bindings import main;main()'
    args=command['argv']
    expected=['-S','-c',code,str(root/'src'),'--root',str(root)]
    for flag,path in [('output','construction'),('data','registered/registered-data.json'),('design','research-round-design.json'),
                      ('materials','materials/materials.json'),('qualification','qualification'),('execution-contract','execution-contract.json')]:
        expected+=['--'+flag,str((base/path).relative_to(root))]
    if (args[1:]!=expected or command.get('seconds')!=7200 or command.get('cleanup_seconds')!=5
            or process.get('returncode') not in {-15,-9} or process.get('cleanup_failure') is not None
            or process.get('outcome')!='FAIL' or not str(process.get('failure','')).startswith('TimeoutError: external deadline:')
            or not 7200<=process.get('elapsed_seconds',0)<=7235):
        raise Blocked('original bounded compiler timeout with acknowledged termination required')
    return seal(dict(version='progressive-preparation-owner-proof/v3',predecessor=receipt,session=session,journal=journal,
        qualification=req['qualification'],execution_contract=contract,
        command=dict(original=reference(root,logs/'command.json'),value=command),
        process=dict(original=reference(root,logs/'process.json'),value=process),
        reason='metadata-only construction timeout before calibration or scientific execution',
        original_costs_retained=True,scientific_allowance_transferred=False,
        historical_qualification_only=True,current_source_requalification_required=True))
