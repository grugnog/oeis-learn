"""Closed prospective learning-pilot profile; parsing never supplies admission."""
from .identity import seal,validate,digest,reference

SPEC=dict(training=32768,parameter_development=128,structural_development=128,training_panel=32,
    oeis_development=8,batch=16,workers=4,seed=20261016,max_updates=1048576,max_tokens=1<<30,
    max_validations=4096,max_training_bins=4096,max_attempt_events=100000,attempt_event_bytes=2048,
    max_source_counters=262144,evaluation_rows=1776,scoring_outcomes=10656,
    snapshot_hours=[0,1.5,3,6,12,24],transient_bytes=16<<30,durable_bytes=32<<30,
    reservations=dict(learner=86400,evaluation=28800,qualification=3600,setup=3600,
        cleanup=900,archive=900,safety=48600))
from copy import deepcopy
LEGACY_SPEC=deepcopy(SPEC)
SPEC=deepcopy(SPEC);SPEC['max_source_counters']=524288
ARMS=('raw','rescue','equal_budget','shuffled','absent','template')
SPLITS=('training_panel','parameter_development','structural_development','oeis_development')
LEGACY_PROFILE=seal(dict(version='pipeline-learning-pilot-profile/v1',spec=LEGACY_SPEC,training=32768,
    batch=16,curriculum=False,replay=False,assisted=False,final=False,precision='float32',
    initialization='fresh_random',selection='visible-only/v2',extrapolation='conditional-backtested/v1'))
PROFILE=seal(dict(LEGACY_PROFILE,version='pipeline-learning-pilot-profile/v2',spec=SPEC))


def parse_profile(record):
    if record not in (LEGACY_PROFILE,PROFILE):raise ValueError('registered closed pilot profile required')
    return record


def preflight_manifest():
    from pathlib import Path
    import json
    value=json.loads((Path(__file__).resolve().parents[3]/'configs/foundation/pipeline_soak.json').read_text())
    value.update(version='pipeline-pilot-validation/v2',scope='engineering-pilot',
        stage_registry_version='pipeline-pilot-registry/v2',workloads=['learning-pilot-preflight/v2'],comparisons=[])
    value['budgets']['transient_bytes']=SPEC['transient_bytes']
    value['budgets']['free_disk_reserve_bytes']=5<<30
    value['budgets']['bundle_members']=4097
    value['budgets']['phase_seconds']={'1':10800,'2':1200,'3':1200,'4':28800,'5':12000}
    return seal(value)


def decision(*,source,prepared=None,qualification=None,cost=None):
    return seal(dict(version='pipeline-pilot-study/v1',state='DRAFT',profile=PROFILE,source=source,
        prepared=prepared,qualification=qualification,cost=cost,authority=None))


def decision_payload(record):
    return digest({k:v for k,v in record.items() if k not in {'identity','state','authority'}})


def qualification_requirements(engine):
    return {'host.contracts','cpu.'+engine+'.semantic','hip.preflight','policy.compare',
        'operations.'+engine+'.workflow','operations.'+engine+'.retirement','pilot.operations.'+engine,
        'pilot.prepare','pilot.session'}


def parse_decision(record,*,root=None,source=None):
    validate(record,{'version','state','profile','source','prepared','qualification','cost','authority'},'pipeline-pilot-study/v1')
    parse_profile(record['profile'])
    from oeis_learn.synthesis.state import require_hash
    require_hash(record['source'])
    if record['state'] not in {'DRAFT','FROZEN','ADMITTED'}:raise ValueError('prospective pilot decision required')
    missing=[k for k in ('prepared','qualification','cost','authority') if record[k] is None]
    if record['state']!='ADMITTED':missing.append('explicit_admission')
    if record['profile']!=PROFILE:missing.append('current_profile')
    if source is not None and source!=record['source']:missing.append('current_source')
    if root is None:missing.append('unresolved_authority')
    # Validate public authority before even resolving private prepared inputs.
    if root is not None and not missing:
        import json
        authority=json.loads(reference(root,record['authority']).read_text())
        validate(authority,{'version','decision_payload','approved','reviewed_by'},'pipeline-study-authority/v1')
        if authority['approved'] is not True or not authority['reviewed_by'] or authority['decision_payload']!=decision_payload(record):
            missing.append('reviewed_decision')
        q=json.loads(reference(root,record['qualification']).read_text())
        validate(q,{'version','source','profile','work','image','engine','resources','cpus','required','receipts','state'},'pipeline-pilot-qualification/v1')
        if q['state']!='PASS' or q['source']!=record['source'] or q['profile']!=PROFILE['identity']:
            missing.append('qualification')
        if set(q['image'])!={'gpu','qualification'}:raise ValueError('both qualified image identities required')
        for image in q['image'].values():require_hash(image)
        require_hash(q['resources'])
        if type(q['cpus']) is not int or not 6<=q['cpus']<=16:raise ValueError('qualified bounded CPU quota required')
        if (q['engine'] not in ('podman','docker') or not isinstance(q['required'],list)
            or len(q['required'])!=len(set(q['required'])) or set(q['required'])!=qualification_requirements(q['engine'])
            or set(q['receipts'])!=set(q['required'])
            or any(q['receipts'].get(k,{}).get('state') not in ('PASS','REUSED') for k in q['required'])):
            missing.append('qualification_receipts')
        for r in q['receipts'].values():
            if set(r)!={'state','identity'}:raise ValueError('qualified receipt identity required')
            require_hash(r['identity'])
        c=json.loads(reference(root,record['cost']).read_text())
        from .pilot_cost import validate_forecast
        if not validate_forecast(c,source=record['source']):missing.append('exact_scope_cost')
        if not missing:
            prepared=json.loads(reference(root,record['prepared']).read_text())
            if c['work']!=prepared['identity'] or q['work']!=prepared['identity']:missing.append('cost_work_binding')
    return dict(status='BLOCKED' if missing else 'ADMITTED',missing=sorted(set(missing)),contract=record)
