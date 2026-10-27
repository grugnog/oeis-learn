"""Opt-in original-clock pilot authority; historical v1 decisions stay unchanged."""
from copy import deepcopy
import json
from .identity import seal,digest,validate,reference
from .pilot_contract import PROFILE as DATA_PROFILE,SPEC as DATA_SPEC,decision_payload
from .pilot_clocks import POLICY

PURPOSE='longrun-clock60'
SPEC=deepcopy(DATA_SPEC)
SPEC['reservations'].update(evaluation=36000,safety=41400)
PROFILE=seal(dict(DATA_PROFILE,version='pipeline-learning-pilot-profile/v3',spec=SPEC,evaluation_policy=POLICY))


def full_clock(purpose):return purpose in ('calibration',PURPOSE)


def cost_authority(candidate,*,source,compatibility):
    from .pilot_calibration_cost import forecast
    expected=forecast(candidate.get('observations',[]),source=candidate.get('source'),work=candidate.get('work'),archives=candidate.get('archives',[]))
    if candidate!=expected or not expected['within_budget']:raise ValueError('complete compatible full-clock calibration required')
    from oeis_learn.synthesis.state import require_hash
    for h in (source,compatibility):require_hash(h)
    return seal(dict(version='pipeline-pilot-cost/v2',profile=PROFILE['identity'],source=source,
        work=candidate['work'],compatibility=compatibility,calibration=candidate,evaluation_policy=POLICY,
        generator_role_seconds=candidate['generator_role_seconds'],scorer_role_seconds=candidate['scorer_role_seconds'],
        evaluation_seconds=candidate['evaluation_seconds'],projected_seconds=candidate['projected_seconds'],
        archives=candidate['archives'],budget_seconds=172800,within_budget=True,status='ADMISSIBLE_COST_ONLY'))


def valid_cost(record,*,source,compatibility):
    try:return record==cost_authority(record['calibration'],source=source,compatibility=compatibility)
    except (KeyError,TypeError,ValueError):return False


def decision(*,source,prepared,qualification,cost,compatibility):
    return seal(dict(version='pipeline-pilot-study/v2',state='DRAFT',profile=PROFILE,source=source,
        prepared=prepared,qualification=qualification,cost=cost,compatibility=compatibility,authority=None))


def requirements(engine):
    return {'host.contracts','engine.'+engine,'image.'+engine+'.gpu','image.'+engine+'.qualification',
        'cpu.'+engine+'.semantic','hip.preflight','policy.compare','operations.'+engine+'.workflow',
        'operations.'+engine+'.retirement','pilot.operations.'+engine}


def parse_decision(record,*,root=None,source=None):
    validate(record,{'version','state','profile','source','prepared','qualification','cost','compatibility','authority'},'pipeline-pilot-study/v2')
    if record['profile']!=PROFILE:raise ValueError('exact full-clock v3 profile required')
    from oeis_learn.synthesis.state import require_hash
    require_hash(record['source'])
    if record['state'] not in ('DRAFT','FROZEN','ADMITTED'):raise ValueError('prospective pilot state required')
    missing=[k for k in ('prepared','qualification','cost','compatibility','authority') if record[k] is None]
    if record['state']!='ADMITTED':missing.append('explicit_admission')
    if source is not None and source!=record['source']:missing.append('current_source')
    if root is None:missing.append('unresolved_authority')
    if not missing:
        read=lambda name:json.loads(reference(root,record[name]).read_text())
        a=read('authority')
        validate(a,{'version','decision_payload','approved','reviewed_by'},'pipeline-study-authority/v2')
        if a['approved'] is not True or not isinstance(a['reviewed_by'],str) or not a['reviewed_by'].strip() or a['decision_payload']!=decision_payload(record):
            missing.append('reviewed_decision')
        q=read('qualification')
        validate(q,{'version','source','profile','work','image','engine','resources','cpus','required','receipts','receipt_refs','state','compatibility'},'pipeline-pilot-qualification/v2')
        compat=read('compatibility')
        from .longrun_import import validate_compatibility
        validate_compatibility(compat,source=record['source'])
        if q['source']!=record['source'] or q['profile']!=PROFILE['identity'] or q['state']!='PASS' or q['compatibility']!=compat['identity']:
            missing.append('qualification')
        if q['engine'] not in ('podman','docker') or type(q['cpus']) is not int or not 6<=q['cpus']<=16:raise ValueError('bounded qualified engine/CPU required')
        require_hash(q['resources'])
        if set(q['image'])!={'gpu','qualification'}:raise ValueError('both qualified images required')
        for h in q['image'].values():require_hash(h)
        required=requirements(q['engine'])
        if (not isinstance(q['required'],list) or len(q['required'])!=len(required) or set(q['required'])!=required
            or set(q['receipts'])!=required):missing.append('qualification_receipts')
        for r in q['receipts'].values():
            if set(r)!={'state','identity'}:raise ValueError('closed receipt identity required')
            require_hash(r['identity'])
            if r['state']!='PASS':missing.append('qualification_receipts')
        if set(q['receipt_refs'])!=required:missing.append('qualification_receipts')
        if not missing:
            for key,ref in q['receipt_refs'].items():
                receipt=json.loads(reference(root,ref).read_text())
                if (seal(receipt)!=receipt or receipt.get('stage')!=key or receipt.get('state')!='PASS'
                    or receipt.get('identity')!=q['receipts'][key]['identity'] or receipt.get('binding',{}).get('source')!=record['source']):
                    missing.append('qualification_receipts')
        c=read('cost')
        if not valid_cost(c,source=record['source'],compatibility=compat['identity']):missing.append('exact_scope_cost')
        if (compat['engine']!=q['engine'] or compat['cpus']!=q['cpus'] or compat['resources']!=q['resources']
            or compat['work']!=q['work'] or c.get('work')!=q['work']):missing.append('cost_work_binding')
        # Never resolve private data until every public approval/qualification/cost gate passed.
        if not missing:
            prepared=read('prepared')
            if prepared['identity']!=q['work'] or prepared['source']!=compat['producer_source']:missing.append('prepared_work_binding')
    return dict(status='BLOCKED' if missing else 'ADMITTED',missing=sorted(set(missing)),contract=record)
