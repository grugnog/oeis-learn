"""Explicit prospective study authority and gates. Parsing cannot authorize execution."""
import json
from .identity import validate,reference,number,digest

FIELDS={'version','kind','decision_id','hypothesis','state','arms','seeds','track','prior_decisions',
        'gates','cost','budgets','denominators','rules','authority','final_plan'}
KINDS={'learnability','representation','curriculum','replay','assistance','encoder','final'}
EDGES={'DRAFT':{'FROZEN'},'FROZEN':{'ADMITTED','BLOCKED'},'ADMITTED':{'RUNNING','BLOCKED'},
       'RUNNING':{'COMPLETE','NEGATIVE','INCONCLUSIVE','INTERRUPTED'}}


def lifecycle(previous,current):
    if current not in EDGES.get(previous,set()):
        raise ValueError('illegal study transition; terminal outcomes are immutable')
    return current


def gate_requirements(kind):
    if kind not in KINDS:
        raise ValueError('unknown study kind')
    gates={'runtime','operations','device','resources','provenance','cost','decision'}
    if kind!='learnability':gates.add('learning')
    if kind=='final':gates.add('final_plan')
    if kind in {'curriculum','replay','assistance','encoder','final'}:gates.add('recipe')
    if kind=='replay':gates.add('curriculum_decision')
    if kind=='assistance':gates.add('reviewed_assistance')
    return gates


def parse_study(record,*,root=None,source=None,final_plan=None):
    validate(record,FIELDS,'pipeline-study/v1')
    required=gate_requirements(record['kind'])
    if record['state'] not in EDGES and record['state'] not in {'COMPLETE','NEGATIVE','INCONCLUSIVE','INTERRUPTED','BLOCKED'}:
        raise ValueError('unknown study state')
    if not record['decision_id'] or not record['hypothesis'] or record['track'] not in {'generic_only','assisted'}:
        raise ValueError('prospective hypothesis and provenance required')
    if not isinstance(record['seeds'],list) or not record['seeds'] or len(set(record['seeds']))!=len(record['seeds']) or any(type(s) is not int for s in record['seeds']):
        raise ValueError('frozen unique seeds required')
    if record['kind']=='representation' and record['seeds']!=[20261016,20261017,20261018]:
        raise ValueError('registered paired seeds required; replacements need a new reviewed policy')
    if record['denominators']!={'visible':20,'horizon':100}:
        raise ValueError('visible20/continuation100 required')
    budgets=record['budgets']
    if set(budgets)!={'wall_seconds','work','disk_bytes','cleanup_seconds','archive_seconds'}:
        raise ValueError('explicit finite shared budget fields required')
    for value in budgets.values():number(value,positive=True)
    if budgets['cleanup_seconds']+budgets['archive_seconds']>=budgets['wall_seconds']:
        raise ValueError('cleanup/archive reservation cannot fit')
    if set(record['rules'])!={'stop','promote'} or record['rules']['stop']!='negative-or-inconclusive':
        raise ValueError('explicit stop/promotion rules required')
    if not isinstance(record['gates'],dict) or set(record['gates'])-required:
        raise ValueError('unknown gate authority')
    missing=sorted(required-set(record['gates']))
    if record['authority'] is None:missing.append('authority')
    if record['cost'] is None:missing.append('cost_receipt')
    if not record['arms']:missing.append('arms')
    if record['state']!='ADMITTED':missing.append('explicit_admission')
    ids=set();totals={'wall_seconds':budgets['cleanup_seconds']+budgets['archive_seconds'],'work':0,'disk_bytes':0}
    for arm in record['arms']:
        if set(arm)!={'id','operation','inputs','config','run','seed','checkpoint','wall_seconds','work','disk_bytes'}:
            raise ValueError('closed arm specification required')
        if arm['id'] in ids or not isinstance(arm['id'],str) or not arm['id'].isalnum():
            raise ValueError('unique alphanumeric arm IDs required')
        ids.add(arm['id'])
        allowed={'learnability'} if record['kind']=='learnability' else {'evaluate'} if record['kind']=='final' else {'train','diagnose'}
        if arm['operation'] not in allowed or arm['seed'] not in record['seeds']:
            raise ValueError('study kind/operation/seed mismatch')
        for key in totals:
            totals[key]+=number(arm[key],positive=True)
    if any(totals[k]>budgets[k] for k in totals):
        raise ValueError('arm reservations must fit shared study budget')
    if root is not None:
        # All authority is read and checked before reading any input/checkpoint ref.
        for name,ref in record['gates'].items():
            gate=json.loads(reference(root,ref).read_text())
            if gate.get('state')!='PASS' or gate.get('scope')!=name or (source is not None and gate.get('source')!=source):
                missing.append('gate:'+name)
        if record['authority'] is not None:
            authority=json.loads(reference(root,record['authority']).read_text())
            if authority.get('decision_payload')!=digest({k:v for k,v in record.items() if k not in {'identity','authority','state'}}) or authority.get('approved') is not True:
                missing.append('reviewed_decision')
        if record['cost'] is not None:
            cost=json.loads(reference(root,record['cost']).read_text())
            if cost.get('scope')!=('diverse32' if record['kind']=='learnability' else 'transfer-screen' if record['kind']=='representation' else record['kind']) or cost.get('within_budget') is not True:
                missing.append('exact_scope_cost')
        if record['kind']=='final':
            if final_plan is None or record['final_plan'] is None or reference(root,record['final_plan'])!=final_plan:
                missing.append('explicit_final_plan')
    if record['kind']=='final' and record['final_plan'] is None:missing.append('final_plan')
    return dict(status='BLOCKED' if missing else 'ADMITTED',missing=sorted(set(missing)),contract=record)
