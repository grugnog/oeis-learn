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
        fields={'id','operation','inputs','config','run','seed','checkpoint','wall_seconds','work','disk_bytes'}
        if not fields<=set(arm) or set(arm)-fields-{'attempt'}:
            raise ValueError('closed arm specification required')
        if arm['id'] in ids or not isinstance(arm['id'],str) or not arm['id'].isalnum():
            raise ValueError('unique alphanumeric arm IDs required')
        ids.add(arm['id'])
        if number(arm.get('attempt',arm)['wall_seconds'],positive=True)<=5:
            raise ValueError('arm attempt must reserve more than five seconds including cleanup')
        allowed={'learnability'} if record['kind']=='learnability' else {'evaluate'} if record['kind']=='final' else {'train','diagnose'}
        if arm['operation'] not in allowed or arm['seed'] not in record['seeds']:
            raise ValueError('study kind/operation/seed mismatch')
        for key in totals:
            totals[key]+=number(arm[key],positive=True)
        if 'attempt' in arm:
            if record['kind']!='final' or set(arm['attempt'])!={'wall_seconds','work','disk_bytes'}:
                raise ValueError('only final plans may predeclare bounded remaining-budget attempts')
            for key,value in arm['attempt'].items():
                if number(value,positive=True)>arm[key]:raise ValueError('attempt exceeds cumulative arm allowance')
            attempts=min(int(arm[k]//arm['attempt'][k]) for k in arm['attempt'])
            totals['wall_seconds']+=(attempts-1)*(budgets['cleanup_seconds']+budgets['archive_seconds'])
    if any(totals[k]>budgets[k] for k in totals):
        raise ValueError('arm reservations must fit shared study budget')
    if root is not None:
        # All authority is read and checked before reading any input/checkpoint ref.
        for name,ref in record['gates'].items():
            gate=json.loads(reference(root,ref).read_text())
            validate(gate,{'version','scope','state','source','image','resources','producer','engine'},'pipeline-study-gate/v1')
            if gate.get('state')!='PASS' or gate.get('scope')!=name or (source is not None and gate.get('source')!=source):
                missing.append('gate:'+name)
        if record['authority'] is not None:
            authority=json.loads(reference(root,record['authority']).read_text())
            validate(authority,{'version','decision_payload','approved','reviewed_by'},'pipeline-study-authority/v1')
            if not authority['reviewed_by'] or authority.get('decision_payload')!=decision_payload(record) or authority.get('approved') is not True:
                missing.append('reviewed_decision')
        if record['cost'] is not None:
            cost=json.loads(reference(root,record['cost']).read_text())
            validate(cost,{'version','scope','within_budget','source','decision_payload','projected_seconds','budget_seconds'},'pipeline-study-cost/v1')
            number(cost['projected_seconds'],positive=True);number(cost['budget_seconds'],positive=True)
            if cost['source']!=source or cost['decision_payload']!=decision_payload(record) or cost['projected_seconds']>.8*cost['budget_seconds'] or cost['budget_seconds']!=record['budgets']['wall_seconds']:
                missing.append('exact_scope_cost')
            if cost.get('scope')!=('diverse32' if record['kind']=='learnability' else 'transfer-screen' if record['kind']=='representation' else record['kind']) or cost.get('within_budget') is not True:
                missing.append('exact_scope_cost')
        if record['kind']=='final':
            if final_plan is None or record['final_plan'] is None or reference(root,record['final_plan'])!=final_plan:
                missing.append('explicit_final_plan')
    if record['kind']=='final' and record['final_plan'] is None:missing.append('final_plan')
    return dict(status='BLOCKED' if missing else 'ADMITTED',missing=sorted(set(missing)),contract=record)


def decision_payload(record):
    """Review the prospective work; subsequent receipts cannot form a hash cycle."""
    return digest({k:v for k,v in record.items() if k not in {'identity','authority','state','gates','cost'}})


class StudyLedger:
    """Durable debit-before-dispatch; a new output directory cannot refund work."""
    def __init__(self, root, contract):
        from pathlib import Path
        from .identity import write
        from oeis_learn.fresh.common import lock
        self.root=Path(root)/digest(contract['decision_id']).split(':')[1]
        self.root.mkdir(parents=True,exist_ok=True)
        self.contract=contract
        with lock(self.root/'study.lock'):
            path=self.root/'contract.json'
            if path.exists():
                if json.loads(path.read_text()) != contract:
                    raise ValueError('reviewed study decision changed; debits cannot be reset')
            else:
                write(path,contract)

    def reservations(self, arm=None):
        values=[]
        for path in sorted(self.root.glob('*.reservation.json')):
            row=json.loads(path.read_text())
            if arm is None or row['arm']==arm:values.append(row)
        return values

    def begin_invocation(self, identity):
        from .identity import write
        from oeis_learn.fresh.common import lock
        with lock(self.root/'study.lock'):
            budgets=self.contract['budgets']
            overhead=budgets['cleanup_seconds']+budgets['archive_seconds']
            previous=sum(json.loads(p.read_text())['wall_seconds'] for p in self.root.glob('*.invocation.json'))
            if previous+overhead+sum(r['wall_seconds'] for r in self.reservations())>budgets['wall_seconds']:
                raise ValueError('study cleanup/archive budget exhausted; no refund')
            write(self.root/(identity+'.invocation.json'),dict(study=self.contract['identity'],wall_seconds=overhead))

    def reserve(self, arm):
        from .identity import write
        from oeis_learn.fresh.common import lock
        if arm not in self.contract['arms']:
            raise ValueError('unregistered study arm')
        with lock(self.root/'study.lock'):
            previous=self.reservations(arm['id'])
            attempt=len(previous)
            amounts=arm.get('attempt',{k:arm[k] for k in ('wall_seconds','work','disk_bytes')})
            if previous:
                if self.contract.get('kind')!='final' or 'attempt' not in arm:
                    raise ValueError('arm already debited; interrupted allowances are not refunded')
                terminal=self.root/(arm['id']+f'.{attempt-1:04d}.terminal.json')
                if not terminal.exists() or json.loads(terminal.read_text())['outcome']!='INTERRUPTED':
                    raise ValueError('only acknowledged interrupted final attempts may use remaining allowance')
            if any(sum(r[k] for r in previous)+amounts[k]>arm[k] for k in amounts):
                raise ValueError('cumulative arm allowance exhausted; no refund')
            budgets=self.contract.get('budgets')
            if budgets:
                all_rows=self.reservations()
                overhead=sum(json.loads(p.read_text())['wall_seconds'] for p in self.root.glob('*.invocation.json'))
                if not overhead:overhead=budgets['cleanup_seconds']+budgets['archive_seconds']
                if any(sum(r[k] for r in all_rows)+amounts[k]+(overhead if k=='wall_seconds' else 0)>budgets[k] for k in amounts):
                    raise ValueError('cumulative shared study allowance exhausted')
            row=dict(study=self.contract['identity'],arm=arm['id'],attempt=attempt,**amounts)
            write(self.root/(arm['id']+f'.{attempt:04d}.reservation.json'),row)
            return row

    def finish(self, arm, outcome, *, attempt=None):
        from .identity import write
        from oeis_learn.fresh.common import lock
        if outcome not in {'COMPLETE','NEGATIVE','INCONCLUSIVE','INTERRUPTED'}:
            raise ValueError('explicit terminal study outcome required')
        with lock(self.root/'study.lock'):
            rows=self.reservations(arm)
            if not rows:raise ValueError('study arm was not debited')
            ordinal=rows[-1]['attempt'] if attempt is None else attempt
            if ordinal not in {r['attempt'] for r in rows}:raise ValueError('attempt was not reserved')
            path=self.root/(arm+f'.{ordinal:04d}.terminal.json')
            if path.exists():
                raise ValueError('terminal study outcome is immutable')
            write(path,dict(study=self.contract['identity'],arm=arm,attempt=ordinal,outcome=outcome,
                charging='full original attempt reservation retained'))
