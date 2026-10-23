"""One archive for separately admitted, independently bounded study decisions."""
import copy
from contextlib import ExitStack
from dataclasses import replace
import json
import uuid
from .identity import validate,reference,number,digest
from .studies import parse_study
from .registry import Stage
from .runner import execute

FIELDS={'version','state','studies','authority','budgets'}


def parse_batch(record,*,root,source,final_plan=None):
    validate(record,FIELDS,'pipeline-study-batch/v1')
    if record['state'] not in {'DRAFT','FROZEN','ADMITTED'}:
        raise ValueError('batch manifest is prospective authority, not a scientific outcome')
    refs=record['studies']
    if not isinstance(refs,list) or not 1<=len(refs)<=16:
        raise ValueError('one to sixteen independent study references required')
    budgets=record['budgets']
    if set(budgets)!={'wall_seconds','work','disk_bytes','cleanup_seconds','archive_seconds'}:
        raise ValueError('finite shared batch budgets required')
    for value in budgets.values():number(value,positive=True)
    totals=dict(wall_seconds=budgets['cleanup_seconds']+budgets['archive_seconds'],work=0,disk_bytes=0)
    studies=[];ids=set()
    for ref in refs:
        path=reference(root,ref)
        study=parse_study(json.loads(path.read_text()),root=root,source=source,final_plan=final_plan)
        contract=study['contract']
        if contract['decision_id'] in ids:raise ValueError('duplicate independent decision in batch')
        ids.add(contract['decision_id'])
        for key in totals:totals[key]+=contract['budgets'][key]
        studies.append(dict(study,reference=ref))
    if any(totals[key]>budgets[key] for key in totals):
        raise ValueError('study reservations exceed shared batch limits')
    missing=[]
    if record['state']!='ADMITTED':missing.append('explicit_batch_admission')
    if record['authority'] is None:missing.append('reviewed_batch_authority')
    else:
        authority=json.loads(reference(root,record['authority']).read_text())
        validate(authority,{'version','decision_payload','approved','reviewed_by'},'pipeline-study-authority/v1')
        payload=digest({k:v for k,v in record.items() if k not in {'identity','authority','state'}})
        if authority['approved'] is not True or not authority['reviewed_by'] or authority['decision_payload']!=payload:
            missing.append('reviewed_batch_authority')
    return dict(status='BLOCKED' if missing else 'ADMITTED',missing=missing,contract=record,studies=studies)


def main(args,root,host_factory,record,source,final_plan):
    decision=parse_batch(record,root=root,source=source,final_plan=final_plan)
    if args.scope_report:
        print(json.dumps(decision,indent=2));return 0
    from .identity import confined
    output=confined(root,args.output,exists=False)
    maximum=record['budgets']['wall_seconds']
    if args.max_seconds is not None:
        if not 0<number(args.max_seconds)<=maximum:raise ValueError('batch ceiling may only be lowered')
        maximum=args.max_seconds
    stages=[];dispatch={};plans=[]
    if decision['status']!='ADMITTED':
        stages=[Stage('batch.admission',6,available=False,reason=', '.join(decision['missing']))]
    else:
        from .cli import study_main
        for index,study in enumerate(decision['studies']):
            options=copy.copy(args);options.study_manifest=study['reference']['path']
            options.max_seconds=None;options.scope_report=False
            plan=study_main(options,root,host_factory,collect=True)
            prefix=f's{index:02d}.'
            plans.append((prefix,plan))
            for stage in plan['stages']:
                wrapped=replace(stage,id=prefix+stage.id,prerequisites=tuple(prefix+p for p in stage.prerequisites))
                stages.append(wrapped);dispatch[wrapped.id]=(plan['adapter'],stage)
    def run(stage,destination):
        adapter,original=dispatch[stage.id]
        return adapter(original,destination)
    from oeis_learn.fresh.common import lock
    with ExitStack() as stack:
        ledgers={str(plan['ledger'].root):plan['ledger'] for _,plan in plans if plan.get('ledger') is not None}
        for name in sorted(ledgers):stack.enter_context(lock(ledgers[name].root/'controller.lock'))
        if output.exists() or output.with_suffix('.zip').exists():raise ValueError('fresh batch output required')
        blocked={}
        for prefix,plan in plans:
            ledger=plan.get('ledger')
            if ledger is not None:
                try:ledger.begin_invocation(uuid.uuid4().hex)
                except ValueError as exc:blocked[prefix]=str(exc)
        stages=[replace(s,available=False,reason=next(reason for prefix,reason in blocked.items() if s.id.startswith(prefix)))
            if any(s.id.startswith(prefix) for prefix in blocked) else s for s in stages]
        return execute(stages,output,adapter=run,binding={'source':source,'batch':record['identity']},
            cap=maximum,reserve=record['budgets']['cleanup_seconds']+record['budgets']['archive_seconds'],contract=record)
