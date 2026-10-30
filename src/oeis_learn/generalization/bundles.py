"""Offline reconciliation; private state and sealed truth are never shareable."""
import json
from .contracts import validate,verify_artifact,artifact_refs
from .metrics import summarize
from .interfaces import EXIT_CODES, ScientificStatus
from oeis_learn.pipeline_validation.identity import seal

CAP=512*1024**2


def bundle_plan(payloads,*,mandatory):
    public=[]; omissions=[]; total=0; seen=set()
    for p in payloads:
        if p['path'] in seen: raise ValueError('duplicate payload')
        seen.add(p['path'])
        if type(p['bytes']) is not int or p['bytes']<0: raise ValueError('invalid payload size')
        if p['private']:
            if p['required'] or p['path'] in mandatory: raise ValueError('mandatory private payload: replace with public identity evidence')
            omissions.append(p['path']); continue
        total+=p['bytes']; public.append(p['path'])
    if not set(mandatory)<=set(public): raise ValueError('missing mandatory public evidence')
    if total>CAP: raise ValueError('oversized mandatory bundle; explicit partial packaging required')
    return dict(payloads=public,omissions=omissions,bytes=total)


def reconcile_ledger(ledger,*,study):
    if set(ledger)!={'study','required','receipts'} or ledger['study']!=study or not ledger['required'] or len(set(ledger['required']))!=len(ledger['required']):
        raise ValueError('conflicting study or mandatory stage roster')
    by_stage={}
    for r in ledger['receipts']:
        if set(r)!={'stage','status','study','original'} or r['study']!=study or r['stage'] in by_stage or r['stage'] not in ledger['required']:
            raise ValueError('conflicting stage receipt')
        if r['status'] not in EXIT_CODES: raise ValueError('unverified REUSED or unknown status')
        by_stage[r['stage']]=r['status']
    statuses=set(by_stage.values())
    for status in ('FAIL','INTERRUPTED','BLOCKED','INCONCLUSIVE'):
        if status in statuses: return status
    return 'PASS' if set(by_stage)==set(ledger['required']) and statuses=={'PASS'} else 'PARTIAL'


def verify_bundle(record,*,expected_study,root):
    q=validate('evidence',record).record()
    inventory=q['inventory']
    if len({p['path'] for p in inventory})!=len(inventory) or sum(p['bytes'] for p in inventory)>CAP: raise ValueError('invalid bundle inventory')
    indexed={p['path']:p for p in inventory}
    for ref in artifact_refs(q):
        if indexed.get(ref['path'])!=ref: raise ValueError('mandatory artifact absent from inventory')
        verify_artifact(root,ref,max_bytes=CAP)
    def read(ref): return json.loads(verify_artifact(root,ref,max_bytes=CAP))
    study=read(q['study'])
    validate('study',study)
    if study['identity']!=expected_study: raise ValueError('conflicting study identity')
    # Original qualification manifests include all offline evidence they reference.
    from .qualification import consume, FACTS
    for ref in q['qualification_producers']:
        qual=read(ref)
        for child in artifact_refs(qual):
            if indexed.get(child['path'])!=child: raise ValueError('missing offline original evidence')
        consume(qual,{k:qual[k] for k in FACTS},root=root,required_suites=qual['suite_selection'],required_capabilities=qual['covered_capabilities'])
    ledger=read(q['stage_ledger']); status=reconcile_ledger(ledger,study=expected_study)
    if q['software_status']!=status: raise ValueError('stale software status')
    if q['scientific_status'] not in {s.value for s in ScientificStatus}: raise ValueError('scientific status')
    measurements=read(q['attempts']); stored=read(q['summary'])
    if set(measurements)!={'roster','outcomes'}: raise ValueError('fixed denominator measurements required')
    summary=summarize(measurements['roster'],measurements['outcomes'])
    actual=dict(n=summary.n,successes=summary.successes,unresolved=summary.unresolved,
                lower=str(summary.lower),upper=str(summary.upper),completion=str(summary.completion),
                macro_lower=str(summary.macro_lower),macro_upper=str(summary.macro_upper))
    if stored!=actual: raise ValueError('stale summary')
    return dict(integrity='PASS',software_status=status,scientific_status=q['scientific_status'],summary=actual,omissions=q['omissions'])
