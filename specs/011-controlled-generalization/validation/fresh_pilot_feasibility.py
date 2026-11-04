"""Temporary stdlib source feasibility; never issues native/HIP qualification."""
import json
import time
from collections import Counter
from pathlib import Path
from oeis_learn.generalization.preparation import scan,exclusion_record
from oeis_learn.generalization.exclusions import ExclusionIndex
from oeis_learn.generalization.refresh import RefreshPlan
from oeis_learn.generalization.budgets import charge_tokens
from oeis_learn.decoder.bigint_codec import encode_body
from oeis_learn.pipeline_validation.identity import seal,write,digest


def selection_identity(rows):
    return digest([row['id'] for row in rows])


def check(design,output,*,workers=4,seconds=21600):
    if seal(design)!=design:raise ValueError('intact frozen design required')
    root=Path(output)
    if root.exists():raise ValueError('fresh feasibility root required')
    root.mkdir(parents=True)
    write(root/'private/design.json',design)
    started=time.monotonic();exclusions=ExclusionIndex();seen=set();cells=[];tokens=0
    report=dict(version='fresh-pilot-feasibility/v1',protocol=design['identity'],status='BLOCKED',
        native_validation='NOT_RUN',hardware_validation='NOT_RUN',cells=cells,training_tokens=0)
    try:
        for cell in [*design['cohorts'],*design['training']]:
            began=time.monotonic()
            remaining=min(21600,seconds-(began-started))
            if remaining<=0:raise TimeoutError('full feasibility deadline exhausted')
            if 'refresh' in cell:
                r=cell['refresh'];rows,_=RefreshPlan(design['descriptors'],seed=r['seed'],capacity=r['capacity']).take(r['cursor'],r['draws'])
            else:rows=cell['rows']
            def progress(draws,admitted):
                print(f"feasibility {cell['name']}: draws={draws} admitted={admitted}",flush=True)
            choice,ledger=scan(rows,**{k:cell[k] for k in ('role','holdout','count','quotas','structural')},
                execute=None,exclusions=exclusions,seconds=min(21600,seconds-(time.monotonic()-started)),
                unique_sources='refresh' in cell,seen_sources=seen,workers=workers,
                length_boundary=cell.get('length_boundary',32),progress=progress)
            write(root/'private/ledgers'/(cell['name']+'.json'),ledger)
            selected=choice.rows
            counts=Counter(tag for row in selected for tag in row['tags'])
            current=dict(name=cell['name'],status=ledger['status'],draws=ledger['draws'],admitted=ledger['admitted'],
                selected=len(selected),selection=selection_identity(selected),shortfalls=ledger['shortfalls'],
                rejection_counts=dict(Counter(r['reason'] for r in ledger['rejections'])),
                rule_counts=dict(Counter(rule for r in ledger['rejections'] for rule in r.get('rules',[]))),
                coverage=dict(counts),elapsed_seconds=time.monotonic()-began)
            cells.append(current)
            if cell['role']=='training':
                seen.update(digest(r['source']) for r in selected)
                current['tokens']=charge_tokens([encode_body(r['source']) for r in selected],bos=1,pad=0)
                tokens+=current['tokens']
            else:exclusions=exclusions.add_many(exclusion_record(row,cell['role'],cell['holdout']) for row in selected)
            report['training_tokens']=tokens
            write(root/'progress.json',seal(report),exclusive=False)
            print(f"feasibility {cell['name']}: {ledger['status']} selected={len(selected)} seconds={current['elapsed_seconds']:.1f}",flush=True)
            if not choice.complete:
                report['reason']='QUOTA_SHORTFALL';break
        else:
            if tokens<design['target_tokens']:report['reason']='TOKEN_CAPACITY_SHORTFALL'
            else:report['status']='FEASIBLE'
    except (Exception,KeyboardInterrupt) as exc:
        report['reason']=type(exc).__name__
        report['status']='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'BLOCKED'
    report['elapsed_seconds']=time.monotonic()-started
    write(root/'result.json',seal(report))
    return seal(report)


def verify_prepared(feasibility,prepared):
    """Reference screen is a gate, never a replacement for native preparation."""
    if seal(feasibility)!=feasibility or feasibility['status']!='FEASIBLE':raise ValueError('full source feasibility required')
    for cell in feasibility['cells']:
        path=Path(prepared)/'cells'/cell['name']/'private/selected.json'
        if path.is_symlink() or path.stat().st_size>256<<20:raise ValueError('bounded native-selected rows required')
        rows=json.loads(path.read_bytes())
        if selection_identity(rows)!=cell['selection']:raise ValueError('native preparation differs from reference feasibility')
