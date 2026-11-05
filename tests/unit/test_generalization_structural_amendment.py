"""Independent frozen-admission failure and conservative source-capacity cases."""
import json
from collections import Counter,defaultdict
from pathlib import Path
import pytest
from oeis_learn.generalization import structural

FIXTURE=Path(__file__).resolve().parents[1]/'fixtures/generalization/pac-failed-admission.json'


def worst_survivors(rows,available):
    groups=defaultdict(list)
    for row in rows:groups[row['source']].append(row)
    ordered=sorted(groups.values(),key=lambda rs:(-len(rs),rs[0]['source']))
    return [r for group in ordered for r in group][:available]


def test_amendment_preserves_draws_and_exclusions_but_declares_new_population():
    old=structural.design(structural.prior_protocol())
    new=structural.amended_design(structural.prior_protocol())
    assert new['version']=='pac-structural-scope/v2' and new['parent_scope']==old['identity']
    assert structural.draw(new)==structural.draw(old)
    assert len(structural.draw(new))==2288
    names={c['name'] for c in structural.original_cells()}
    assert names=={'static',*[f'refresh-{i:02}' for i in range(1,12)],*[f'{r}-{h}' for r in ('selection','assessment') for h in ('parameter','structural')]}
    assert len(names)==16
    assert new['numeric_law']==old['numeric_law'] and new['inference']==old['inference'] and new['controls']==old['controls']
    assert [c['draws'] for c in new['cells']]==[c['draws'] for c in old['cells']]
    assert sum(c['count']>0 for c in new['cells'])==23
    category_tasks=Counter();category_families=Counter()
    for c in new['cells']:
        category=new['families'][c['family']]['category'];category_tasks[category]+=c['count']
        if c['count']:category_families[category]+=1
    assert category_tasks==dict(certified=68,ambiguous=26,unresolved=34)
    assert category_families==dict(certified=15,ambiguous=3,unresolved=5)


def test_reported_survivors_guarantee_new_quotas_even_under_worst_source_duplicates():
    fixture=json.loads(FIXTURE.read_text());old=structural.design(structural.prior_protocol())
    new=structural.amended_design(structural.prior_protocol());rows=structural.draw(old)
    kept=[]
    for cell in fixture['availability']['families']:
        group=[r for r in rows if r['family']==cell['family']]
        kept.extend(worst_survivors(group,cell['available']))
    rejected={r['id'] for r in rows}-{r['id'] for r in kept}
    results={r['id']:dict(reference=['1']*100,native=['1']*100) for r in kept}
    with pytest.raises(structural.QuotaDeficit):
        structural.admit(old,rows,rows,results,exclusion_cells=structural.original_cells(),rejected_ids=rejected)
    roster=structural.admit(new,rows,rows,results,exclusion_cells=structural.original_cells(),rejected_ids=rejected)
    assert len(roster)==128 and len({r['family'] for r in roster.values()})==23
    from oeis_learn.generalization.metrics import pac_structural_metrics
    scope=structural.scope(new)
    metrics=pac_structural_metrics(roster,dict.fromkeys(roster,False),scope=scope)
    assert metrics['certified']['families']==15 and metrics['certified']['tasks']==68
    with pytest.raises(ValueError):pac_structural_metrics(roster,{})
    with pytest.raises(ValueError):pac_structural_metrics(roster,{},scope=dict(scope,design='sha256:'+'0'*64))
    # A changed quota is a changed scientific scope, never an implicit retry.
    changed=json.loads(json.dumps(new));changed['cells'][0]['count']+=1
    from oeis_learn.pipeline_validation.identity import seal
    with pytest.raises(ValueError):structural.verify(seal(changed))
