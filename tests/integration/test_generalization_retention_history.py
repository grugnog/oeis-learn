"""Whole-run retention counterexamples."""

def test_long_retention_history_keeps_endpoints_exact_family_totals_and_resume():
    from types import SimpleNamespace
    from fractions import Fraction
    from oeis_learn.pipeline_validation.generalization_diagnostics import PacObservers
    current={'update':0}
    families=['f0','f1'];probes={f:[f+'-'+str(i) for i in range(8)] for f in families}
    rows={key:dict(id=key,family=f,tokens=[3,2]) for f,ids in probes.items() for key in ids}
    batch=SimpleNamespace(families={f:[f] for f in families},anchors={'f0':'f0','f1':'f1'},inventory={'identity':'fixture'},probes=probes,probe_rows=rows,probe_truth={k:{} for k in rows})
    def observe(selected):
        if current['update']==4096 and selected[0]['family']=='f0':return None
        return dict(loss_sum=float(current['update']+1+families.index(selected[0]['family'])),tokens=8,recall=[False]*8)
    def semantic(row,truth,**kwargs):return None if row['family']=='f0' and current['update']%512==0 else row['family']=='f1'
    def observer(snapshot=None):return PacObservers(batch,arm='C',seed=7,observe=observe,semantic=semantic,snapshot=snapshot,coordinate_clock=lambda:current['update']/2)
    full=observer();split=observer()
    for u in range(32769):
        current['update']=u
        if u:
            row=dict(id='gradient',family='f0',tokens=[3,2])
            for o in (full,split):o.committed([row],channel='bandit')
        for o in (full,split):
            o.common(u)
            if u%256==0:
                for j in range(70):o.event(dict(kind='recent_only',update=u))
        if u==8192:split=observer(split.snapshot())
    a=full.snapshot();assert a==split.snapshot()
    h=a['history'];assert len(a['recent'])==64 and len(h['bins'])<=64
    assert h['bins'][0]['first']['coordinates']['update']==0
    assert h['bins'][-1]['last']['coordinates']==dict(update=32768,tokens=65536,examples=32768,seconds=16384.)
    assert sum(b['count'] for b in h['bins'])==129
    totals={f:{k:sum((Fraction(b['totals'][f][k]) for b in h['bins']),Fraction()) for k in h['bins'][0]['totals'][f]} for f in families}
    assert totals['f0']['loss_censored']==1 and totals['f0']['loss_completed']==128
    assert totals['f0']['loss_sum']==sum(u+1 for u in range(0,32769,256) if u!=4096)
    assert totals['f0']['semantic_attempts']==129*8 and totals['f0']['semantic_completed']==64*8 and totals['f0']['semantic_success']==0
    assert totals['f1']['semantic_completed']==totals['f1']['semantic_success']==129*8


def test_history_restore_rejects_lost_coverage_and_declares_legacy_gap():
    import copy
    import pytest
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation.generalization_diagnostics import PacObservers
    b=SimpleNamespace(families={'f':['x']},anchors={'x':'f'},inventory={'identity':'i'},probes={'f':['x']},probe_rows={'x':{}},probe_truth={'x':{}})
    make=lambda snapshot=None:PacObservers(b,arm='B',seed=1,observe=lambda rows:None,snapshot=snapshot)
    o=make();o.common(0);s=o.snapshot()
    bad=copy.deepcopy(s);bad['history']['bins']=[]
    with pytest.raises(ValueError,match='coverage'):make(bad)
    legacy=copy.deepcopy(s);legacy.pop('history');legacy['version']='pac-observers/v1'
    restored=make(legacy);restored.common(256);h=restored.snapshot()['history']
    assert h['omitted_legacy_panels']==1 and h['bins'][0]['first']['coordinates']['update']==256
    assert h['bins'][0]['totals']['f']['loss_censored']==1
