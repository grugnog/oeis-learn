"""Original data reuse never transfers a historical HIP or campaign authority."""
import pytest
from oeis_learn.generalization.progressive.contracts import reference,seal,digest
from oeis_learn.pipeline_validation.progressive_workstation_state import write
from oeis_learn.pipeline_validation.progressive_preparation_reuse import adopt


@pytest.fixture
def donor(tmp_path,monkeypatch):
    folder=tmp_path/'run-state/discovery/original';base=folder/'prepare-research'
    source={'src/oeis_learn/pipeline_validation/progressive_research_census.py':digest('unchanged')}
    write(base/'qualification/source.json',source)
    write(base/'qualification/qualification.json',dict(test_boundary=True))
    write(folder/'session.json',seal(dict(binding=dict(source=digest(source)))))
    (folder/'session.lock').touch()
    data=seal(dict(version='progressive-registered-data/v1',design=seal(dict(seeds=[17,19]))))
    values={'registered-data':('registered/registered-data.json',data),
        'native-census':('census/census.json',seal(dict(diagnostic=False,training_started=False,assessment_exposed=False))),
        'materials':('materials/materials.json',seal(dict(status='COMPLETE',diagnostic=False))),
        'sensitivity':('sensitivity/sensitivity.json',seal(dict(status='COMPLETE')))}
    steps={}
    for name,(rel,value) in values.items():
        write(base/rel,value);steps['prepare-research/'+name]=dict(status='COMPLETE',outputs=[reference(tmp_path,base/rel)])
    write(folder/'journal.json',seal(dict(steps=steps)))
    # Named original-Q and native-material test boundaries. No hardware claimed.
    from oeis_learn.pipeline_validation import progressive_handoff,progressive_research_bindings
    from oeis_learn.experiments import source_snapshot
    monkeypatch.setattr(progressive_handoff,'original_authority',lambda *a:(dict(source_closure=reference(base/'qualification','source.json')),base/'qualification'))
    monkeypatch.setattr(source_snapshot,'collect_snapshot',lambda _:source.copy())
    monkeypatch.setattr(progressive_research_bindings,'inputs',lambda root,d,design,m:None)
    return tmp_path,folder,source,steps


def test_completed_data_reuses_original_refs_with_single_recipient(donor):
    root,folder,source,steps=donor;recipient=root/'run-state/discovery/new'
    originals={p:p.read_bytes() for p in folder.rglob('*') if p.is_file()}
    values,outputs,receipt=adopt(root,str(folder.relative_to(root)),recipient)
    assert receipt['hardware_authority_transferred'] is False and receipt['original_costs_retained']
    assert all(p.read_bytes()==raw for p,raw in originals.items())
    assert adopt(root,str(folder.relative_to(root)),recipient)[2]==receipt
    with pytest.raises(ValueError):adopt(root,str(folder.relative_to(root)),root/'run-state/discovery/sibling')


@pytest.mark.parametrize('change',['source','run','construction','incomplete','tamper'])
def test_changed_or_scored_donors_cannot_be_reused(donor,change):
    root,folder,source,steps=donor
    if change=='source':source['src/oeis_learn/pipeline_validation/progressive_research_census.py']=digest('changed')
    if change=='run':steps['run-research/invocation-0000']=dict(status='COMPLETE',outputs=[])
    if change=='construction':(folder/'prepare-research/construction').mkdir()
    if change=='incomplete':steps['prepare-research/native-census']['status']='BLOCKED'
    if change=='tamper':write(folder/'prepare-research/materials/materials.json',seal(dict(status='changed')))
    write(folder/'journal.json',seal(dict(steps=steps)))
    with pytest.raises(ValueError):adopt(root,str(folder.relative_to(root)),root/'run-state/discovery/new')
    assert not (root/'run-state/discovery-preparation-revisions').exists()
