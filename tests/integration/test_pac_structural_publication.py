"""Receipt-publication regression; tiny inputs isolate the hash-binding boundary.

Expensive scope admission is stubbed here, not claimed as native/quota evidence.
The existing preparation publisher, paired-target builder and file hashes are real.
"""
import json
from pathlib import Path


def test_structural_truth_is_in_admission_artifact_closure(tmp_path,monkeypatch):
    from tests.unit.test_generalization_execution import pac_setup
    from tests.integration.test_generalization_recipes import pac_envelope
    from tests.integration.test_generalization_preparation import pac_rows
    from oeis_learn.pipeline_validation import generalization_preparation as preparation
    from oeis_learn.pipeline_validation.generalization_execution import bind_pac,pac_budget_projection,POLICY_ID
    from oeis_learn.pipeline_validation.identity import seal,digest,write
    from oeis_learn.generalization.contracts import artifact_refs
    from oeis_learn.generalization import structural
    from oeis_learn.generalization import preparation as policy
    from oeis_learn.synthesis.reference import evaluate
    rows,probes=pac_rows()
    def put(name,value):
        write(tmp_path/name,value);return preparation.reference(tmp_path/name,tmp_path)
    original=put('static.json',[dict(r,group=digest(r['id']),variant=r['id'],values=[str(i+2) for i in range(100)]) for r in rows])
    gradient=put('gradient.json',rows)
    plan=dict(version='pac-preparation/v1',originals=[dict(name='static',role='training',holdout='parameter',count=len(rows),ref=original)],
        rows=gradient,probes=put('probes.json',probes),protocol={'cells':[]},seed=7,batch_size=2)
    exp,control,profiles,_=pac_setup()
    exp=seal(dict(exp,backend=POLICY_ID,budget_graph=pac_budget_projection(pac_envelope()),seed=7))
    runtime=dict(version='pac-runtime/v1',study='',purpose='research',recipe=put('recipe.json',{}),source=put('source.json',{}),
        images=[digest('cpu'),digest('gpu')],control=control,profiles=profiles,preparation=plan,prepared=None,
        envelope=pac_envelope(),feature=None,engine='podman')
    exp,runtime=bind_pac(exp,runtime)
    candidate=dict(rows[0],id='structural-fixture',group=digest('structural-group'),variant='structural-variant')
    values=[str(i+2) for i in range(100)]
    monkeypatch.setattr(preparation,'validate_pac_preparation',lambda *a:plan)
    monkeypatch.setattr(structural,'draw',lambda *a:[candidate])
    monkeypatch.setattr(policy,'reference_candidate',lambda row:((dict(row,values=values),None),None))
    monkeypatch.setattr(structural,'admit',lambda *a,**kw:{digest(['selection',candidate['id'],candidate['source']]):{'family':candidate['family'],'category':'certified'}})
    build=preparation.prepare_pac_training
    monkeypatch.setattr(preparation,'prepare_pac_training',lambda *a,**kw:build(*a,**dict(kw,purpose='acceptance')))
    def execute(source,indices):
        value=evaluate(source,indices);return dict(outcome=value.outcome,values=list(value.values))
    output=tmp_path/'published'
    preparation.prepare_pac(tmp_path,exp,runtime,output=output,execute=execute,seconds=300)
    truth=preparation.reference(output/'private/structural-targets.json',output)
    admission=json.loads((output/'admission.json').read_text())
    assert truth in list(artifact_refs(admission)), 'structural truth bytes must be hash-bound before inference publication'
    preparation.verify_pac_structural(output,admission)
    import pytest
    path=output/truth['path'];path.write_text('[]')
    with pytest.raises(ValueError):preparation.verify_pac_structural(output,admission)
