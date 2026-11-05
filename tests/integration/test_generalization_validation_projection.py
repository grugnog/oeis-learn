"""Actual custodian projection includes the sealed structural truth dependency."""
import json
from pathlib import Path
import pytest
from oeis_learn.pipeline_validation.identity import seal,write,digest
from oeis_learn.pipeline_validation.generalization_preparation import reference,verify_pac_structural


def inputs(root,version):
    from tests.unit.test_generalization_structural_amendment import amended_roster
    from oeis_learn.generalization import structural
    root.mkdir()
    rows=[];roster={}
    for i,metadata in enumerate(amended_roster().values()):
        row=dict(id=str(i),source=f'(program (return (int {i})))',values=[str(i)]*100,
            indices=list(map(str,range(100))),family=metadata['family'],role='selection',holdout='structural')
        rows.append(row);roster[digest(['selection',row['id'],row['source']])]=metadata
    write(root/'private/structural-targets.json',rows)
    truth=reference(root/'private/structural-targets.json',root)
    admission=seal(dict(version=version,status='PASS',structural=truth,roster=roster,
        **(dict(structural_scope=structural.scope(structural.amended_design(structural.prior_protocol()))) if version=='pac-admission/v3' else {}),
        # Original-cell provenance is metadata here; validation must not chase
        # every reference in an already-verified producer into unrelated roots.
        originals=[dict(ref=dict(path='unprojected-original.json',bytes=1,sha256=digest('original')))]))
    write(root/'admission.json',admission)
    write(root/'training.json',[])
    training=reference(root/'training.json',root)
    bundle=seal(dict(version='pac-prepared/v1',role='training',literal=[training],parameterized=[training],
        probes=training,inventory=training,pairs=training))
    write(root/'bundle.json',bundle);write(root/'recipe.json',{});write(root/'source.json',{})
    from tests.integration.test_generalization_recipes import pac_envelope
    runtime=dict(version='pac-runtime/v1',prepared=dict(training=reference(root/'bundle.json',root),admission=reference(root/'admission.json',root)),
        preparation=None,recipe=reference(root/'recipe.json',root),source=reference(root/'source.json',root),
        images=[digest('cpu'),digest('gpu')],envelope=pac_envelope())
    return runtime,bundle,admission


@pytest.mark.parametrize('version',['pac-admission/v2','pac-admission/v3'])
def test_canonical_validate_reads_structural_truth_after_real_projection(tmp_path,monkeypatch,version):
    from oeis_learn.pipeline_validation import generalization_commands as commands,generalization_roles as roles
    from oeis_learn.generalization.role_contracts import Role,pac_training_refs
    from oeis_learn.generalization.contracts import verify_artifact
    base=tmp_path/'prepared';runtime,bundle,admission=inputs(base,version);called=[]
    def invoke(host,image,root,role,channels,**kwargs):
        assert Role(role)==Role.PREPARATION and kwargs['operation']=='validate-runtime'
        projected=root/channels['mechanical_sources']
        actual=json.loads(verify_artifact(projected,runtime['prepared']['admission']))
        assert len(verify_pac_structural(projected,actual))==128
        assert not (projected/'unprojected-original.json').exists()
        assert not list((root/channels['isolation_truth']).iterdir())
        called.append(True);return dict(status='PASS',plan={'closure':'verified'})
    monkeypatch.setattr(roles,'invoke',invoke)
    assert commands._validate(None,tmp_path,tmp_path/'qualify',base,{},runtime,seconds=900)=={'closure':'verified'}
    assert called and (base/'private/structural-targets.json').exists()
    assert not (tmp_path/'qualify/validation/private/input').exists()
    for arm in 'ABCD':
        assert admission['structural'] not in pac_training_refs(bundle,arm)


def test_missing_and_tampered_structural_input_reject_before_role_launch(tmp_path,monkeypatch):
    from oeis_learn.pipeline_validation import generalization_commands as commands,generalization_roles as roles
    runtime,_,_=inputs(tmp_path/'prepared','pac-admission/v3')
    path=tmp_path/'prepared/private/structural-targets.json';original=path.read_bytes()
    monkeypatch.setattr(roles,'invoke',lambda *a,**k:pytest.fail('role launched despite broken truth reference'))
    path.write_text('[]')
    with pytest.raises(ValueError):commands._validate(None,tmp_path,tmp_path/'changed',tmp_path/'prepared',{},runtime,seconds=900)
    path.unlink()
    with pytest.raises(ValueError):commands._validate(None,tmp_path,tmp_path/'missing',tmp_path/'prepared',{},runtime,seconds=900)
