import json
import pytest
from tests.helpers.progressive_fixture import envelope
from oeis_learn.generalization.progressive.contracts import reference
from oeis_learn.generalization.progressive.cohorts import (
    census,
    publish,
    validate_cohort,
    DEFAULT_PROFILE,
)
from oeis_learn.pipeline_validation.identity import digest, seal


def fixture(root, role="confirmation", counts=(1, 1, 1, 1), panel="confirm-diversity"):
    exposure = envelope(
        "exposure", dict(previous=None, detector_limits=["finite relation checks only"], events=[])
    )
    rows = []
    for stratum, count in zip(("parameter", "composition", "unseen_structure", "oeis"), counts):
        for i in range(count):
            id = f"{panel}-{stratum}-{i}"
            indices = list(map(str, range(100)))
            values = [str(n + i) for n in range(100)]
            folder = root / id
            folder.mkdir(parents=True)
            (folder / "prompt.json").write_text(
                json.dumps(dict(id=id, indices=indices[:20], values=values[:20]))
            )
            (folder / "truth.json").write_text(
                json.dumps(dict(id=id, indices=indices, values=values))
            )
            rows.append(
                dict(
                    id=id,
                    source="synthetic",
                    offset="0",
                    stratum=stratum,
                    groups=["relation:" + id],
                    structure=digest(id),
                    behavior=digest(id + "behavior"),
                    composition=digest(id + "composition"),
                    operations=["add"],
                    difficulty=dict(depth=1, nodes=3, work=300),
                    visible=reference(root, folder / "prompt.json"),
                    truth=reference(root, folder / "truth.json"),
                )
            )
    cohort = envelope(
        "cohort",
        dict(
            role=role,
            panel=panel,
            decision="diversity",
            profile=dict(DEFAULT_PROFILE),
            required_per_stratum=128,
            minimum_groups=16,
            group_policy="exclusive",
            detector_limits=["finite relation checks only"],
            exposure=exposure["identity"],
            rows=rows,
            strata=["parameter", "composition", "unseen_structure", "oeis"],
        ),
    )
    return cohort, exposure


def test_count_availability_before_scoring_missing_strata_do_not_disappear(tmp_path):
    c, e = fixture(tmp_path, counts=(1, 1, 0, 1))
    r = census(c, exposure=e)
    assert r["status"] == "INCOMPLETE"
    assert r["strata"]["unseen_structure"] == dict(
        tasks=0, groups=0, task_deficit=128, group_deficit=16
    )
    assert r["strata"]["parameter"]["task_deficit"] == 127
    publish(tmp_path, "out", c, exposure=e)
    assert (tmp_path / "out/census.json").exists() and not (tmp_path / "out/public").exists()


def test_indexed_history_blocks_old_values_at_publication_and_freeze(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_inputs import ExposureIndex
    from oeis_learn.generalization.progressive.cohorts import validate_frozen_panels
    c,e=fixture(tmp_path)
    truth=json.loads((tmp_path/c['payload']['rows'][0]['truth']['path']).read_text())
    with ExposureIndex(tmp_path/'prior.sqlite') as index:
        index.add(dict(truth,id='prior-id'),origin='original-assessment',role='assessment')
    e=seal(dict(e,payload=dict(e['payload'],indexes=[reference(tmp_path,'prior.sqlite')])))
    c=seal(dict(c,payload=dict(c['payload'],exposure=e['identity'])))
    # Empty in-memory events do not erase the original private history.
    with pytest.raises(ValueError,match='indexed original exposure'):
        publish(tmp_path,'not-published',c,exposure=e)
    with pytest.raises(ValueError,match='indexed original exposure'):
        validate_frozen_panels(tmp_path,[c],{e['identity']:e})
    assert not (tmp_path/'not-published').exists()


def test_index_history_retains_parameter_structure_exception_and_hash_binding(tmp_path):
    from oeis_learn.pipeline_validation.progressive_research_inputs import ExposureIndex
    from oeis_learn.generalization.progressive.indexed_history import open_indexes,check_panel
    from oeis_learn.generalization.progressive.generator import structure_id
    c,e=fixture(tmp_path,counts=(1,0,0,0))
    source='(program (return (add $index (int 1))))'
    c['payload']['rows'][0]['structure']=structure_id(source)
    c['payload']['group_policy']='parameter_structure_sharing'
    with ExposureIndex(tmp_path/'prior.sqlite') as index:
        index.add(dict(id='prior',source=source,indices=list(map(str,range(100))),
                       values=[str(n+10000) for n in range(100)]),origin='original',role='training')
    e=seal(dict(e,payload=dict(e['payload'],indexes=[reference(tmp_path,'prior.sqlite')])))
    c=seal(dict(c,payload=dict(c['payload'],exposure=e['identity'])))
    with open_indexes(tmp_path,[e]) as indexes:
        check_panel(tmp_path,c,indexes)
        exclusive=seal(dict(c,payload=dict(c['payload'],group_policy='exclusive')))
        with pytest.raises(ValueError,match='indexed original exposure'):
            check_panel(tmp_path,exclusive,indexes)
    with (tmp_path/'prior.sqlite').open('ab') as f:f.write(b'changed')
    with pytest.raises(ValueError,match='identity'):
        with open_indexes(tmp_path,[e]):pass


def test_indexed_history_cannot_disappear_from_an_extension(tmp_path):
    from oeis_learn.generalization.progressive.exposure import append,validate_extension
    c,e=fixture(tmp_path)
    (tmp_path/'index').write_bytes(b'identity-only-fixture')
    e=seal(dict(e,payload=dict(e['payload'],indexes=[reference(tmp_path,'index')])))
    child=append(e,[],created_at=e['created_at'])
    assert child['payload']['indexes']==e['payload']['indexes']
    validate_extension(e,child)
    erased=dict(child['payload']);erased.pop('indexes')
    with pytest.raises(ValueError,match='erased'):
        validate_extension(e,seal(dict(child,payload=erased)))


def test_tiny_acceptance_census_publishes_separate_truth_and_visible(tmp_path):
    c, e = fixture(tmp_path)
    c = seal(dict(c, payload=dict(c["payload"], required_per_stratum=1, minimum_groups=1)))
    r = publish(tmp_path, "out", c, exposure=e)
    assert r["status"] == "COMPLETE"
    assert len(list((tmp_path / "out/public").glob("*.json"))) == 4
    public = [json.loads(p.read_text()) for p in (tmp_path / "out/public").glob("*.json")]
    assert all(len(p["data"]["values"]) == 20 for p in public)
    assert all(set(p["data"]) == {"id", "indices", "values"} for p in public)
    assert len(list((tmp_path / "out/private").glob("*.json"))) == 4


def test_confirmation_exposure_cross_panel_relation_and_source_offset(tmp_path):
    c, e = fixture(tmp_path)
    d, _ = fixture(tmp_path, panel="final")
    d["payload"]["role"] = "assessment"
    d["payload"]["rows"][0]["groups"] = c["payload"]["rows"][0]["groups"]
    d = seal(d)
    with pytest.raises(ValueError):
        validate_cohort(d, exposure=e, other_cohorts=[c])
    group = c["payload"]["rows"][0]["groups"][0]
    e["payload"]["events"] = [
        dict(
            group=group,
            relations=[],
            role="development",
            endpoint="parameter",
            source="old",
            access="truth",
            campaign="old",
        )
    ]
    e = seal(e)
    c["payload"]["exposure"] = e["identity"]
    c = seal(c)
    with pytest.raises(ValueError):
        validate_cohort(c, exposure=e)


def test_parameter_holdout_can_share_structure_only_by_declared_policy(tmp_path):
    a, e = fixture(tmp_path, panel="a")
    b, _ = fixture(tmp_path, panel="b")
    b["payload"]["rows"][0]["structure"] = a["payload"]["rows"][0]["structure"]
    b = seal(b)
    with pytest.raises(ValueError):
        validate_cohort(b, exposure=e, other_cohorts=[a])
    b = seal(dict(b, payload=dict(b["payload"], group_policy="parameter_structure_sharing")))
    validate_cohort(b, exposure=e, other_cohorts=[a])
