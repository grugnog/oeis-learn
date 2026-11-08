"""Authority is closed, prospective, bounded and distinct from historical PAC."""

import json
import pytest
from tests.helpers.progressive_fixture import blueprint, task, envelope, graph, caps
from oeis_learn.pipeline_validation.identity import seal, digest
from oeis_learn.generalization.progressive.contracts import validate, verify_refs, reference
from oeis_learn.generalization.progressive.campaign import validate_task, validate_runtime
from oeis_learn.generalization.progressive.graph import validate_graph


def test_blueprints_are_never_executable():
    assert validate("blueprint", blueprint())["payload"]["executable"] is False
    changed = blueprint()
    changed["payload"]["executable"] = True
    with pytest.raises(ValueError):
        validate("blueprint", seal(changed))


@pytest.mark.parametrize("change", ["unknown", "bool", "nan", "seal", "old", "track", "nested"])
def test_closed_envelopes(change):
    record = task()
    if change == "unknown":
        record["unplanned"] = "ignored"
    if change == "nested":
        record["payload"]["initializer"]["fallback"] = True
    if change == "bool":
        record["payload"]["seed"] = True
    if change == "nan":
        record["payload"]["caps"]["elapsed_us"] = float("nan")
    if change == "old":
        record["version"] = "PAC-screen/v1"
    if change == "track":
        record["track"] = "synthetic"
    if change == "seal":
        record["payload"]["seed"] = 2
    with pytest.raises(ValueError):
        validate("task", record if change in {"seal", "nan"} else seal(record))


def test_training_ceiling_and_initializer():
    validate_task(task())
    for changed in [
        task(caps=caps(elapsed_us=86_400_000_001)),
        task(startup_us=99_000_000),
        task(initializer=dict(mode="full_state", artifact=None, compatibility=digest("x"))),
    ]:
        with pytest.raises(ValueError):
            validate_task(changed)
    with pytest.raises(ValueError):
        validate_task(task(), track="oeis_derived")


def test_cyclic_unbounded_and_unfunded_graphs():
    assert validate_graph(graph(), caps=caps(**{k: 200 * v for k, v in caps().items()}))
    changed = graph()
    changed["nodes"][0]["dependencies"] = ["diversity"]
    with pytest.raises(ValueError):
        validate_graph(changed, caps=caps())
    changed = graph()
    changed["nodes"][0]["maximum_visits"] = 0
    with pytest.raises(ValueError):
        validate_graph(changed, caps=caps())
    changed = graph()
    changed["edges"][0]["target"] = "assessment"
    with pytest.raises(ValueError):
        validate_graph(changed, caps=caps())
    with pytest.raises(ValueError):
        validate_graph(graph(), caps=caps(tokens=1))


def test_references_reject_aliases_duplicates_and_symlinks(tmp_path):
    (tmp_path / "a").write_text("a")
    ref = reference(tmp_path, "a")
    assert verify_refs(tmp_path, [ref])
    for path in ["./a", "x/../a", "/a", "a//x"]:
        with pytest.raises(ValueError):
            verify_refs(tmp_path, [dict(ref, path=path)])
    with pytest.raises(ValueError):
        verify_refs(tmp_path, [ref, ref])
    (tmp_path / "link").symlink_to(tmp_path / "a")
    with pytest.raises(ValueError):
        reference(tmp_path, "link")
    (tmp_path / "hard").hardlink_to(tmp_path / "a")
    with pytest.raises(ValueError):
        verify_refs(tmp_path, [ref, reference(tmp_path, "hard")])


def test_forged_qualification_is_not_a_runtime(tmp_path):
    (tmp_path / "producer.json").write_text(json.dumps(seal(dict(status="PASS"))))
    runtime = envelope(
        "runtime",
        dict(
            scope=dict(
                source=digest("source"),
                images=[digest("cpu"), digest("gpu")],
                model=digest("shape"),
                language="bigint-body/v1",
                precision="fp32",
                device="hip",
                execution_policy=digest("numeric"),
            ),
            qualification=[reference(tmp_path, "producer.json")],
            operations=["train"],
            role_policy=digest("roles"),
            fixture=False,
        ),
    )
    with pytest.raises(ValueError):
        validate_runtime(runtime, root=tmp_path, current_facts={})
    runtime["purpose"] = "research"
    runtime["payload"]["fixture"] = True
    with pytest.raises(ValueError):
        validate_runtime(seal(runtime), root=tmp_path, current_facts={})
