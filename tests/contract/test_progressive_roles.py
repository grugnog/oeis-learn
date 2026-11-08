"""Real files and hard links cannot cross private truth boundaries."""

import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write
from oeis_learn.generalization.progressive.projections import describe, project, remove_projection
from oeis_learn.pipeline_validation.identity import seal
from oeis_learn.pipeline_validation.progressive_roles import REGISTRY


def artifact(root, name, kind, role, data):
    atomic_write(
        root / name,
        seal({"version": "discovery-channel/v1", "kind": kind, "role": role, "data": data}),
    )
    return describe(root, name)


def test_every_execution_contract_row_has_closed_registry():
    assert {s.role for s in REGISTRY.values()} == {
        "custodian",
        "learner",
        "generator",
        "admission",
        "scorer",
        "assessment",
        "proposal",
        "component",
        "review",
        "verifier",
    }
    assert set(REGISTRY["generate"].channels) == {"visible", "model"}
    assert REGISTRY["train"].channels == ("training",)
    assert REGISTRY["propose-trial"].channels == ("search",)
    with pytest.raises(TypeError):
        REGISTRY["execute-python"] = None


def test_generator_projection_contains_visible_only(tmp_path):
    p = artifact(
        tmp_path,
        "prompt.json",
        "visible_prompt",
        "generator",
        {"id": "x", "indices": [str(i) for i in range(20)], "values": [str(i) for i in range(20)]},
    )
    (tmp_path / "weights.pt").write_bytes(b"actual fixture weights")
    from oeis_learn.generalization.progressive.contracts import reference

    w = artifact(
        tmp_path,
        "weights.json",
        "checkpoint",
        "generator",
        {
            "artifact": reference(tmp_path, "weights.pt"),
            "compatibility": "sha256:" + "b" * 64,
            "profile": "bigint-body/v1",
        },
    )
    truth = artifact(
        tmp_path, "truth.json", "confirmation_truth", "scorer", {"secret": "123456789"}
    )
    views = project(
        tmp_path,
        "projection",
        "generate",
        {"visible": [p], "model": [w]},
        registry=REGISTRY,
        forbidden=[truth.ref],
    )
    assert len(list(views["visible"].iterdir())) == 1
    assert (views["model"] / "weights.pt").read_bytes() == b"actual fixture weights"
    assert not any("123456789" in f.read_text() for f in (tmp_path / "projection").rglob("*.json"))
    remove_projection(tmp_path / "projection")
    with pytest.raises(ValueError):
        project(tmp_path, "bad", "generate", {"visible": [truth], "model": [w]}, registry=REGISTRY)
    with pytest.raises(ValueError):
        project(tmp_path, "alias", "generate", {"visible": [p], "model": [p]}, registry=REGISTRY)


@pytest.mark.parametrize(
    "operation,channel,kind,role",
    [
        ("train", "training", "constructive_witness", "learner"),
        ("train", "training", "training_continuation", "learner"),
        ("generate", "visible", "assessment_truth", "generator"),
        ("admit-training", "admission_truth", "development_truth", "admission"),
        ("propose-trial", "search", "confirmation_truth", "proposal"),
    ],
)
def test_wrong_types_rejected_even_with_valid_hash(tmp_path, operation, channel, kind, role):
    a = artifact(tmp_path, "a.json", kind, role, {"values": ["987654321"]})
    inputs = {c: [] for c in REGISTRY[operation].channels}
    inputs[channel] = [a]
    with pytest.raises(ValueError):
        project(tmp_path, "out", operation, inputs, registry=REGISTRY)


def test_visible_envelope_cannot_smuggle_witness(tmp_path):
    a = artifact(
        tmp_path,
        "a.json",
        "visible_prompt",
        "generator",
        {"id": "x", "indices": ["0"], "values": ["0"], "witness": "private"},
    )
    with pytest.raises(ValueError):
        project(tmp_path, "out", "generate", {"visible": [a], "model": []}, registry=REGISTRY)


def test_nested_model_payload_cannot_alias_forbidden_truth(tmp_path):
    from oeis_learn.generalization.progressive.contracts import reference

    (tmp_path / "secret.json").write_text('{"secret":"unseen continuation"}')
    w = artifact(
        tmp_path,
        "weights.json",
        "checkpoint",
        "generator",
        {
            "artifact": reference(tmp_path, "secret.json"),
            "compatibility": "sha256:" + "b" * 64,
            "profile": "bigint-body/v1",
        },
    )
    with pytest.raises(ValueError, match="aliases private truth"):
        project(
            tmp_path,
            "bad",
            "generate",
            {"visible": [], "model": [w]},
            registry=REGISTRY,
            forbidden=[reference(tmp_path, "secret.json")],
        )
