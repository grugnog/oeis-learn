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


@pytest.mark.parametrize("failure", ["late", "interrupted"])
def test_installed_admission_keeps_native_work_charged(tmp_path, monkeypatch, failure):
    import time
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive.contracts import load, digest
    from oeis_learn.pipeline_validation.progressive_role import execute
    from tests.unit.test_progressive_budgets import role_authority
    from oeis_learn.synthesis.client import SynthesisClient

    member = dict(
        id="training",
        group="group",
        role="training",
        source_snapshot=digest("snapshot"),
        indices=list(map(str, range(100))),
        values=list(map(str, range(100))),
    )
    values = [
        seal(
            dict(
                version="discovery-channel/v1",
                role="admission",
                kind="candidate",
                data=dict(task="training", source="(program (return $index))"),
            )
        ),
        seal(
            dict(
                version="discovery-channel/v1",
                role="admission",
                kind="training_continuation",
                data=member,
            )
        ),
    ]
    for channel, value in zip(("candidates", "admission_truth"), values):
        atomic_write(tmp_path / "inputs" / channel / (value["identity"][7:] + ".json"), value)
    a, journal, task = role_authority(
        tmp_path, "admit-training", values, useful_us=1_000_000 if failure == "late" else 5_000_000
    )
    atomic_write(tmp_path / "authority.json", a)
    calls = []

    class Worker:
        def call(self, request, *, deadline):
            calls.append(request)
            if failure == "interrupted":
                raise InterruptedError("interrupted native admission")
            time.sleep(max(0, deadline - time.monotonic()) + 0.01)
            return dict(outcome="ok", values=member["values"])

    class Client:
        def __init__(self, *args):
            self.worker = Worker()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def execute(self, source, indices, *, deadline):
            return self.worker.call({"candidates": [{"source": source}]}, deadline=deadline)

    monkeypatch.setattr("oeis_learn.synthesis.client.SynthesisClient", Client)
    error = TimeoutError if failure == "late" else InterruptedError
    with pytest.raises(error):
        execute(
            "admit-training",
            inputs=tmp_path / "inputs",
            output=tmp_path / "out",
            authority=tmp_path / "authority.json",
        )
    assert len(calls) == 1
    import json

    events = [
        json.loads(line) for line in (tmp_path / "out/work-debits.jsonl").read_text().splitlines()
    ]
    assert events[-1]["used"]["native_work"] == 1
    assert journal.task_remaining(task)["native_work"] == task["payload"]["caps"]["native_work"] - 2
    assert not (tmp_path / "out/admission.json").exists()


def test_component_projection_copies_only_bound_training_archive_closure(tmp_path):
    from oeis_learn.generalization.progressive.archive import Archive
    from oeis_learn.generalization.progressive.component_mining import policy
    from oeis_learn.generalization.progressive.contracts import (
        atomic_write,
        reference,
        seal,
        load,
        verify_refs,
    )
    from oeis_learn.generalization.progressive.projections import describe, project
    from oeis_learn.pipeline_validation.progressive_roles import REGISTRY
    from tests.unit.test_progressive_archive import admitted

    archive = Archive(tmp_path / "training")
    cut = archive.publish([admitted("a", "first"), admitted("b", "second")])
    atomic_write(tmp_path / "training/cut.json", cut)
    next_cut = archive.publish([], previous=cut)
    atomic_write(tmp_path / "training/next.json", next_cut)
    paths = []
    for name, kind, data in [
        (
            "archive",
            "verified_training_archive",
            dict(
                membership="training",
                artifact=reference(tmp_path, "training/cut.json"),
                profile="bigint-body/v1",
                compatibility="sha256:" + "a" * 64,
            ),
        ),
        (
            "next-archive",
            "verified_training_archive",
            dict(
                membership="training",
                artifact=reference(tmp_path, "training/next.json"),
                profile="bigint-body/v1",
                compatibility="sha256:" + "a" * 64,
            ),
        ),
        ("policy", "component_policy", dict(policy=policy(), previous=None)),
    ]:
        path = name + ".json"
        paths.append(path)
        atomic_write(
            tmp_path / path,
            seal(dict(version="discovery-channel/v1", role="component", kind=kind, data=data)),
        )
    items = [describe(tmp_path, p) for p in paths]
    result = project(
        tmp_path,
        "projected",
        "extract-components",
        {"archive": items},
        registry=REGISTRY,
        authorized=[v.ref for v in items],
    )
    projected = result["archive"] / "training"
    assert load(projected / "cut.json") == cut
    assert load(projected / "next.json") == next_cut
    assert len(verify_refs(projected, cut["payload"]["entries"])) == 2
    assert not (result["archive"] / "private").exists()


def test_committed_search_library_projection_copies_exact_payload_without_sources(tmp_path):
    from tests.unit.test_progressive_component_mining import sources
    from oeis_learn.generalization.progressive.component_mining import LibraryStore
    from oeis_learn.generalization.progressive.component_search import search_input
    from oeis_learn.generalization.progressive.contracts import reference, digest, load
    from oeis_learn.generalization.progressive.budgets import role_work_requirements

    store = LibraryStore(tmp_path, "library", purpose="acceptance", track="synthetic_strict")
    value = store.publish(sources(tmp_path))
    bundle = search_input(store, value)
    atomic_write(tmp_path / "search-library.json", bundle)
    lib = artifact(
        tmp_path,
        "envelope.json",
        "library",
        "generator",
        dict(
            artifact=reference(tmp_path, "search-library.json"),
            compatibility=digest("model"),
            profile="progressive-component-search/v1",
        ),
    )
    prompt = artifact(
        tmp_path,
        "prompt.json",
        "visible_prompt",
        "generator",
        dict(id="heldout", indices=list(map(str, range(20))), values=list(map(str, range(20)))),
    )
    views = project(
        tmp_path, "projection", "generate", {"model": [lib], "visible": [prompt]}, registry=REGISTRY
    )
    assert load(views["model"] / "search-library.json") == bundle
    assert not list((tmp_path / "projection").rglob("source-*.json"))
    values = [load(tmp_path / x.ref["path"]) for x in (lib, prompt)]
    costs = role_work_requirements("generate", values)
    assert costs["attempts"] == costs["draws"] == 8
    assert costs["tokens"] == 16384
    assert costs["native_work"] > 64
    (tmp_path / "search-library.json").write_text("{}")
    with pytest.raises(ValueError):
        project(
            tmp_path, "bad", "generate", {"model": [lib], "visible": [prompt]}, registry=REGISTRY
        )
