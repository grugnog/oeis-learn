"""Sealed source selection, finite evidence and independent novelty boundaries."""

import copy
import pytest
from oeis_learn.generalization.progressive.contracts import atomic_write, digest, reference, seal

SOURCE = "(program (return (add $index (int 1))))"


def selected(root):
    from oeis_learn.generalization.progressive.claims import seal_candidate

    atomic_write(root / "snapshot.json", dict(source="hand-checked diagnostic", offset="0"))
    candidate = seal(
        dict(
            version="discovery-candidates/v1",
            model=digest("diagnostic model"),
            checkpoint=None,
            model_control="random",
            recipe=digest("diagnostic recipe"),
            methods=["correct"],
            prompts=digest("visible"),
            clock={},
            counts={},
            elapsed_us=1,
            records={
                "new": dict(
                    selections={"equal_budget": dict(selected=dict(source=SOURCE, verified=True))},
                    costs={},
                )
            },
        )
    )
    atomic_write(root / "selection.json", candidate)
    return seal_candidate(
        root,
        reference(root, "selection.json"),
        "new",
        source_snapshot=reference(root, "snapshot.json"),
        offset="0",
        budget={
            k: 1
            for k in (
                "tokens",
                "draws",
                "rejections",
                "attempts",
                "native_work",
                "elapsed_us",
                "disk_bytes",
                "archive_bytes",
            )
        },
        output="sealed.json",
        purpose="acceptance",
        track="synthetic_strict",
    )


def dossier(root, client=None):
    from oeis_learn.generalization.progressive.claims import create_dossier

    candidate = selected(root)
    return create_dossier(
        root,
        candidate,
        indices=list(map(str, range(100))),
        expected=[str(i + 1) for i in range(100)],
        output="dossier.json",
        client=client,
    )


def test_finite_dossier_preserves_source_offset_and_unreviewed_novelty(tmp_path):
    from oeis_learn.generalization.progressive.claims import validate_dossier

    value = dossier(tmp_path)
    assert value["payload"]["claim"]["statement"] == "finite_agreement"
    assert value["payload"]["claim"]["novelty"] == "not_reviewed"
    assert value["payload"]["source"] == SOURCE and value["payload"]["offset"] == "0"
    assert value["payload"]["tested"]["observed"] == [str(i + 1) for i in range(100)]
    assert value["payload"]["untested_scope"] and not value["payload"]["training_eligible"]
    assert validate_dossier(tmp_path, value) == value
    (tmp_path / "selection.json").write_text("{}")
    with pytest.raises(ValueError):
        validate_dossier(tmp_path, value)


def test_counterexample_cannot_acquire_finite_or_proved_label(tmp_path):
    from oeis_learn.generalization.progressive.claims import create_dossier, validate_dossier

    ref = selected(tmp_path)
    value = create_dossier(
        tmp_path,
        ref,
        indices=list(map(str, range(100))),
        expected=["0"] * 100,
        output="dossier.json",
    )
    assert value["payload"]["claim"]["statement"] == "conjecture"
    assert len(value["payload"]["counterexamples"]) == 100
    bad = copy.deepcopy(value)
    bad["payload"]["claim"]["statement"] = "finite_agreement"
    with pytest.raises(ValueError):
        validate_dossier(tmp_path, seal(bad))


def test_review_is_new_record_bound_to_original_candidate_and_prior_work(tmp_path):
    from oeis_learn.generalization.progressive.claims import review_dossier

    value = dossier(tmp_path)
    before = (tmp_path / "dossier.json").read_bytes()
    candidate = value["payload"]["candidate"]["sha256"]
    atomic_write(
        tmp_path / "review.json",
        dict(
            candidate=candidate,
            prior_work=["explicit comparison"],
            finding="independent diagnostic reviewer fixture",
        ),
    )
    evidence = reference(tmp_path, "review.json")
    reviewer = dict(
        name="Independent reviewer fixture",
        independent=True,
        evidence=evidence,
        candidate=candidate,
        scope="finite diagnostic prior-work comparison",
    )
    claim = dict(
        value["payload"]["claim"],
        novelty="independently_confirmed",
        review=dict(
            reviewer="reviewer-1", evidence=evidence, candidate=candidate, scope=reviewer["scope"]
        ),
    )
    result = review_dossier(
        tmp_path, value, claim, reviewers={"reviewer-1": reviewer}, output="reviewed.json"
    )
    assert result["payload"]["claim"]["novelty"] == "independently_confirmed"
    assert result["payload"]["claim"]["statement"] == "finite_agreement"
    assert result["parents"] == [value["identity"]]
    assert (tmp_path / "dossier.json").read_bytes() == before
    with pytest.raises(ValueError):
        review_dossier(tmp_path, value, claim, reviewers={}, output="spoof.json")
    with pytest.raises(ValueError):
        review_dossier(
            tmp_path,
            value,
            dict(claim, statement="proved", proof={"version": "legacy-ir-proof/v1"}),
            reviewers={"reviewer-1": reviewer},
            output="legacy.json",
        )


def test_native_dossier_executes_source_and_keeps_proof_axis_unverified(tmp_path, fitting_client):
    value = dossier(tmp_path, fitting_client)
    assert value["payload"]["tested"]["native"] is not None
    assert value["payload"]["claim"]["proof"] is None


def review_inputs(root):
    from tests.contract.test_progressive_roles import artifact
    from oeis_learn.generalization.progressive.projections import project
    from oeis_learn.pipeline_validation.progressive_roles import REGISTRY

    value = dossier(root)
    original = artifact(root, "dossier-channel.json", "sealed_dossier", "review", value)
    request = artifact(
        root,
        "review-channel.json",
        "review_evidence",
        "review",
        dict(claim=value["payload"]["claim"]),
    )
    views = project(
        root, "projection", "review-claim", {"dossier": [original, request]}, registry=REGISTRY
    )
    return value, views, [original, request]


def test_review_projection_binds_original_execution_and_cannot_enter_training(tmp_path):
    from oeis_learn.generalization.progressive.claims import validate_dossier
    from oeis_learn.generalization.progressive.projections import project
    from oeis_learn.pipeline_validation.progressive_roles import REGISTRY
    from oeis_learn.generalization.progressive.storage import projection_size

    value, views, descriptors = review_inputs(tmp_path)
    assert validate_dossier(views["dossier"], value) == value
    assert (views["dossier"] / "dossier.json.execution.json").is_file()
    size, _ = projection_size(tmp_path, [v.ref for v in descriptors])
    assert size > sum(v.ref["bytes"] for v in descriptors)
    with pytest.raises(ValueError):
        project(tmp_path, "leak", "train", {"training": [descriptors[0]]}, registry=REGISTRY)
    with pytest.raises(ValueError):
        project(
            tmp_path,
            "forbidden",
            "review-claim",
            {"dossier": descriptors},
            registry=REGISTRY,
            forbidden=[value["payload"]["tested"]["execution"]],
        )


def test_installed_native_review_preserves_finite_scope(tmp_path, fitting_client):
    from oeis_learn.pipeline_validation.progressive_role import execute
    from oeis_learn.generalization.progressive.contracts import load
    from tests.unit.test_progressive_budgets import role_authority

    value, views, descriptors = review_inputs(tmp_path)
    values = [load(tmp_path / d.ref["path"]) for d in descriptors]
    authority, _, _ = role_authority(tmp_path, "review-claim", values, useful_us=30_000_000)
    atomic_write(tmp_path / "authority.json", authority)
    result = execute(
        "review-claim",
        inputs=tmp_path / "projection",
        output=tmp_path / "result",
        authority=tmp_path / "authority.json",
    )
    assert result["mechanical_status"] == "COMPLETE"
    reviewed = load(tmp_path / "result/dossier.json")
    assert reviewed["payload"]["claim"] == value["payload"]["claim"]
    assert reviewed["parents"] == [value["identity"]]
    assert not reviewed["payload"]["training_eligible"]


def test_live_novelty_axis_contract():
    from tests.unit.test_progressive_claims import (
        test_novelty_axis_and_spoofed_independent_reviewer,
    )

    test_novelty_axis_and_spoofed_independent_reviewer()


@pytest.mark.parametrize("change", ["extra", "offset", "range", "untested", "source", "execution"])
def test_dossier_scope_and_original_execution_cannot_be_relabelled(tmp_path, change):
    from oeis_learn.generalization.progressive.claims import validate_dossier

    value = dossier(tmp_path)
    bad = copy.deepcopy(value)
    if change == "extra":
        bad["payload"]["trusted_proof"] = True
    if change == "offset":
        bad["payload"]["offset"] = "1"
    if change == "range":
        bad["payload"]["tested"]["indices"][0] = "-1"
    if change == "untested":
        bad["payload"]["untested_scope"] = []
    if change == "source":
        bad["payload"]["source"] = "(program (return (int 0)))"
    if change == "execution":
        bad["payload"]["tested"]["native"] = digest("unrelated native producer")
    with pytest.raises(ValueError):
        validate_dossier(tmp_path, seal(bad))


def test_conflicting_original_comparison_references_are_not_deduplicated(tmp_path):
    from oeis_learn.generalization.progressive.claims import validate_dossier

    value = dossier(tmp_path)
    ref = value["payload"]["source_snapshot"]
    value["payload"]["known_baselines"] = [dict(ref, sha256=digest("wrong")), ref]
    with pytest.raises(ValueError, match="conflicting"):
        validate_dossier(tmp_path, seal(value))
