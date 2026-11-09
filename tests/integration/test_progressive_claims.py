"""Replayable polynomial promotion through immutable dossier review."""

import copy

import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write, digest, reference, seal
from tests.unit.test_progressive_dossiers import SOURCE, dossier, selected


def proof(root, **changes):
    value = (
        dict(
            version="progressive-polynomial-certificate/v1",
            source=digest(SOURCE),
            coefficients=["1", "1"],
            arithmetic="mathematical_integers",
            assumptions=[],
            machine_domain=None,
        )
        | changes
    )
    atomic_write(root / "proof.json", value)
    return reference(root, "proof.json")


def test_scoped_proof_replay_is_immutable_and_never_proves_novelty_or_sequence(tmp_path):
    from oeis_learn.generalization.progressive.claims import review_dossier, validate_dossier

    value = dossier(tmp_path)
    before = (tmp_path / "dossier.json").read_bytes()
    claim = dict(value["payload"]["claim"], statement="proved", proof=proof(tmp_path))
    reviewed = review_dossier(tmp_path, value, claim, reviewers={}, output="proved.json")
    p = reviewed["payload"]
    assert p["claim"]["novelty"] == "not_reviewed" and not p["training_eligible"]
    assert p["proof_check"]["scope"] == "program_equals_polynomial_only"
    assert p["proof_check"]["sequence_law"] == "unproved"
    assert p["proof_check"]["machine_safety"] == "unverified"
    assert p["machine_safety"] == "finite_execution_only_no_unbounded_safety"
    assert reviewed["parents"] == [value["identity"]]
    assert (tmp_path / "dossier.json").read_bytes() == before
    assert validate_dossier(tmp_path, reviewed) == reviewed
    again = review_dossier(tmp_path, reviewed, claim, reviewers={}, output="replayed.json")
    assert again["payload"]["proof_check"] == p["proof_check"]
    assert again["parents"] == [reviewed["identity"]]


@pytest.mark.parametrize(
    "change", ["scope", "checker", "coefficient", "machine", "assumptions", "missing"]
)
def test_resealed_proof_scope_cannot_override_replay(tmp_path, change):
    from oeis_learn.generalization.progressive.claims import review_dossier, validate_dossier

    value = dossier(tmp_path)
    claim = dict(value["payload"]["claim"], statement="proved", proof=proof(tmp_path))
    reviewed = review_dossier(tmp_path, value, claim, reviewers={}, output="proved.json")
    bad = copy.deepcopy(reviewed)
    if change == "missing":
        del bad["payload"]["proof_check"]
    elif change == "assumptions":
        bad["payload"]["assumptions"] = ["unproved recurrence"]
    else:
        key, val = {
            "scope": ("sequence_law", "proved"),
            "checker": ("checker", digest("other checker")),
            "coefficient": ("coefficients", ["0", "1"]),
            "machine": ("machine_safety", "finite_numeric_range_verified"),
        }[change]
        bad["payload"]["proof_check"][key] = val
        bad["payload"]["proof_check"] = seal(bad["payload"]["proof_check"])
    with pytest.raises(ValueError):
        validate_dossier(tmp_path, seal(bad))


def test_range_argument_does_not_expand_other_dossier_evidence(tmp_path):
    from oeis_learn.generalization.progressive.claims import review_dossier

    value = dossier(tmp_path)
    claim = dict(
        value["payload"]["claim"],
        statement="proved",
        proof=proof(tmp_path, machine_domain={"lower": "-100", "upper": "1000"}),
    )
    reviewed = review_dossier(tmp_path, value, claim, reviewers={}, output="proved.json")
    assert reviewed["payload"]["proof_check"]["machine_safety"] == "finite_numeric_range_verified"
    assert reviewed["payload"]["proof_check"]["resource_safety"] == "unverified"
    for field in ("tested", "untested_scope", "counterexamples", "generation_budget"):
        assert reviewed["payload"][field] == value["payload"][field]


def test_true_program_identity_cannot_promote_a_refuted_sequence_claim(tmp_path):
    from oeis_learn.generalization.progressive.claims import create_dossier, review_dossier

    candidate = selected(tmp_path)
    value = create_dossier(
        tmp_path,
        candidate,
        indices=list(map(str, range(100))),
        expected=["0"] * 100,
        output="refuted.json",
    )
    claim = dict(value["payload"]["claim"], statement="proved", proof=proof(tmp_path))
    with pytest.raises(ValueError, match="counterexample"):
        review_dossier(tmp_path, value, claim, reviewers={}, output="forbidden.json")
    assert not (tmp_path / "forbidden.json").exists()


@pytest.mark.parametrize("change", ["coefficients", "source", "legacy", "assumptions", "bytes"])
def test_failed_promotion_preserves_original_dossier(tmp_path, change):
    from oeis_learn.generalization.progressive.claims import review_dossier

    value = dossier(tmp_path)
    before = (tmp_path / "dossier.json").read_bytes()
    changes = {
        "coefficients": dict(coefficients=["0", "1"]),
        "source": dict(source=digest(SOURCE + " ")),
        "legacy": dict(version="legacy-ir-proof/v1"),
        "assumptions": dict(assumptions=["assuming the theorem"]),
        "bytes": {},
    }[change]
    ref = proof(tmp_path, **changes)
    if change == "bytes":
        ref["bytes"] = (256 << 10) + 1
    with pytest.raises(ValueError):
        review_dossier(
            tmp_path,
            value,
            dict(value["payload"]["claim"], statement="proved", proof=ref),
            reviewers={},
            output="forbidden.json",
        )
    assert (tmp_path / "dossier.json").read_bytes() == before
    assert not (tmp_path / "forbidden.json").exists()
