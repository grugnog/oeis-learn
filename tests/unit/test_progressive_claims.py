"""T064 prospective contracts for T069/T070/T071; no proof authority yet.

These tests fail for the absent owners before implementation. Expected failures
are conditional solely on those modules being absent; any installed owner must
pass the assertions. They are not a native/software acceptance receipt.
"""

import copy
import importlib.util
import pytest

from oeis_learn.generalization.progressive.contracts import digest


def pending(module):
    return pytest.mark.xfail(
        importlib.util.find_spec("oeis_learn.generalization.progressive." + module) is None,
        strict=True,
        raises=ModuleNotFoundError,
        reason=f"prospective T069–T071 {module} owner",
    )


SOURCE = "(program (return (mul (add $index (int 1)) (add $index (int 1)))))"


def certificate(**changes):
    return (
        dict(
            version="progressive-polynomial-certificate/v1",
            source=digest(SOURCE),
            coefficients=["1", "2", "1"],
            arithmetic="mathematical_integers",
            assumptions=[],
            machine_domain=None,
        )
        | changes
    )


@pending("polynomial_certificate")
def test_source_bound_true_polynomial_has_separate_machine_safety_limit():
    from oeis_learn.generalization.progressive.polynomial_certificate import check

    result = check(SOURCE, certificate())
    assert result["statement"] == "proved"
    assert result["machine_safety"] == "unverified"
    assert result["arithmetic"] == "mathematical_integers"


@pending("polynomial_certificate")
@pytest.mark.parametrize(
    "change",
    [
        dict(coefficients=["1", "1", "1"]),
        dict(source=digest("other source")),
        dict(version="legacy-ir-proof/v1"),
    ],
)
def test_wrong_coefficients_unbound_source_and_legacy_proof_cannot_promote(change):
    from oeis_learn.generalization.progressive.polynomial_certificate import check

    with pytest.raises(ValueError):
        check(SOURCE, certificate(**change))


@pending("polynomial_certificate")
def test_loop_is_unsupported_even_when_finite_values_match_polynomial():
    from oeis_learn.generalization.progressive.polynomial_certificate import check

    source = "(program (local $b0 bigint) (local $c0 control) (set $b0 (int 0)) (loop $c0 (control 3) (set $b0 (add (get $b0) (int 1)))) (return (get $b0)))"
    with pytest.raises(ValueError):
        check(source, certificate(source=digest(source), coefficients=["3"]))


@pending("claims")
def test_novelty_axis_and_spoofed_independent_reviewer():
    from oeis_learn.generalization.progressive.claims import validate_claim

    base = dict(statement="finite_agreement", novelty="not_reviewed", proof=None, review=None)
    assert validate_claim(base, reviewers={})["statement"] == "finite_agreement"
    with pytest.raises(ValueError):
        validate_claim(
            base
            | dict(
                novelty="independently_confirmed",
                review=dict(reviewer="model", evidence=digest("self assertion")),
            ),
            reviewers={},
        )
    with pytest.raises(ValueError):
        validate_claim(
            base | dict(statement="proved", proof=dict(version="legacy-ir-proof/v1")), reviewers={}
        )


@pending("quarantine")
def test_refutation_keeps_originals_and_enumerates_all_typed_descendants():
    from oeis_learn.generalization.progressive.quarantine import refute

    graph = [
        dict(id="target", kind="target", parents=[]),
        dict(id="archive", kind="archive", parents=["target"]),
        dict(id="library", kind="library", parents=["archive"]),
        dict(id="update", kind="consumed_update", parents=["library"]),
        dict(id="checkpoint", kind="checkpoint", parents=["update"]),
        dict(id="dossier", kind="dossier", parents=["checkpoint"]),
        dict(id="clean", kind="checkpoint", parents=[]),
    ]
    original = copy.deepcopy(graph)
    result = refute(graph, roots=["target"])
    assert set(result["affected"]) == {
        "target",
        "archive",
        "library",
        "update",
        "checkpoint",
        "dossier",
    }
    assert graph == original
