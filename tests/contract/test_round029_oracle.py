"""Private practice projections and finite ambiguity witnesses for FR-010/FR-011."""

import pytest

from oeis_learn.decoder.bigint_codec import decode_body
from oeis_learn.generalization.progressive.contracts import seal
from oeis_learn.generalization.progressive.family_search import (
    oracle_projection,
    witnessed_ambiguity,
)
from oeis_learn.pipeline_validation.progressive_velocity_roles import validate_role_value
from oeis_learn.synthesis.reference import evaluate
from oeis_learn.synthesis.reference_parser import parse


def practice(source, uses, *, sharing="explicit"):
    indices = list(map(str, range(-7, 93)))
    result = evaluate(source, indices)
    assert result.outcome == "ok"
    return seal(
        dict(
            version="round029-practice-witness/v1",
            family="private-practice-family",
            source=source,
            indices=indices,
            values=list(result.values),
            sharing=sharing,
            occurrences=[dict(ordinal=i, use=use) for i, use in enumerate(uses)],
        )
    )


def project(record):
    return oracle_projection(record, record["identity"])


def test_only_the_offset_and_shared_skeleton_reach_the_fitter():
    source = "(program (return (add (sub $index (int -7)) (add (int 431) (int 431)))))"
    row = practice(source, ["first_index", "p0", "p0"])
    result = project(row)
    assert result["disposition"] == "ready"
    skeleton = decode_body(result["skeleton"]["tokens"])
    assert parse(skeleton, allow_parameters=True) == parse(
        source.replace("(int 431)", "(param.get p0)"), allow_parameters=True
    )
    assert result["visible"] == {k: row[k][:20] for k in ("indices", "values")}
    assert set(result["skeleton"]) == {"tokens"} and "431" not in skeleton
    validate_role_value("round029_skeleton", result["skeleton"])
    validate_role_value("round029_visible", result["visible"])
    for kind, channel in (("round029_skeleton", "skeleton"), ("round029_visible", "visible")):
        with pytest.raises(ValueError):
            validate_role_value(kind, dict(result[channel], witness=row["source"]))
    with pytest.raises(ValueError, match="commitment"):
        oracle_projection(seal(dict(row, family="switched")), row["identity"])
    with pytest.raises(ValueError, match="sharing"):
        project(seal(dict(row, sharing="distinct-occurrences")))
    with pytest.raises(ValueError, match="shared"):
        project(practice(source.replace("(int 431)", "(int 432)", 1), ["first_index", "p0", "p0"]))


def test_more_than_eight_occurrences_with_one_shared_parameter_are_unsupported():
    expr = "(int 813)"
    for _ in range(8):
        expr = "(add (int 813) " + expr + ")"
    result = project(practice("(program (return " + expr + "))", ["p0"] * 9))
    assert result == {"disposition": "unsupported_occurrence_count"}


def test_control_taint_cannot_leave_the_private_coefficient_in_the_projection():
    source = "(program (return (select (lt $index (int 118)) (int 3) (int 3))))"
    assert project(practice(source, ["p0", "p1", "p1"])) == {"disposition": "unsupported_control"}
    assert project(practice(source, ["first_index", "p1", "p1"])) == {
        "disposition": "unsupported_structural_literal"
    }
    with pytest.raises(ValueError, match="occurrence"):
        project(practice(source, ["p0", "p1"]))
    assert project(practice("(program (return (sub $index (int -7))))", ["first_index"])) == {
        "disposition": "zero_unknowns"
    }


def test_ambiguity_requires_two_legal_groundings_with_a_verified_suffix_difference():
    source = "(program (return (select (le $index (int 12)) (int 8) (int 917))))"
    row = practice(source, ["last_visible_index", "p0", "p1"])
    first = {"p0": "8", "p1": "917"}

    def compare(left, right):
        return witnessed_ambiguity(row, row["identity"], left, right)

    evidence = compare(first, dict(first, p1="21"))
    assert (
        evidence["disposition"] == "witnessed_ambiguity"
        and evidence["first_different_index"] == "13"
    )
    assert len(set(evidence["groundings"])) == 2
    assert compare(first, first) is None
    assert compare(first, dict(first, p0="7")) is None
    with pytest.raises(ValueError):
        compare(first, {"p0": "8"})
