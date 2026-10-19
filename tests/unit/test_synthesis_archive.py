import pytest
from oeis_learn.fresh.candidate_archive import retain_alternatives, training_representative
from oeis_learn.synthesis.discovery_status import validate_discovery


def rows():
    return [
        dict(
            id=str(i),
            structure=f"s{i % 3}",
            source="x" * (i + 1),
            cost=12 - i,
            verified=True,
            verification="sha256:" + "1" * 64,
            track="generic_only",
            profile="bigint-body/v1",
            assignment={"p0": "1"},
        )
        for i in range(12)
    ]


def test_alternatives_structure_tradeoffs_cap_and_omission_ledger():
    result = retain_alternatives(rows())
    assert len(result["retained"]) == 8 and len(result["omitted"]) == 4
    assert {r["structure"] for r in result["retained"]} == {"s0", "s1", "s2"}
    assert result == retain_alternatives(list(reversed(rows())))
    before = list(result["retained"])
    assert training_representative(result)["source"] == "x"
    assert before == result["retained"]
    with pytest.raises(ValueError):
        retain_alternatives([dict(rows()[0], verified=False)])


def test_discovery_fields_never_promote_finite_match_to_proof_or_novelty():
    finite = dict(
        finite_match="sha256:" + "1" * 64,
        conjecture=None,
        bounded_verification=None,
        proved_statement=None,
        novelty_review=None,
        historical_status=None,
    )
    assert validate_discovery(finite) == finite
    with pytest.raises(ValueError):
        validate_discovery({**finite, "proved_statement": True})
    with pytest.raises(ValueError):
        validate_discovery({**finite, "novelty_review": "automatic"})
    historical = {**finite, "finite_match": None, "historical_status": "historical_unreplayed"}
    assert validate_discovery(historical) == historical
