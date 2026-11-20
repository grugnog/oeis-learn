import pytest
from tests.helpers.progressive_fixture import envelope
from oeis_learn.generalization.progressive.exposure import (
    append,
    exposed,
    validate_extension,
    detect,
)
from oeis_learn.pipeline_validation.identity import seal, digest


def test_seed_cohort_renaming_cannot_erase_exposure():
    registry = envelope(
        "exposure", dict(previous=None, detector_limits=["finite windows only"], events=[])
    )
    event = dict(
        group="old",
        relations=["source:related"],
        role="development",
        endpoint="unseen_structure",
        source="fresh-002",
        access="outcome",
        campaign="old-012",
    )
    newer = append(registry, [event], created_at=registry["created_at"])
    validate_extension(registry, newer)
    assert exposed(newer, "source:related", role="assessment", endpoint="unseen_structure")
    assert exposed(registry, "fresh-002-reseeded", role="confirmation", endpoint="composition")
    changed = seal(dict(newer, payload=dict(newer["payload"], events=[])))
    with pytest.raises(ValueError):
        validate_extension(newer, changed)


def test_detector_groups_prefix_shift_source_and_structure():
    def row(id, values):
        return dict(
            id=id,
            indices=list(map(str, range(100))),
            values=list(map(str, values)),
            structure=digest(id),
            composition=digest(id + "c"),
            relations=[],
        )

    a = row("a", range(100))
    b = row("b", range(1, 101))
    c = row("c", list(range(20)) + [1000] * 80)
    result = detect([a, b, c])
    groups = [set(r["groups"]) for r in result]
    assert groups[0] & groups[1] and groups[0] & groups[2]
    assert "not all-domain" in result[0]["limits"]
