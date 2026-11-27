import pytest
from oeis_learn.generalization.progressive.contracts import digest, seal
from oeis_learn.generalization.progressive.selection import rank


from tests.helpers.progressive_trial import evidence


def row(**changes):
    return (
        dict(
            id="baseline",
            value=0.3,
            elapsed_us=100,
            tokens=4000000,
            status="COMPLETE",
            role="development",
            evidence=evidence(),
        )
        | changes
    )


def test_missing_evidence_and_supplied_eligibility_cannot_win():
    good = row()
    assert rank([good])["ordered"] == ["baseline"]
    for bad in [
        row(id="bad", evidence=None),
        row(id="bad", value=1),
        row(id="bad", evidence=evidence(completion=False)),
        row(id="bad", evidence=evidence(retention=True)),
    ]:
        assert rank([good, bad])["ordered"] == ["baseline"]


def test_forged_or_filtered_producer_evidence_is_rejected():
    e = evidence()
    e["outcomes"] = e["outcomes"][:1]
    assert rank([row(evidence=seal(e))])["ordered"] == []


def test_resealed_research_label_without_original_producers_cannot_win():
    assert rank([row(evidence=seal(dict(evidence(), scope="NATIVE_VERIFIED")))])["ordered"] == []


def test_complete_tell_cannot_feed_sampler_unverified_scalars(tmp_path):
    from oeis_learn.generalization.progressive.trials import TrialJournal

    j = TrialJournal(tmp_path, study=digest("study"))
    j.propose(0, parameters={"lr_multiplier": 1}, seed=13)
    j.dispatch(0)
    result = {k: v for k, v in row().items() if k != "id"}
    with pytest.raises(ValueError):
        j.tell(0, {k: v for k, v in result.items() if k != "evidence"})
    assert j.tell(0, result)["value"] == 0.3
