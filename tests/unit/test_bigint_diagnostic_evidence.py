import time
from types import SimpleNamespace

import pytest

from oeis_learn.fresh import bigint_diagnostics as diagnostics
from oeis_learn.fresh.synthesis_policy import select_visible
from oeis_learn.synthesis.admission import VisiblePrompt, source_identity
from oeis_learn.synthesis.contracts import ResourceProfile


def selection(source="(program (return $index))"):
    prompt = VisiblePrompt(tuple(map(str, range(20))), tuple(map(str, range(20))))
    chosen = select_visible(
        [
            dict(
                source=source,
                source_sha256=source_identity(source),
                verified=True,
                visible_count=20,
                cost=0,
            )
        ]
    )
    return dict(
        selected=chosen, prompt_sha256=prompt.sha256, verification=dict(prompt_sha256=prompt.sha256)
    )


def truth():
    return {"prompt": dict(indices=list(map(str, range(100))), values=list(map(str, range(100))))}


class Client:
    limits = ResourceProfile()

    def execute(self, source, indices, **kwargs):
        return dict(outcome="ok", values=list(indices))


def test_expired_score_cannot_inherit_generation_completeness():
    result = diagnostics.score(
        {"prompt": dict(selections={"rescue": selection()}, complete=True)},
        truth(),
        Client(),
        deadline=time.monotonic() - 1,
    )
    outcome = result["outcomes"]["prompt"]["rescue"]
    assert not outcome["complete"] and not outcome["success"]
    assert outcome["scoring"]["reason"] == "deadline"


def test_no_selection_uses_its_own_arm_evidence():
    result = diagnostics.score(
        {
            "prompt": dict(
                selections={},
                complete=False,
                arm_evidence={
                    "rescue": dict(complete=True, reason="unsupported"),
                    "absent": dict(complete=False, reason="generation_incomplete"),
                },
            )
        },
        truth(),
        Client(),
        deadline=time.monotonic() + 10,
    )
    assert result["outcomes"]["prompt"]["rescue"]["complete"]
    assert not result["outcomes"]["prompt"]["absent"]["complete"]
    assert not result["outcomes"]["prompt"]["rescue"]["success"]


def test_legacy_missing_selection_is_explicitly_unattributed():
    result = diagnostics.score(
        {"prompt": dict(selections={}, complete=True)},
        truth(),
        Client(),
        deadline=time.monotonic() + 10,
    )
    outcome = result["outcomes"]["prompt"]["rescue"]
    assert not outcome["complete"]
    assert outcome["reason"] == "legacy_arm_evidence_missing"


def test_selected_candidate_with_complete_independent_horizon_is_successful():
    result = diagnostics.score(
        {"prompt": dict(selections={"rescue": selection()}, complete=False)},
        truth(),
        Client(),
        deadline=time.monotonic() + 10,
    )
    outcome = result["outcomes"]["prompt"]["rescue"]
    assert outcome["complete"] and outcome["success"]
    assert outcome["scoring"]["production"]["coverage"] == 100
    assert outcome["scoring"]["reference"]["coverage"] == 100


@pytest.mark.parametrize(
    "production",
    [
        dict(outcome="ok", values=["0"]),
        dict(outcome="resource_exhausted", reason="deadline", values=[]),
    ],
)
def test_partial_or_failed_execution_is_incomplete(production):
    client = SimpleNamespace(limits=ResourceProfile(), execute=lambda *a, **kw: production)
    result = diagnostics.score(
        {"prompt": dict(selections={"rescue": selection()}, complete=True)},
        truth(),
        client,
        deadline=time.monotonic() + 10,
    )
    assert not result["outcomes"]["prompt"]["rescue"]["complete"]


def test_rescue_coverage_does_not_include_later_stochastic_fits(monkeypatch):
    from oeis_learn.decoder.bigint_codec import decode_body, encode_body
    from oeis_learn.fresh.bigint_config import defaults

    tokens = encode_body("(program (return $index))")
    source = decode_body(tokens)
    monkeypatch.setattr(
        diagnostics,
        "_raw",
        lambda *a, **kw: (None, dict(complete=True, outcome="visible_mismatch", reason=None)),
    )
    fits = iter(["unsupported", "resource_exhausted"])
    client = SimpleNamespace(fit=lambda *a, **kw: dict(outcome=next(fits), admitted=[]))
    row = dict(id="prompt", indices=list(map(str, range(20))), values=["0"] * 20)
    portfolio = diagnostics._portfolio(
        None, None, row, defaults(), client, deadline=time.monotonic() + 60, attempt_count=2
    )
    next(portfolio)
    portfolio.send(
        dict(
            source=source,
            tokens=tokens,
            complete=True,
            generation_seconds=0,
            generation_batch_size=1,
        )
    )
    with pytest.raises(StopIteration) as stopped:
        portfolio.send(
            dict(
                source=source,
                tokens=tokens,
                complete=True,
                generation_seconds=0,
                generation_batch_size=1,
            )
        )
    evidence = stopped.value.value["arm_evidence"]
    assert evidence["rescue"]["complete"]
    assert not evidence["equal_budget"]["complete"]
    assert evidence["rescue"]["fitting"]["outcomes"] == ["unsupported"]
