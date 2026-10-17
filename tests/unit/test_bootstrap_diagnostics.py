"""Diagnostic evidence must separate model errors from incomplete measurement."""

import pytest

from oeis_learn.fresh.bootstrap_config import learning_defaults
from oeis_learn.fresh.common import digest
from oeis_learn.fresh.diagnostics import evaluate
from oeis_learn.fresh.learnability import controls, select_view
from oeis_learn.decoder.program_codec import BOS_ID, encode_body
from tests.unit.test_bootstrap_workflow import ReferenceFixtureRunner
from tests.unit.test_fresh_workflow import examples, small_models


def test_controls_are_closed_bounded_and_do_not_change_learning_defaults():
    before = learning_defaults()
    assert controls()["max_updates"] == 3000
    assert controls(dict(controls(), max_updates=7000, budget_seconds=7200))["max_updates"] == 7000
    assert learning_defaults() == before
    for changes in (
        {"max_updates": True},
        {"budget_seconds": 0},
        {"view": "random"},
        {"diagnostic_seconds": 1000, "budget_seconds": 600},
        {"extra": 1},
    ):
        with pytest.raises(ValueError):
            controls(dict(controls(), **changes))


def test_shortest_control_retains_original_deterministic_selection():
    rows = [
        dict(examples()["training"][0], program_id=digest(n), tokens=[n] * (40 - n))
        for n in range(40)
    ]
    assert (
        select_view(rows, "shortest")
        == sorted(rows, key=lambda r: (len(r["tokens"]), r["program_id"]))[:32]
    )
    with pytest.raises(ValueError, match="32 admitted"):
        select_view(rows[:31], "shortest")
    with pytest.raises(ValueError, match="prepare more"):
        select_view([dict(r, tokens=encode_body("i256.const 1")) for r in rows], "diverse")


def test_diverse_view_selects_only_input_rows_and_excludes_offset_dominance():
    from oeis_learn.data.bootstrap_programs import ConnectedSampler, default_config
    from oeis_learn.sandbox.reference import execute_term
    from oeis_learn.sandbox.wat_ast import parse

    generator = ConnectedSampler(default_config())
    rows = []
    for counter in range(0, 1024, 4):
        sample = generator.sample(counter)
        program = parse(sample["canonical_source"])
        values = [execute_term(program, n) for n in range(20)]
        if any(v.outcome for v in values):
            continue
        rows.append(
            dict(
                program_id=digest(sample),
                values=[str(v.value) for v in values],
                tokens=sample["body_tokens"],
            )
        )
    selected = select_view(rows, "diverse")
    assert len(selected) == 32
    assert all(any(row is original for original in rows) for row in selected)
    assert select_view(list(reversed(rows)), "diverse") == selected
    assert len({digest(row["values"]) for row in selected}) == 32
    affine = sum(
        len({int(b) - int(a) for a, b in zip(row["values"], row["values"][1:])}) <= 1
        for row in selected
    )
    assert affine <= 16
    assert max(map(lambda r: len(r["tokens"]), selected)) <= 128


def proposal(self, visible, seed, ledger, *, greedy):
    source = f"i256.const {visible[0]}"
    return source, dict(
        method="greedy",
        tokens=[BOS_ID] + encode_body(source),
        complete=True,
        failure=None,
        encoder_sha256=digest("e"),
        decoder_sha256=digest("d"),
    )


def diagnostic(monkeypatch, propose=proposal, runner=None, **kwargs):
    encoder, decoder, _ = small_models(None, device="cpu")
    monkeypatch.setattr("oeis_learn.evaluation.search.ModelProposer.stochastic", propose)
    return evaluate(
        encoder,
        decoder,
        examples(),
        dict(learning_defaults(), diagnostic_rows=3, diagnostic_seconds=30),
        runner or ReferenceFixtureRunner(),
        **kwargs,
    )


def test_timeout_is_not_incorrect_and_keeps_emitted_literal(monkeypatch):
    from oeis_learn.evaluation.search import SearchExhausted

    def timeout(*args, **kwargs):
        error = SearchExhausted("shared search budget exhausted", reason="deadline")
        error.tokens = [BOS_ID] + encode_body("i256.const 11")[:-2]
        raise error

    result = diagnostic(monkeypatch, timeout, compare_generation=True)
    training = result["splits"]["training"]
    assert training["outcomes"] == {"timeout": 3}
    assert training["evaluated"] == training["exact"] == 0
    assert not training["coverage_complete"]
    assert not training["generation_parity"]["passed"]
    assert all(a["tokens"][0] == BOS_ID for a in training["attempts"])


def test_incomplete_is_evaluated_but_not_a_complete_or_correct_program(monkeypatch):
    def incomplete(*args, **kwargs):
        source, result = proposal(*args, **kwargs)
        result.update(
            complete=False,
            failure="token_cap_without_eos",
            tokens=[BOS_ID] + encode_body(source)[:-2],
        )
        return "", result

    training = diagnostic(monkeypatch, incomplete)["splits"]["training"]
    assert training["outcomes"] == {"incomplete": 3}
    assert training["evaluated"] == 3
    assert training["exact"] == training["completed"] == 0


def test_free_generation_parity_covers_every_training_prompt_and_detects_mismatch(monkeypatch):
    def mismatch(self, *args, **kwargs):
        source, result = proposal(self, *args, **kwargs)
        if not self.use_cache:
            source = "i256.const 111"
            result["tokens"] = [BOS_ID] + encode_body(source)
        return source, result

    result = diagnostic(monkeypatch, mismatch, compare_generation=True, detail=True)
    training = result["splits"]["training"]
    assert training["exact"] == 3
    assert (
        training["generation_parity"]["requested"]
        == training["generation_parity"]["evaluated"]
        == 3
    )
    assert not training["generation_parity"]["passed"]
    assert all(a["teacher_forced"]["cache_parity"]["passed"] for a in training["attempts"])
    assert all(a["visible_values"] and a["target_token_ids"] for a in training["attempts"])


@pytest.mark.parametrize("literal", [11, -11, 111, -111, 101, 110, 2**200 + 11])
def test_teacher_trace_covers_literal_delimiters_eos_and_cached_positions(literal):
    from oeis_learn.fresh.token_diagnostic import teacher_trace

    encoder, decoder, _ = small_models(None, device="cpu")
    row = dict(values=[str(literal)] * 20, tokens=encode_body(f"i256.const {literal}"))
    result = teacher_trace(encoder, decoder, row)
    assert result["cache_parity"]["passed"]
    assert [t["target"] for t in result["tokens"]] == row["tokens"]
    spellings = [t["spelling"] for t in result["tokens"]]
    assert "</int>" in spellings and spellings[-1] == "<eos>"
    assert all(0 <= t["raw_probability"] <= 1 for t in result["tokens"])


def test_search_ledger_distinguishes_time_and_token_caps():
    from oeis_learn.evaluation.search import SearchExhausted, SearchLedger

    ledger = SearchLedger(max_tokens=1)
    with pytest.raises(SearchExhausted) as failed:
        ledger.charge(tokens=2)
    assert failed.value.reason == "token_limit"
    ledger.charged_ns = ledger.budget_ns
    with pytest.raises(SearchExhausted) as failed:
        ledger.charge()
    assert failed.value.reason == "deadline"
