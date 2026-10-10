from copy import deepcopy
import pytest
from oeis_learn.evaluation.search import SearchLedger, SearchExhausted, SearchSession, typed_repairs


class Evidence:
    def __init__(self, source):
        self.source = source

    def to_dict(self):
        # Two distinct algorithms have identical observed values; both execute.
        return {"outcome": None, "outputs": ["1"] * 20, "reference_outputs": ["1"] * 20}


class Runner:
    def __init__(self):
        self.sources = []

    def run_single(self, source, indices, **kwargs):
        assert indices == list(range(20))
        self.sources.append(source)
        return Evidence(source)


class Proposer:
    def stochastic(self, visible, seed, ledger, **kwargs):
        assert len(visible) == 20
        ledger.charge(expansions=1, tokens=4)
        source = "i256.const 1" if seed % 2 else "i256.const 0 i256.const 1 i256.add"
        return source, {"method": "stochastic"}

    def beam(self, visible, ledger, **kwargs):
        ledger.charge(expansions=1, tokens=4)
        return [("i256.const 1", {"method": "beam"})] * 8


def test_search_dedup_by_source_only_matched_budgets_and_resume():
    runner = Runner()
    proposer = Proposer()
    session = SearchSession([1] * 20, portfolio="baseline16")
    for _ in range(3):
        session.step(proposer, runner)
    state = deepcopy(session.state_dict())
    resumed = SearchSession.from_state(state)
    a, b = session.step(proposer, runner), resumed.step(proposer, runner)
    assert a == b
    report = session.run(proposer, runner)
    assert report["spent"]["attempts"] == 16 and len(report["results"]) == 16
    assert len(set(runner.sources)) == 2  # equal observed outputs do not collapse source identity
    portfolio = SearchSession([1] * 20).run(proposer, Runner())
    assert portfolio["spent"]["attempts"] == 32 and portfolio["label"] == "model+search"
    assert portfolio["comparison_key"] != report["comparison_key"]
    assert len(typed_repairs("i256.const 1")) == 2


def test_shared_budget_caps_and_no_refund():
    ledger = SearchLedger(max_attempts=1, max_expansions=1, max_tokens=1)
    ledger.charge(attempts=1, expansions=1, tokens=1)
    with pytest.raises(SearchExhausted):
        ledger.charge(attempts=1)
    resumed = SearchLedger.from_state(ledger.state_dict())
    assert resumed.attempts == 1 and resumed.elapsed() >= ledger.charged_ns
    with pytest.raises(ValueError):
        SearchSession([1] * 20, ledger=ledger, portfolio="baseline16")
