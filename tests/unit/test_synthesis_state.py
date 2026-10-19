import copy
import pytest
from oeis_learn.synthesis.state import (
    BudgetLedger,
    ExperimentContract,
    validate_checkpoint,
    FinalPlan,
    FinalStudy,
    CandidateState,
)

H = "sha256:" + "1" * 64
J = "sha256:" + "2" * 64


def contract():
    return ExperimentContract(
        data=H,
        split=H,
        model=H,
        inputs=H,
        decoder=H,
        solver=H,
        runtime=H,
        resources=H,
        policy=H,
        track="generic_only",
        foundation=None,
        seed=17,
        budget=100,
    )


def plan():
    return FinalPlan(
        cohort=H,
        decisions=tuple((k, H) for k in FinalPlan.DECISIONS),
        arms=(("full", 17, H, H, H, H, 20), ("control", 17, J, H, H, H, 20)),
        denominator=128,
        controls=("correct", "shuffled", "absent", "template"),
        budget=40,
    )


def test_conservative_debit_interruption_exact_resume_and_weights_only():
    c = contract()
    ledger = BudgetLedger(100)
    ledger.start("batch1", 30)
    recovered = BudgetLedger.restore(ledger.snapshot())
    assert recovered.remaining == 70
    with pytest.raises(ValueError):
        recovered.start("batch1", 30)
    recovered.finish("batch1", "interrupted")
    assert recovered.remaining == 70
    checkpoint = {
        k: H
        for k in ("weights", "optimizer", "rng", "data_order", "selection", "curriculum", "replay")
    }
    checkpoint.update(contract=c.sha256, ledger=recovered.snapshot(), state="interrupted")
    validate_checkpoint(checkpoint, c)
    with pytest.raises(ValueError):
        validate_checkpoint({"weights": H}, c)
    with pytest.raises(ValueError):
        validate_checkpoint({**checkpoint, "contract": J}, c)
    with pytest.raises(ValueError):
        validate_checkpoint({**checkpoint, "state": "completed"}, c)
    with pytest.raises(TypeError):
        ExperimentContract(**vars(c), unknown=True)


def test_final_plan_all_arms_shared_budget_and_no_restored_freshness():
    p = plan()
    s = FinalStudy(H)
    with pytest.raises(ValueError):
        s.start("full", H, 5, operation="train")
    s.authorize(p)
    s.start("full", H, 10, operation="evaluate_final")
    assert s.state == "exposed"
    restored = FinalStudy.restore(s.snapshot(), p)
    assert restored.state == "exposed" and restored.ledger.remaining == 30
    restored.finish("full", "interrupted")
    restored.start("full", H, 10, operation="evaluate_final")
    restored.finish("full", "completed")
    assert restored.status["control"] == "unstarted"
    with pytest.raises(ValueError):
        restored.start("full", H, 1, operation="evaluate_final")
    changed = copy.copy(p)
    object.__setattr__(changed, "budget", 41)
    with pytest.raises(ValueError):
        restored.authorize(changed)
    with pytest.raises(ValueError):
        restored.start("control", H, 1, operation="evaluate_final")
    restored.start("control", J, 20, operation="evaluate_final")
    assert restored.ledger.remaining == 0


def test_incomplete_decisions_and_stale_provisional_responses():
    p = plan()
    with pytest.raises(ValueError):
        FinalPlan(**{**vars(p), "decisions": p.decisions[:-1]})
    state = CandidateState("candidate", 3, H)
    assert not state.response(2, H, "assignment_found")
    assert state.state == "decoded"
    with pytest.raises(ValueError):
        state.admit({"outcome": "assignment_found"})
    assert state.response(3, H, "assignment_found")
    with pytest.raises(ValueError):
        state.admit({"outcome": "verified_solution", "admitted": False})


def test_partial_batches_and_late_results_keep_whole_debit():
    from oeis_learn.synthesis.state import BatchState, analysis_identity

    ledger = BudgetLedger(10)
    batch = BatchState(("a", "b"), 3, H, ledger, batch_id="batch", debit=7)
    batch.start_candidate("a")
    assert batch.response("a", 3, H, "assignment_found")
    states = batch.close("interrupted")
    assert states == {"a": "provisional", "b": "unstarted"}
    assert ledger.remaining == 3
    assert not batch.response("a", 3, H, "assignment_found")
    assert analysis_identity(H, H, J) != analysis_identity(H, H, H)
