"""Actual CPU proposer/learner/native scorer diagnostic, never research authority."""

import torch

from oeis_learn.fresh.bigint_config import resolve
from oeis_learn.generalization.budgets import Ledger
from oeis_learn.generalization.progressive.contracts import digest, reference
from oeis_learn.generalization.progressive.models import resolve_profile
from oeis_learn.generalization.progressive.trials import TrialJournal
from oeis_learn.generalization.progressive.tuning import propose
from oeis_learn.pipeline_validation.progressive_evaluation import generate, score
from oeis_learn.pipeline_validation.progressive_training import PolicyUpdate, resolve_update_policy


def test_real_cpu_proposal_learning_frozen_generation_and_native_score(tmp_path, fitting_client):
    j = TrialJournal(tmp_path / "journal", study=digest("diagnostic-tuning"))
    proposal = propose(j, 0)
    assert propose(TrialJournal(tmp_path / "journal", study=j.binding["study"]), 0) == proposal
    recipe = resolve({"version": "bigint-bootstrap/v1"})
    p = resolve_profile(recipe, seed=proposal["seed"])
    runner = PolicyUpdate(
        p, resolve_update_policy(p, schedule_horizon_tokens=4000000), device="cpu"
    )
    from oeis_learn.decoder.bigint_codec import BOS_ID, encode_body

    ids = [BOS_ID, *encode_body("(program (return (add $index (int 1))))")]
    tokens = torch.tensor([ids] * 16)
    j.dispatch(0)
    result, ledger = runner.update(
        [list(range(1, 21))] * 16,
        tokens,
        ledger=Ledger(4000000).reserve("diagnostic-update", 16 * (len(ids) - 1)),
    )
    assert ledger.updates == 1 and result["loss"] > 0
    torch.save(
        {"encoder": runner.encoder.state_dict(), "decoder": runner.decoder.state_dict()},
        tmp_path / "weights.pt",
    )
    truth = [
        {
            "id": str(k),
            "indices": list(map(str, range(100))),
            "values": [str(i + k) for i in range(100)],
        }
        for k in range(2)
    ]
    prompts = [
        {"id": r["id"], "indices": r["indices"][:20], "values": r["values"][:20]} for r in truth
    ]
    candidates = generate(
        tmp_path,
        "generated",
        prompts,
        recipe,
        client=fitting_client,
        checkpoint=reference(tmp_path, "weights.pt"),
        seconds=10,
        methods=("correct",),
    )
    scored = score(tmp_path, candidates, truth, client=fitting_client, output="scored", seconds=30)
    value = sum(r["correct"]["success"] for r in scored["outcomes"].values()) / 2
    j.tell(
        0,
        {
            "status": "PARTIAL",
            "tokens": ledger.consumed,
            "elapsed_us": 1,
            "value": value,
            "role": "development",
        },
    )
    assert not j.equal_token_eligible(target=4000000)
