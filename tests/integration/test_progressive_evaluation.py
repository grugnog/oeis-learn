"""Actual CPU checkpoint, canonical model sampler and packaged native scorer."""

import pytest

from oeis_learn.fresh.bigint_config import resolve
from oeis_learn.generalization.progressive.baselines import load_random
from oeis_learn.generalization.progressive.contracts import reference
from oeis_learn.pipeline_validation.progressive_evaluation import generate, score


def test_real_cpu_native_generate_freeze_then_score(tmp_path, fitting_client):
    import torch

    recipe = resolve({"version": "bigint-bootstrap/v1"})
    e, d, _ = load_random(recipe, seed=13)
    torch.save({"encoder": e.state_dict(), "decoder": d.state_dict()}, tmp_path / "weights.pt")
    truth = [
        {
            "id": str(j),
            "indices": list(map(str, range(100))),
            "values": [str(i + j) for i in range(100)],
        }
        for j in range(2)
    ]
    prompts = [
        {"id": r["id"], "indices": r["indices"][:20], "values": r["values"][:20]} for r in truth
    ]
    candidate = generate(
        tmp_path,
        "generated",
        prompts,
        recipe,
        client=fitting_client,
        checkpoint=reference(tmp_path, "weights.pt"),
        seconds=10,
    )
    before = (tmp_path / candidate["path"]).read_bytes()
    result = score(tmp_path, candidate, truth, client=fitting_client, output="scored", seconds=30)
    assert result["denominator"] == 2
    assert set(result["outcomes"]) == {"0", "1"}
    assert (tmp_path / candidate["path"]).read_bytes() == before
    assert result["claim"] == "finite_agreement_only"
    with (tmp_path / candidate["path"]).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError):
        score(tmp_path, candidate, truth, client=fitting_client, output="bad")


def test_generator_rejects_truth_before_model_loading(tmp_path):
    with pytest.raises(ValueError, match="closed visible"):
        generate(
            tmp_path,
            "out",
            [{"id": "x", "indices": ["0"] * 20, "values": ["0"] * 20, "witness": "secret"}],
            {},
            client=None,
        )
    assert not (tmp_path / "out").exists()


def test_conventional_native_fits_and_hidden_reference_scoring(tmp_path, fitting_client):
    recipe = resolve({"version": "bigint-bootstrap/v1"})
    truth = [
        {
            "id": str(j),
            "indices": list(map(str, range(100))),
            "values": [str(i + j) for i in range(100)],
        }
        for j in range(2)
    ]
    prompts = [
        {"id": r["id"], "indices": r["indices"][:20], "values": r["values"][:20]} for r in truth
    ]
    ref = generate(
        tmp_path,
        "conventional",
        prompts,
        recipe,
        client=fitting_client,
        model_control="random",
        seed=13,
        methods=("polynomial",),
        seconds=30,
    )
    result = score(tmp_path, ref, truth, client=fitting_client, output="scored")
    assert all(row["polynomial"]["success"] for row in result["outcomes"].values())
    assert all(row["polynomial"]["complete"] for row in result["outcomes"].values())


def test_scorer_rejects_unsealed_or_mutated_candidate_bytes(tmp_path):
    from oeis_learn.generalization.progressive.contracts import atomic_write

    atomic_write(tmp_path / "fake.json", {"version": "discovery-candidates/v1"})
    with pytest.raises(ValueError, match="sealed generator"):
        score(tmp_path, reference(tmp_path, "fake.json"), [], client=None, output="out")
    assert not (tmp_path / "out").exists()


def test_component_generation_uses_committed_cut_and_records_scoring_cost(tmp_path, monkeypatch):
    """Real CPU model scoring with a controlled portfolio boundary; no native claim."""
    from tests.unit.test_progressive_component_mining import sources
    from oeis_learn.generalization.progressive.component_mining import LibraryStore
    from oeis_learn.generalization.progressive.component_search import search_input
    from oeis_learn.generalization.progressive.contracts import load
    import oeis_learn.fresh.bigint_diagnostics as driver

    store = LibraryStore(tmp_path, "library", purpose="acceptance", track="synthetic_strict")
    bundle = search_input(store, store.publish(sources(tmp_path)))
    seen = []

    def portfolio(decoder, memory, prompt, recipe, client, **kwargs):
        import time

        proposal = kwargs["component_proposals"]
        seen.append(proposal(4, min(kwargs["deadline"], time.monotonic() + 60)))
        assert kwargs["attempt_count"] == 8
        return dict(
            raw=None,
            rescue=None,
            equal_budget=None,
            arm_evidence={
                k: dict(complete=False, reason="DIAGNOSTIC_ONLY")
                for k in ("raw", "rescue", "equal_budget")
            },
        )

    monkeypatch.setattr(driver, "portfolio", portfolio)
    recipe = resolve({"version": "bigint-bootstrap/v1"})
    prompts = [
        dict(
            id="heldout",
            indices=list(map(str, range(20))),
            values=[str(2 * i + 1) for i in range(20)],
        )
    ]
    ref = generate(
        tmp_path,
        "generated",
        prompts,
        recipe,
        client=None,
        model_control="random",
        seed=13,
        methods=("correct",),
        component_library=bundle,
    )
    result = load(tmp_path / ref["path"])
    cost = result["records"]["heldout"]["costs"]["components"]
    assert seen[0]["complete"] and cost["scoring_tokens"] > 0
    assert cost["library"] == bundle["library"]["identity"]
    assert cost["discovery_work"] == bundle["discovery_work"]
    assert cost["commit"] == bundle["commit"]
    with pytest.raises(ValueError, match="one ordinary"):
        generate(tmp_path, "bad", prompts, recipe, client=None, component_library=bundle)
