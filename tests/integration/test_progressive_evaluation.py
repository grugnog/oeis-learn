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


def test_expired_diagnostic_fit_remains_charged_and_unresolved(tmp_path):
    import time
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.generalization.progressive.contracts import load
    class Budget:
        deadline=time.monotonic()+30
        used=[]
        def consume(self,**cost):self.used.append(cost)
    class Client:
        def fit(self,*args,**kw):raise TimeoutError('late native work remains charged')
    budget=Budget()
    recipe=resolve_profile(resolve({'version':'bigint-bootstrap/v1'}),shape='smaller_width',seed=13,dropout=0)
    prompts=[dict(id='expired',indices=list(map(str,range(20))),values=list(map(str,range(20))))]
    candidate=generate(tmp_path,'expired',prompts,recipe,client=Client(),model_control='random',seed=13,methods=('correct',),forced={'expired':['(program (return $index))']},budget=budget)
    row=load(tmp_path/candidate['path'])['records']['expired']
    assert budget.used==[dict(draws=1,attempts=1,rejections=1)]
    assert row['selections']['equal_budget'] is None
    assert row['arm_evidence']['equal_budget']['complete'] is False
    assert row['costs']['diagnostic']['expired_fits']==1
    assert row['costs']['diagnostic']['autonomous_discoveries']==0


def test_selector_selection_time_spends_the_live_search_allowance(monkeypatch):
    """Use both installed choice and scoring roles with a deterministic slow selection."""
    from types import SimpleNamespace
    from oeis_learn.fresh import bigint_diagnostics
    from oeis_learn.fresh.synthesis_policy import select_visible
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.pipeline_validation import progressive_round028_selector as selector
    from oeis_learn.synthesis.admission import VisiblePrompt, source_identity
    from oeis_learn.synthesis.contracts import ResourceProfile
    from oeis_learn.synthesis.reference import evaluate

    import time

    clock = [time.monotonic()]
    monkeypatch.setattr(selector, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    truth = dict(
        id="private-case", indices=list(map(str, range(100))), values=["0"] * 20 + ["1"] * 80
    )
    prompt = dict(truth, indices=truth["indices"][:20], values=truth["values"][:20])
    visible = VisiblePrompt(tuple(prompt["indices"]), tuple(prompt["values"]))
    bodies = [
        "(program (return (int 0)))",
        "(program (return (select (lt $index (int 20)) (int 0) (int 1))))",
    ]
    model = selector.fit_ranker(
        [dict(case="training", values=["0"] * 20, positive=bodies[1], negative=bodies[0])]
    )
    bank = [
        dict(
            prompt_sha256=visible.sha256,
            selected=select_visible(
                [
                    dict(
                        source=b,
                        source_sha256=source_identity(b),
                        verified=True,
                        visible_count=20,
                        cost=0.0,
                    )
                ]
            ),
        )
        for b in bodies
    ]

    def portfolio(*args, **kwargs):
        clock[0] += 59
        return dict(
            selector_candidates={"equal_budget": bank},
            equal_budget=bank[0],
            details=[{}] * 8,
            arm_evidence={"equal_budget": {"complete": True}},
        )

    original = selector.choose

    def slow(*args):
        clock[0] += 2
        return original(*args)

    monkeypatch.setattr(bigint_diagnostics, "portfolio", portfolio)
    monkeypatch.setattr(selector, "choose", slow)
    client = SimpleNamespace(
        limits=ResourceProfile(),
        execute=lambda b, indices, **kw: dict(
            outcome="ok", values=list(evaluate(b, indices).values)
        ),
    )
    rows = {}
    for mode in ("bank", "live_incumbent", "live_learned", "reference"):
        packet = seal(
            dict(
                block=mode,
                prompt=prompt,
                mode=mode,
                proposer="template_only",
                recipe={"model": {}},
                weights=None,
                seed=1,
                model=model,
                shuffled=["1"] * 20,
            )
        )
        value = selector.selector_choice(packet, client)
        if mode == "live_learned":
            assert value["elapsed_us"] == 61_000_000
        rows.update(
            {
                k: dict(chosen=v, complete=value["complete"], comparable=value["comparable"][k])
                for k, v in value["choices"].items()
            }
        )
    result = selector.selector_score(dict(truth=truth, selections=rows), client)["outcomes"]
    assert (result["incumbent"], result["learned"], result["live_learned"]) == ("F", "S", "U")


def test_selector_common_replication_gate_and_design_tampering():
    from oeis_learn.generalization.progressive.contracts import digest, seal
    from oeis_learn.generalization.progressive.selector_replication import (
        formal_result,
        freeze_panel,
        joint_cells,
    )

    roster = [
        dict(task=f"{s}-{g}", stratum=s, group=f"family-{g}")
        for s in ("composition", "unseen_structure", "parameter", "oeis", "retention")
        for g in range(16)
    ]
    design = freeze_panel(digest("registered-selector"), roster, 7200)
    models = [
        dict(
            seed=seed,
            model=digest([i, "model"]),
            stream=digest([i, "stream"]),
            training=digest([i, "rows"]),
        )
        for i, seed in enumerate(design["seeds"])
    ]
    scores = {r["task"]: dict(reference="F", live_learned="S") for r in roster}
    cells = joint_cells(roster, [scores, scores])
    result = formal_result(design, models, cells)
    assert result["verdict"] == "CONFIRMED_GAIN" and len(design["links"]) == 16
    observed = result["metrics"]["cells"]["candidate"][str(design["seeds"][0])]
    assert "oeis" in observed and "parameter" in observed
    for changed in (
        dict(links=[]),
        dict(comparisons=2),
        dict(resamples=100),
        dict(sizes=design["sizes"][1:]),
    ):
        with pytest.raises(ValueError):
            formal_result(seal(dict(design, **changed)), models, cells)
    with pytest.raises(ValueError, match="independent"):
        formal_result(design, [models[0], dict(models[1], stream=models[0]["stream"])], cells)
    with pytest.raises(ValueError):
        joint_cells(roster, [scores, {}])
