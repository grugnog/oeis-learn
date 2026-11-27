import pytest
from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.generalization.progressive.graph import velocity_look
from oeis_learn.pipeline_validation.progressive_evaluation import velocity_outcome


def test_profiles_and_final_selected_outcome_have_separate_identities():
    cells = [dict(task=str(i), group=str(i // 2), stratum="oeis") for i in range(6)]
    retain = [dict(task="ret", group="retention", stratum="retention")]
    args = dict(
        registration=digest("study"), method="00", seed=1, development=cells, retention=retain
    )
    cheap = [velocity_look(**args, look=k, profile="common") for k in (0, 1000000, 4000000)]
    assert all(r["members"] == cells[::2] and r["attempts"] == 1 for r in cheap)
    full = velocity_look(**args, look=4000000, profile="full")
    assert full["attempts"] == 8 and full["members"] == cells + retain
    assert full["identity"] != cheap[-1]["identity"]
    with pytest.raises(ValueError):
        velocity_look(**args, look=1000000, profile="full")
    evidence = dict(case="0", outcome_ref=digest("score"), choice_ref=digest("selected"),
                    scorer_ref=digest("independent reference"), truth_ref=digest("exact indexed truth"))
    observation = velocity_outcome(selected_exact=True, closed=False, complete=False, evidence=evidence)
    assert observation["status"] == "solve" and not observation["closed"]
    receipt = observation["receipt"]
    assert all(receipt[k] == v for k, v in evidence.items()) and receipt["status"] == observation["status"]
    assert velocity_outcome(selected_exact=None, closed=True, complete=True)["status"] == "miss"
    assert (
        velocity_outcome(selected_exact=None, closed=False, complete=False)["status"]
        == "unresolved"
    )


def test_shuffled_probe_changes_encoder_values_and_keeps_true_fitting_and_scoring(tmp_path, monkeypatch):
    from oeis_learn.fresh import bigint_diagnostics
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.generalization.progressive.program_search_bank import replay_source
    from oeis_learn.pipeline_validation import progressive_evaluation as evaluation

    xs = list(range(-7, 93))
    truth = [dict(id="a", indices=list(map(str, xs)), values=list(map(str, xs))),
             dict(id="b", indices=list(map(str, xs)), values=[str(n+1) for n in xs])]
    prompts = [dict(id=r["id"], indices=r["indices"][:20], values=r["values"][:20]) for r in truth]
    cells = [dict(task=p["id"], group=p["id"], stratum="oeis") for p in prompts]
    coords = dict(registration=digest("study"), method="00", seed=29, development=cells, retention=[])
    common = velocity_look(**coords, look=4_000_000, profile="common")
    full = velocity_look(**coords, look=4_000_000, profile="full")
    probe = evaluation.velocity_probe(common, prompts, mode="shuffled", inference_seed=57)
    assert probe["donors"] == {"a": "b", "b": "a"}
    assert probe != evaluation.velocity_probe(full, prompts, mode="shuffled", inference_seed=57)
    captured = []
    encoder_values = []
    baseline = evaluation.load_baseline
    def traced_baseline(*args, **kwargs):
        encoder, decoder, counts = baseline(*args, **kwargs)
        forward = encoder.forward_from_sequences
        def traced(rows, **kw):
            encoder_values.append(rows)
            return forward(rows, **kw)
        monkeypatch.setattr(encoder, "forward_from_sequences", traced)
        return encoder, decoder, counts
    monkeypatch.setattr(evaluation, "load_baseline", traced_baseline)

    expected_attempts = 1
    def portfolio(decoder, memory, recipient, *args, **kwargs):
        captured.append(recipient)
        assert kwargs["attempt_count"] == expected_attempts
        return dict(raw=None, rescue=None, equal_budget=None,
                    arm_evidence={k: {"complete": True, "reason": "EMPTY_PORTFOLIO"} for k in ("raw", "rescue", "equal_budget")})
    monkeypatch.setattr(bigint_diagnostics, "portfolio", portfolio)
    recipe = resolve({"version":"bigint-bootstrap/v1"})
    artifact = evaluation.generate(tmp_path, "generated", prompts, recipe, client=None, model_control="random",
                                   seed=29, methods=("correct",), registered_probe=probe)
    body = load(tmp_path / artifact["path"])
    assert body["clock"]["velocity_probe"] == probe["identity"] and captured == prompts
    with pytest.raises(ValueError, match="another probe"):
        evaluation.score_velocity(tmp_path, artifact, truth, client=None, output="wrong-profile",
                                  probe=evaluation.velocity_probe(full, prompts, mode="shuffled", inference_seed=57))
    scored = evaluation.score_velocity(tmp_path, artifact, truth, client=None, output="scored", probe=probe)
    assert set(scored["outcomes"]) == {"a", "b"}
    assert encoder_values == [[list(map(int, prompts[1]["values"]))], [list(map(int, prompts[0]["values"]))]]
    assert replay_source("(program (return $index))", captured[0], truth[0]) == "solve"
    assert replay_source("(program (return $index))", captured[1], truth[1]) == "prefix_mismatch"
    expected_attempts = 8
    endpoint = evaluation.velocity_probe(full, prompts, mode="shuffled", inference_seed=57)
    eight = evaluation.generate(tmp_path, "endpoint", prompts, recipe, client=None, model_control="random",
                                seed=29, methods=("correct",), registered_probe=endpoint)
    with pytest.raises(ValueError, match="another probe"):
        evaluation.score_velocity(tmp_path, eight, truth, client=None, output="eight-as-one", probe=probe)
    with pytest.raises(ValueError):
        evaluation.velocity_probe(common, [prompts[0], dict(prompts[1], values=prompts[0]["values"])],
                                  mode="shuffled", inference_seed=57)
