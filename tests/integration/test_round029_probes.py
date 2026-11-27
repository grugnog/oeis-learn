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
    truth = [dict(id="b", indices=list(map(str, xs)), values=list(map(str, xs))),
             dict(id="a", indices=list(map(str, xs)), values=[str(n+1) for n in xs])]
    prompts = [dict(id=r["id"], indices=r["indices"][:20], values=r["values"][:20]) for r in truth]
    cells = [dict(task=p["id"], group=p["id"], stratum="oeis") for p in prompts]
    coords = dict(registration=digest("study"), method="00", seed=29, development=cells, retention=[])
    common = velocity_look(**coords, look=4_000_000, profile="common")
    full = velocity_look(**coords, look=4_000_000, profile="full")
    import torch
    from oeis_learn.generalization.progressive.contracts import reference, seal
    recipe = resolve({"version": "bigint-bootstrap/v1"})
    encoder, decoder, _ = evaluation.load_baseline("random", recipe, seed=29)
    initial = bigint_diagnostics.model_snapshot(encoder, decoder)
    parameter = next(encoder.parameters())
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    parameter.sum().backward()
    optimizer.step()
    trained = bigint_diagnostics.model_snapshot(encoder, decoder)
    assert trained != initial
    def save_weights(name):
        path = tmp_path / name
        torch.save(dict(encoder=encoder.state_dict(), decoder=decoder.state_dict()), path)
        return reference(tmp_path, path)
    weights = save_weights("trained.pt")
    model = evaluation.model_receipt(common, profile=digest(recipe), weights=trained,
                                     checkpoint=weights["sha256"], origin=digest("qualified cut"))
    def view(look, mode="shuffled", model=model):
        return evaluation.velocity_probe(look, prompts, mode=mode, inference_seed=57, model=model)
    probe = view(common)
    assert probe["donors"] == {"a": "b", "b": "a"}
    assert probe != view(full)
    with pytest.raises(ValueError, match="checkpoint|model"):
        evaluation.generate(tmp_path, "missing", prompts, recipe, client=None, model_control="random",
                            seed=29, methods=("correct",), registered_probe=probe)
    with torch.no_grad():
        parameter.add_(1)
    wrong = save_weights("other.pt")
    with pytest.raises(ValueError, match="checkpoint|model"):
        evaluation.generate(tmp_path, "wrong", prompts, recipe, client=None,
                            checkpoint=wrong, methods=("correct",), registered_probe=probe)
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
    artifact = evaluation.generate(tmp_path, "generated", prompts, recipe, client=None, model_control="earlier_checkpoint",
                                   checkpoint=weights, reference_recipe=recipe, methods=("correct",), registered_probe=probe)
    body = load(tmp_path / artifact["path"])
    assert body["clock"]["velocity_probe"] == probe["identity"] and captured == prompts
    with pytest.raises(ValueError, match="another probe"):
        evaluation.score_velocity(tmp_path, artifact, truth, client=None, output="wrong-profile",
                                  probe=view(full))
    scored = evaluation.score_velocity(tmp_path, artifact, truth, client=None, output="scored", probe=probe)
    assert set(scored["outcomes"]) == {"a", "b"}
    assert encoder_values == [[list(map(int, prompts[1]["values"]))], [list(map(int, prompts[0]["values"]))]]
    assert replay_source("(program (return $index))", captured[0], truth[0]) == "solve"
    assert replay_source("(program (return $index))", captured[1], truth[1]) == "prefix_mismatch"
    expected_attempts = 8
    endpoint = view(full)
    eight = evaluation.generate(tmp_path, "endpoint", prompts, recipe, client=None, model_control="earlier_checkpoint",
                                checkpoint=weights, reference_recipe=recipe, methods=("correct",), registered_probe=endpoint)
    with pytest.raises(ValueError, match="another probe"):
        evaluation.score_velocity(tmp_path, eight, truth, client=None, output="eight-as-one", probe=probe)
    with pytest.raises(ValueError):
        evaluation.velocity_probe(common, [prompts[0], dict(prompts[1], values=prompts[0]["values"])],
                                  mode="shuffled", inference_seed=57, model=model)

    from oeis_learn.generalization.progressive.contracts import atomic_write
    path = tmp_path / "substituted-candidates.json"
    atomic_write(path, seal(dict(body, model=initial)))
    with pytest.raises(ValueError, match="model"):
        evaluation.score_velocity(tmp_path, reference(tmp_path, path), truth, client=None, output="substituted", probe=probe)

    zero = velocity_look(**coords, look=0, profile="common")
    zero_model = evaluation.model_receipt(zero, profile=digest(recipe), weights=initial,
                                          checkpoint=None, origin=digest("prepared random start"))
    expected_attempts = 1
    start = evaluation.generate(tmp_path, "start", prompts, recipe, client=None, model_control="random",
                                seed=29, methods=("correct",), registered_probe=view(zero, model=zero_model))
    assert load(tmp_path / start["path"])["model"] == initial
    incumbent = velocity_look(**dict(coords, method="D"), look=4_000_000, profile="full")
    from oeis_learn.pipeline_validation import progressive_velocity_donor
    donor = dict(method="D", look=None, inference=True, weights=weights["sha256"], recipe=digest(recipe),
                 registration=digest("donor"), lineage=digest("D"))
    monkeypatch.setattr(progressive_velocity_donor, "inventory", lambda _: dict(cuts=[donor]))
    incumbent_model = evaluation.donor_probe_model(tmp_path, incumbent, recipe, weights)
    with pytest.raises(ValueError, match="donor weights"):
        evaluation.donor_probe_model(tmp_path, incumbent, recipe, wrong)
    expected_attempts = 8
    d_probe = view(incumbent, mode="true", model=incumbent_model)
    d = evaluation.generate(tmp_path, "incumbent", prompts, recipe, client=None, model_control="retained_D",
                            checkpoint=weights, methods=("correct",), registered_probe=d_probe)
    assert evaluation.score_velocity(tmp_path, d, truth, client=None, output="d-score", probe=d_probe)
    with pytest.raises(ValueError, match="checkpoint|model"):
        evaluation.generate(tmp_path, "other-d", prompts, recipe, client=None, checkpoint=wrong,
                            methods=("correct",), registered_probe=d_probe)


def test_registered_derangement_uses_each_donor_once_within_its_own_stratum():
    from oeis_learn.pipeline_validation.progressive_evaluation import model_receipt, velocity_probe

    cells = [
        dict(task=str(i), group=str(i), stratum=s)
        for i, s in enumerate(("oeis", "parameter", "oeis", "parameter"))
    ]
    look = velocity_look(registration=digest("study"), method="00", seed=29, look=0,
                         profile="common", development=cells, retention=[])
    prompts = [dict(id=r["task"], indices=list(map(str, range(20))), values=[r["task"]] * 20)
               for r in cells]
    origin = model_receipt(look, profile=digest("recipe"), weights=digest("initial"),
                           checkpoint=None, origin=digest("prepared start"))
    probe = velocity_probe(look, prompts, mode="shuffled", inference_seed=57, model=origin)
    assert probe["donors"] == {"0": "2", "2": "0", "1": "3", "3": "1"}
    prompts[2]["values"] = prompts[0]["values"]
    with pytest.raises(ValueError, match="stratum oeis"):
        velocity_probe(look, prompts, mode="shuffled", inference_seed=57, model=origin)


@pytest.mark.parametrize("method,expected_calls", [("template", 8), ("polynomial", 1), ("linear_recurrence", 1)])
def test_registered_conventional_opportunity_and_selected_scoring(tmp_path, method, expected_calls):
    import time
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.pipeline_validation import progressive_evaluation as ev
    from oeis_learn.synthesis.admission import source_identity
    from oeis_learn.synthesis.contracts import ResourceProfile
    from oeis_learn.synthesis.reference import evaluate

    class Client:
        limits = ResourceProfile()
        calls = []

        def fit(self, source, visible, *, policy, deadline, seed):
            assert len(visible.values) == len(visible.indices) == 20
            assert 0 < deadline - time.monotonic() <= 4 and type(seed) is int
            self.calls.append(source)
            accepted = []
            if method == "polynomial":
                assert list(evaluate(source, visible.indices).values) == list(visible.values)
                accepted.append(
                    dict(
                        source=source,
                        transformed_source=source,
                        source_sha256=source_identity(source),
                        original_source_sha256=source_identity(source),
                        assignment={},
                        verification=dict(prompt_sha256=visible.sha256),
                    )
                )
            return dict(complete=True, outcome="ok", admitted=accepted)

        def execute(self, source, indices, *, deadline):
            result = evaluate(source, indices, deadline=deadline)
            return dict(values=list(result.values), outcome=result.outcome)

    truth = dict(
        id="p", indices=list(map(str, range(100))), values=[str(i + 1) for i in range(100)]
    )
    prompt = dict(id="p", indices=truth["indices"][:20], values=truth["values"][:20])
    look = velocity_look(
        registration=digest("study"),
        method=method,
        seed=29,
        look=4_000_000,
        profile="full",
        development=[dict(task="p", group="f", stratum="oeis")],
        retention=[],
    )
    probe = ev.velocity_probe(look, [prompt], mode="true", inference_seed=57)
    client = Client()
    measurements, elapsed = {}, {}
    ref = ev.generate(
        tmp_path,
        "generated",
        [prompt],
        resolve({"version": "bigint-bootstrap/v1"}),
        client=client,
        methods=(method,),
        registered_probe=probe,
        measurements=measurements,
        case_elapsed_us=elapsed,
        case_boundary_dir=tmp_path / "boundaries",
    )
    body = load(tmp_path / ref["path"])
    clock = body["clock"]["clocks"][0]
    assert len(client.calls) == clock["started"] == expected_calls and clock["allowed"] == 8
    assert elapsed == {"p": clock["elapsed_us"]} and "publish_candidates_us" in measurements
    assert set(body["records"]["p"]["selections"]) == {"equal_budget"}
    scored = ev.score_velocity(tmp_path, ref, [truth], client=client, output="scored", probe=probe)
    assert scored["outcomes"]["p"]["equal_budget"]["success"] == (method == "polynomial")
    from oeis_learn.generalization.progressive.contracts import atomic_write, reference, seal
    # A selected answer remains independently testable when another fit is unfinished.
    record = body["records"]["p"]
    record["arm_evidence"]["equal_budget"].update(complete=False, reason="unfinished")
    path = tmp_path / "partial-candidates.json"
    atomic_write(path, seal(body))
    partial = reference(tmp_path, path)
    for name, target in (("match", truth), ("mismatch", dict(truth, values=truth["values"][:-1] + ["-1"]))):
        score = ev.score_velocity(tmp_path, partial, [target], client=client, output=name, probe=probe)
        outcome = score["outcomes"]["p"]["equal_budget"]
        expected = ("solve" if name == "match" else "miss") if method == "polynomial" else "unresolved"
        assert outcome["status"] == expected and not outcome["complete"] and not outcome["closed"]
        assert outcome["receipt"]["truth_ref"] == digest(target)
        from oeis_learn.generalization.progressive.velocity_metrics import contrast
        row = dict(case="p", family="f", stratum="oeis", observations={"selected": outcome})
        bound = contrast([row], {"selected": 1}, stratum="oeis", looks={"selected": probe})
        assert bound["lower_pp"] == (100 if expected == "solve" else 0)
        assert bound["upper_pp"] == (0 if expected == "miss" else 100)

    with pytest.raises(ValueError):
        ev.velocity_probe(look, [prompt], mode="shuffled", inference_seed=57)
    with pytest.raises(ValueError):
        ev.generate(
            tmp_path,
            "incorrect",
            [prompt],
            {},
            client=client,
            methods=("correct",),
            registered_probe=probe,
        )


def test_registered_polynomial_uses_installed_native_fit_and_independent_score(
    tmp_path, fitting_client
):
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.pipeline_validation import progressive_evaluation as ev

    truth = dict(
        id="offset", indices=list(map(str, range(-5, 95))), values=list(map(str, range(100)))
    )
    prompt = dict(id="offset", indices=truth["indices"][:20], values=truth["values"][:20])
    look = velocity_look(
        registration=digest("study"),
        method="polynomial",
        seed=29,
        look=4_000_000,
        profile="full",
        development=[dict(task="offset", group="f", stratum="oeis")],
        retention=[],
    )
    probe = ev.velocity_probe(look, [prompt], mode="true", inference_seed=57)
    ref = ev.generate(
        tmp_path,
        "generated",
        [prompt],
        resolve({"version": "bigint-bootstrap/v1"}),
        client=fitting_client,
        methods=("polynomial",),
        registered_probe=probe,
    )
    scored = ev.score_velocity(
        tmp_path, ref, [truth], client=fitting_client, output="score", probe=probe
    )
    assert scored["outcomes"]["offset"]["equal_budget"]["success"]
