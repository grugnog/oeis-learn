"""Run decisions and candidate identity at the installed pilot boundaries."""

import pytest

from oeis_learn.generalization.progressive.contracts import digest, reference, seal
from oeis_learn.pipeline_validation import progressive_evaluation as evaluation
from oeis_learn.pipeline_validation import progressive_round028_pilot as pilot
from oeis_learn.pipeline_validation.pilot_clocks import DESCRIPTIVE, CORRECT_FULL, clock_record


def test_decoder_receives_registered_seed(tmp_path, monkeypatch):
    import torch
    from oeis_learn.decoder import bigint_generation
    from oeis_learn.generalization.progressive.models import construct
    from oeis_learn.synthesis.contracts import identity
    from tests.unit.test_progressive_training_policy import profile

    recipe = profile()
    encoder, decoder, _ = construct(recipe, device="cpu")
    checkpoint = tmp_path / "weights.pt"
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, checkpoint)
    row = dict(id="example", indices=list(map(str, range(20))), values=list(map(str, range(20))))

    class Captured(Exception):
        pass

    def capture(decoder, requests, **kwargs):
        assert [r.seed for r in requests] == [17 + int(identity(row["id"])[7:14], 16)]
        raise Captured()

    monkeypatch.setattr(bigint_generation, "sample_batch", capture)
    with pytest.raises(Captured):
        evaluation.generate(
            tmp_path,
            "out",
            [row],
            recipe,
            client=object(),
            model_control="earlier_checkpoint",
            reference_recipe=recipe,
            device="cpu",
            checkpoint=reference(tmp_path, checkpoint),
            methods=("correct",),
            evaluation_policy=DESCRIPTIVE,
            inference_seed=17,
        )


def test_clock_summary_cannot_upgrade_partial_case():
    model, candidate = digest("w"), digest("c")
    prompts = [dict(id="x")]
    clock = clock_record(
        prompt="x",
        mode="correct",
        snapshot=model,
        start=0,
        scope_deadline=59,
        finish=59,
        expected=8,
        started=1,
        terminal=1,
        policy=CORRECT_FULL,
    )
    gen = dict(
        checkpoint=model,
        recipe=digest({}),
        prompts=digest(prompts),
        model=model,
        records={"x": {}},
        clock=seal(dict(policy=CORRECT_FULL, work=model, source=digest({}), expected_portfolios=1,
                        missing_portfolios=0, clocks=[clock], cost_complete=True)),
    )
    score = dict(
        candidates=candidate,
        denominator=1,
        outcomes={
            "x": {
                "equal_budget": {
                    "success": False,
                    "complete": False,
                    "scoring": {"complete": False},
                }
            }
        },
    )
    with pytest.raises(ValueError, match="clock"):
        pilot.score_projection(gen, score, candidate, prompts, {"sha256": model}, {}, CORRECT_FULL)
    # All eight draws can finish early even when the parent has clipped the last second.
    short = clock_record(prompt="x", mode="correct", snapshot=model, start=0, scope_deadline=59,
                         finish=20, expected=8, started=8, terminal=8, policy=CORRECT_FULL)
    assert short["cost_complete"]
    gen["clock"] = seal(dict(gen["clock"], clocks=[short], inference_seed=71))
    _, comparable = pilot.score_projection(
        gen, score, candidate, prompts, {"sha256": model}, {}, CORRECT_FULL, inference_seed=71
    )
    assert not comparable


def test_resident_scorer_finishes_an_admitted_choice_with_both_engine_allowances(tmp_path, monkeypatch):
    from oeis_learn.fresh import bigint_diagnostics
    from oeis_learn.pipeline_validation import progressive_pilot_worker as worker
    from oeis_learn.generalization.progressive.contracts import atomic_write, load
    from tests.unit.test_bigint_diagnostic_evidence import Client, selection
    import time

    # The second choice starts just before the batch envelope closes. Each engine
    # needs 29 of its 30 seconds to check the 100 terms of that already sealed choice.
    now = [100.0]
    monkeypatch.setattr(time, "monotonic", lambda: now[0])
    durations = []
    reference_evaluate = bigint_diagnostics.reference_evaluate

    class SlowClient(Client):
        def execute(self, source, indices, *, deadline):
            durations.append(deadline - now[0])
            now[0] += 29
            return super().execute(source, indices)

    def independent(source, indices, limits, *, deadline):
        durations.append(deadline - now[0])
        now[0] += 29
        return reference_evaluate(source, indices, limits, deadline=deadline)

    monkeypatch.setattr(bigint_diagnostics, "reference_evaluate", independent)
    exact = list(map(str, range(100)))
    truth = [dict(id=key, values=exact, indices=exact) for key in ("first", "last")]
    prompts = [dict(id=r["id"], indices=exact[:20], values=exact[:20]) for r in truth]
    sealed = selection()
    records = {r["id"]: {"selections": {"equal_budget": sealed},
                         "arm_evidence": {"equal_budget": {"complete": True}}} for r in truth}
    path = tmp_path / "inputs/candidates.json"
    atomic_write(path, seal(dict(version="discovery-candidates/v1", checkpoint=digest("weights"),
        model=digest("model"), model_control="earlier_checkpoint", recipe=digest("recipe"),
        methods=["correct"], prompts=digest(prompts), records=records, clock={}, counts={}, elapsed_us=1)))
    data = dict(block="qualify-resident-endpoint", candidates=reference(tmp_path, path), rows=truth, seconds=59)
    scored = worker.evaluate_role(tmp_path, data, SlowClient(), "score")
    outcomes = load(tmp_path / "output" / scored["scores"]["path"])["outcomes"]
    assert durations == [30] * 4
    assert all(v["equal_budget"]["success"] for v in outcomes.values())
    now[0] = 100
    durations.clear()
    with pytest.raises(TimeoutError, match="before opening"):
        worker.evaluate_role(tmp_path, dict(data, seconds=58), SlowClient(), "score")
    assert durations == [30, 30]


@pytest.mark.parametrize("segmented", [False, True])
def test_added_clock_margins_preserve_the_original_resident_acceptance(tmp_path, segmented):
    from oeis_learn.pipeline_validation.preparation_recovery import retain

    reg = {"identity": digest("registered worker")}
    cold = seal(dict(cases=160, cold_attempt=1, candidate=digest("cold choices")))
    old = seal(dict(cases=160, candidate=digest("segmented choices"))) if segmented else cold
    path = tmp_path / "qualification-scale.json"
    retain(path, old)
    acceptance = {"checks": {"resident_scale": old["identity"]}}
    if segmented:
        retain(tmp_path / "resident-measurements.json", seal(dict(
            registration=reg["identity"], original=old["identity"], scale=cold)))
    before = {p: p.read_bytes() for p in tmp_path.glob("*.json")}
    extended = seal(dict(cold, minimum_case_allowance_us=60_000_000, slowest_case_margin_us=1_000_000))
    for _ in range(2):
        assert pilot.retain_resident_scale(tmp_path, reg["identity"], extended) == old["identity"]
        assert pilot.accepted_scale(tmp_path, reg, acceptance) == extended
    assert all(p.read_bytes() == raw for p, raw in before.items())
    with pytest.raises(ValueError, match="changed"):
        pilot.retain_resident_scale(tmp_path, reg["identity"], seal(dict(extended, candidate=digest("different"))))


def test_qualification_retry_keeps_failed_cost_and_original_inputs(tmp_path, monkeypatch):
    from oeis_learn.pipeline_validation import progressive_pilot_recovery as recovery

    calls = []

    class Scorer:
        area = tmp_path

        def __call__(self, role, data, *, seconds, inputs):
            calls.append(data)
            return dict(result="reopened")

    original = seal(dict(block="exposed", candidates=digest("choices")))

    def previous(runner, role, data, inputs):
        assert data["candidates"] == original["candidates"]
        return (tmp_path, 0, 123) if data["block"] == "exposed" else None

    monkeypatch.setattr(recovery, "interrupted", previous)
    for _ in range(2):
        assert recovery.qualification_role(Scorer(), "score", original, seconds=300, inputs=[]) == {
            "result": "reopened"
        }
    assert calls[0] == calls[1] and calls[0]["block"] == "exposed-retry-1"
    from oeis_learn.generalization.progressive.contracts import load

    assert load(tmp_path / "qualification-costs/exposed-score.json")["wall_us"] == 123


def test_unstarted_research_assignment_reopens(tmp_path):
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive.contracts import atomic_write
    from oeis_learn.pipeline_validation.progressive_pilot_recovery import interrupted

    atomic_write(
        tmp_path / "assignments/ordinary-0-search.json", seal({"request": digest("pending")})
    )
    assert (
        interrupted(SimpleNamespace(area=tmp_path), "search", {"block": "ordinary-0"}, []) is None
    )


def test_long_resident_clock_and_telemetry(tmp_path):
    from oeis_learn.pipeline_validation.progressive_pilot_recovery import resident_seconds
    from oeis_learn.tracking.pipeline_profile import Recorder
    from oeis_learn.pipeline_validation.progressive_generator_probe import usage_summary
    import json

    seconds = resident_seconds(160)
    assert seconds >= 160 * 60 + 121 and seconds <= evaluation._maximum_seconds(None)
    rec = Recorder(tmp_path, instrumented=False, planned_seconds=seconds)
    assert rec.sample_interval * (rec.MAX_SAMPLES - 1) >= seconds
    with (tmp_path / "samples.jsonl").open("w") as handle:
        for t in (0, rec.sample_interval, 3 * rec.sample_interval + 1):
            handle.write(json.dumps({"time": t, "gpu": {"gpu_busy_percent": 50}}) + "\n")
    measured = usage_summary(tmp_path, seconds * 1_000_000, interval=rec.sample_interval)
    assert measured["covered_us"] == rec.sample_interval * 1_000_000


def test_endpoint_resident_reads_real_checkpoint_and_closes_all_independent_clocks(
    tmp_path, monkeypatch
):
    from oeis_learn.decoder import bigint_generation
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.generalization.progressive.models import construct
    from oeis_learn.pipeline_validation import progressive_pilot_worker as worker
    from oeis_learn.pipeline_validation import progressive_qualification as qualification
    from oeis_learn.pipeline_validation.progressive_pilot_recovery import resident_seconds
    from oeis_learn.synthesis.contracts import identity
    from tests.unit.test_progressive_training_policy import profile
    import torch

    monkeypatch.setattr(qualification, "require_hip", lambda: None)
    generate = evaluation.generate

    def cpu_generator(*args, **kwargs):
        assert kwargs["device"] == "cuda"
        return generate(*args, **dict(kwargs, device="cpu"))

    monkeypatch.setattr(evaluation, "generate", cpu_generator)
    # A deterministic monotonic clock can exercise a slow cold load and a final
    # valid 59-second portfolio without making the installed recovery suite wait.
    import time
    wall_clock = time.monotonic
    nanos = time.monotonic_ns
    delay = [0]
    monkeypatch.setattr(time, "monotonic", lambda: wall_clock() + delay[0])
    monkeypatch.setattr(time, "monotonic_ns", lambda: nanos() + delay[0] * 1_000_000_000)
    loader = evaluation.load_baseline
    def cold_load(*args, **kwargs):
        result = loader(*args, **kwargs)
        delay[0] += 180
        return result
    monkeypatch.setattr(evaluation, "load_baseline", cold_load)
    draws = []

    def sample(decoder, requests):
        if len(draws) == 159 * 8:
            delay[0] += 59
        draws.extend((row.seed, row.temperature) for row in requests)
        return [
            dict(
                tokens=[],
                source="",
                complete=False,
                generation_batch_size=1,
                generation_seconds=0.0,
            )
            for _ in requests
        ]

    monkeypatch.setattr(bigint_generation, "sample_batch", sample)
    recipe = profile()
    encoder, decoder, _ = construct(recipe, device="cpu")
    weights = tmp_path / "inputs/weights.pt"
    weights.parent.mkdir()
    (tmp_path / "output").mkdir()
    torch.save({"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, weights)
    truth = [
        {
            "id": f"exposed-{i:03d}",
            "indices": list(map(str, range(100))),
            "values": [str(i % 8)] * 100,
        }
        for i in range(160)
    ]
    prompts = [{k: v if k == "id" else v[:20] for k, v in r.items()} for r in truth]
    seconds = resident_seconds(len(prompts))
    weight = reference(tmp_path, weights)
    data = dict(
        block="qualify-resident-endpoint",
        target=4_000_000,
        arm="ordinary",
        seconds=seconds,
        inference_seed=971,
        rows=prompts,
        recipe=recipe,
        weights=weight,
    )
    found = worker.evaluate_role(tmp_path, data, None, "search")
    assert set(found["measurements"]) == {"load_and_identify_checkpoint_us", "publish_candidates_us"}
    assert found["measurements"]["load_and_identify_checkpoint_us"] >= 180_000_000
    assert found["measurements"]["publish_candidates_us"] >= 0
    ref = found["candidates"]
    output = tmp_path / "output"
    candidates = load(output / ref["path"], maximum=128 << 20)
    scored = worker.evaluate_role(
        tmp_path,
        dict(block=data["block"], rows=truth, seconds=seconds, candidates=reference(tmp_path, output / ref["path"])),
        None,
        "score",
    )
    assert set(scored["measurements"]) == {"score_choices_us", "publish_scores_us"}
    scored = scored["scores"]
    result, closed = pilot.score_projection(
        candidates,
        load(output / scored["path"]),
        ref["sha256"],
        prompts,
        weight,
        recipe,
        CORRECT_FULL,
        inference_seed=971,
    )
    assert closed and len(result) == 160 and len(draws) == 160 * 8
    last = candidates["clock"]["clocks"][-1]
    assert 59 <= last["finish"] - last["start"] < 60 and last["cost_complete"]
    assert len(list((output / "case-boundaries").glob("*/choice.json"))) == 160
    for idx, prompt in enumerate(prompts):
        seed = 971 + int(identity(prompt["id"])[7:14], 16)
        assert draws[idx * 8][0] == seed
        assert any(temperature > 0 for _, temperature in draws[idx * 8 : idx * 8 + 8])

    from oeis_learn.pipeline_validation import progressive_pilot_recovery as recovery
    import shutil

    interrupted = tmp_path / "interrupted"
    keep = digest(prompts[0]["id"])[7:]
    shutil.copytree(output / "case-boundaries" / keep, interrupted / "case-boundaries" / keep)
    monkeypatch.setattr(
        recovery,
        "interrupted",
        lambda runner, role, req, inputs: (
            (interrupted, 1, 2) if req["block"] == "exposed" else None
        ),
    )
    calls = []

    def tail(owner, role, request, **options):
        assert request["rows"] == prompts[1:]
        calls.append(request)
        part, unopened, unsealed = recovery.recover_choices(
            tmp_path, output, "tail", request["rows"], weight, recipe, "ordinary", 4_000_000, 971
        )
        assert not unopened and not unsealed
        owner.last_output = part[1].parent
        return {"candidates": reference(owner.last_output, part[1])}

    class Runner:
        area = tmp_path
        __call__ = tail

    for _ in range(2):
        owner = Runner()
        built = recovery.qualification_role(
            owner, "search", seal(dict(data, block="exposed")), seconds=seconds + 120, inputs=[]
        )
        joined = load(owner.last_output / built["candidates"]["path"], maximum=128 << 20)
        assert (
            joined["records"] == candidates["records"]
            and joined["clock"]["clocks"] == candidates["clock"]["clocks"]
        )
        assert load(tmp_path / "qualification-costs/exposed-search.json")["wall_us"] == 2
    assert calls[0] == calls[1]


def test_qualification_retries_an_incomplete_score_without_repeating_search(tmp_path):
    from oeis_learn.generalization.progressive.contracts import atomic_write, load
    from oeis_learn.pipeline_validation.progressive_pilot_qualification import score_choices

    choices = {
        "records": {
            "a": {
                "selections": {"equal_budget": {"selected": {"source_sha256": digest("fixture")}}}
            },
            "b": {"selections": {"equal_budget": None}},
        }
    }
    path = tmp_path / "choices.json"
    atomic_write(path, seal(choices))
    calls = []

    class Scorer:
        area = tmp_path
        phase = "cpu"

        def __call__(self, role, request, *, seconds, inputs):
            assert role == "score" and inputs == [(path, "/work/inputs/candidates.json")]
            ref = request["candidates"]
            assert ref["sha256"] == reference(tmp_path, path)["sha256"]
            calls.append(request["block"])
            complete = request["block"] != "exposed"
            checked = (
                dict(
                    complete=True,
                    production={"outcome": "ok", "coverage": 100},
                    reference={"outcome": "ok", "coverage": 100},
                )
                if complete
                else {"complete": False, "outcome": "not_attempted", "reason": "deadline"}
            )
            result = seal(
                dict(
                    version="discovery-scoring/v1",
                    candidates=ref["sha256"],
                    denominator=2,
                    outcomes={
                        "a": {
                            "equal_budget": {
                                "scoring": checked,
                                "candidate_sha256": digest("fixture"),
                            }
                        },
                        "b": {
                            "equal_budget": {
                                "scoring": {"outcome": "no_selected_candidate"},
                                "candidate_sha256": None,
                            }
                        },
                    },
                )
            )
            self.last_output = tmp_path / request["block"]
            from oeis_learn.pipeline_validation.preparation_recovery import retain

            retain(self.last_output / "scores.json", result)
            self.last_wall_us = 320
            self.last_producer = seal(dict(request=request["identity"], elapsed_us=300))
            return {"scores": reference(self.last_output, self.last_output / "scores.json")}

    scorer = Scorer()
    args = (scorer, "exposed", [{"id": "a"}, {"id": "b"}], choices, path, 2040)
    with pytest.raises(pilot.PilotDeficit) as failed:
        score_choices(*args)
    assert failed.value.code == "QUALIFICATION_SCORER_INCOMPLETE"
    assert failed.value.counts["unscored_choices"] == 1
    for _ in range(2):
        scores, _, count = score_choices(*args)
        assert count == 1 and scores["denominator"] == 2
    assert calls == ["exposed", "exposed", "exposed-rescore-1", "exposed", "exposed-rescore-1"]
    debit = load(tmp_path / "qualification-costs/exposed-incomplete-score.json")
    assert debit["wall_us"] == 320 and debit["role"] == "score"


def test_collector_keeps_the_reason_for_a_blocked_qualification(tmp_path, monkeypatch):
    import json
    import runpy
    import subprocess
    import zipfile
    from pathlib import Path
    from types import SimpleNamespace

    repo = Path(__file__).resolve().parents[2]
    collect = runpy.run_path(
        str(repo / "specs/015-learned-program-search/validation/collect_audit.py")
    )["collect"]
    folder = tmp_path / "run-state/discovery-tools"
    folder.mkdir(parents=True)

    def blocked(cmd, *, stdout, **options):
        assert cmd[cmd.index("--action") + 1] == "qualify-pilot"
        stdout.write(
            b'{"mechanical_status":"BLOCKED","reason":"QUALIFICATION_SCORER_INCOMPLETE",'
            b'"deficit":{"unscored_choices":1},"progress":{"phase":"accept_frozen_pilot",'
            b'"controller":"' + b'a' * 40 + b'","candidate":"' + b'b' * 40 +
            b'","role":"full_size_score","private":"hidden-case"}}\n'
        )
        return SimpleNamespace(returncode=2)

    monkeypatch.setattr(subprocess, "run", blocked)
    assert collect(tmp_path, "a" * 40, "collect-pilot-qualification") == 2
    with zipfile.ZipFile(
        next(folder.glob("*/round028-pilot-qualification-diagnostics.zip"))
    ) as archive:
        outcome = json.loads(archive.read("qualification.json"))
        assert outcome["reason"] == "QUALIFICATION_SCORER_INCOMPLETE"
        assert outcome["deficit"]["unscored_choices"] == 1
        assert outcome["controller"] == "a" * 40 and outcome["worker"] == "b" * 40
        assert outcome["progress"] == {"phase": "accept_frozen_pilot", "role": "full_size_score"}
        assert b"hidden-case" not in archive.read("qualification.json")


def test_recovered_work_is_scored_before_measuring_a_single_cold_resident(tmp_path, monkeypatch):
    from oeis_learn.generalization.progressive.contracts import atomic_write
    from oeis_learn.pipeline_validation import progressive_pilot_qualification as qualification
    from types import SimpleNamespace

    cpu = SimpleNamespace(area=tmp_path)
    gpu = SimpleNamespace(area=tmp_path)
    sample = [
        dict(id=str(i), values=[str(i)] * 20, indices=list(map(str, range(20)))) for i in range(8)
    ]
    truth = [dict(r, values=r["values"] * 5, indices=list(map(str, range(100)))) for r in sample]
    model = digest("weights")
    weight = {"sha256": model}
    seen = []

    def role(worker, operation, data, **kwargs):
        rows = data["rows"]
        ids = [r["id"] for r in rows]
        assert len(set(ids)) == len(rows) == 160
        assert all(len(r["values"]) == (20 if operation == "search" else 100) for r in rows)
        seen.append((operation, data["block"], ids))
        folder = tmp_path / (data["block"] + "-" + operation)
        worker.last_output = folder
        if operation == "search":
            clocks = [
                clock_record(
                    prompt=k,
                    mode="correct",
                    snapshot=model,
                    start=1.0,
                    finish=60.0,
                    scope_deadline=61.0,
                    expected=8,
                    started=8,
                    terminal=8,
                    policy=CORRECT_FULL,
                )
                for k in ids
            ]
            doc = seal(
                dict(
                    version="discovery-candidates/v1",
                    checkpoint=model,
                    model=model,
                    recipe=digest({}),
                    prompts=digest(rows),
                    records={k: {"selections": {"equal_budget": None}} for k in ids},
                    elapsed_us=10_000_000_000,
                    clock=seal(
                        dict(
                            policy=CORRECT_FULL,
                            work=model,
                            inference_seed=71,
                            source=digest({}),
                            expected_portfolios=160,
                            missing_portfolios=0,
                            cost_complete=True,
                            clocks=clocks,
                        )
                    ),
                )
            )
            name = "candidates"
            partial = len(seen) == 1
            worker.last_producer = {
                "elapsed_us": 10_100_000_000,
                **({"recovered": "segments"} if partial else {}),
            }
            worker.last_wall_us = None if partial else 10_200_000_000
        else:
            doc = seal(
                dict(
                    version="discovery-scoring/v1",
                    candidates=data["candidates"]["sha256"],
                    denominator=160,
                    outcomes={
                        k: {
                            "equal_budget": {
                                "success": False,
                                "complete": False,
                                "candidate_sha256": None,
                                "scoring": {"complete": False, "outcome": "no_selected_candidate"},
                            }
                        }
                        for k in ids
                    },
                )
            )
            name = "scores"
            worker.last_producer = {"elapsed_us": 50, "result": {}}
            worker.last_wall_us = 80
        path = folder / (name + ".json")
        atomic_write(path, doc)
        return {name: reference(folder, path)}

    monkeypatch.setattr(qualification, "qualification_role", role)
    scale = qualification.resident_scale(
        cpu, gpu, sample, truth, {}, weight, tmp_path / "weights", 71, lambda *args, **kwargs: None
    )
    assert [(op, name) for op, name, _ in seen] == [
        ("search", "qualify-resident-endpoint"),
        ("score", "qualify-resident-endpoint"),
        ("search", "qualify-resident-endpoint-cold-1"),
        ("score", "qualify-resident-endpoint-cold-1"),
    ]
    assert not set(seen[0][2]) & set(seen[2][2])
    assert scale["cold_attempt"] == 1 and scale["slowest_case_us"] == 59_000_000
    assert scale["minimum_case_allowance_us"] == 60_000_000
    assert scale["slowest_case_margin_us"] == 1_000_000
    assert scale["search_supervisor_margin_us"] == 21_120_000_000 - 10_200_000_000
    assert scale["score_supervisor_margin_us"] == 21_120_000_000 - 80
