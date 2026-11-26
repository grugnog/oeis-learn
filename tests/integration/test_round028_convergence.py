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
        clock=seal(dict(policy=CORRECT_FULL, work=model, clocks=[clock], cost_complete=True)),
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
    draws = []

    def sample(decoder, requests):
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
        target=4_000_000,
        arm="ordinary",
        seconds=seconds,
        inference_seed=971,
        rows=prompts,
        recipe=recipe,
        weights=weight,
    )
    ref = worker.evaluate_role(tmp_path, data, None, "search")["candidates"]
    output = tmp_path / "output"
    candidates = load(output / ref["path"], maximum=128 << 20)
    scored = worker.evaluate_role(
        tmp_path,
        dict(rows=truth, seconds=seconds, candidates=reference(tmp_path, output / ref["path"])),
        None,
        "score",
    )["scores"]
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
