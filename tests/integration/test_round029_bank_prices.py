"""Private scoring of a frozen set of admissions on the previously exposed timing panel."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive.contracts import digest, load, seal
from oeis_learn.generalization.progressive.program_search_timing import BLOCKS
from oeis_learn.synthesis.admission import source_identity
from tests.integration.test_round029_isolated_control_prices import OPERATOR, api, panel


class Native:
    def __init__(self):
        self.calls = []

    def execute(self, source, indices, *, deadline):
        self.calls.append(source)
        return dict(outcome="ok", values=indices)


def reference(source, indices, limits, *, deadline):
    return SimpleNamespace(outcome="ok", values=indices)


def specimen():
    members, visible, truths = panel()
    for p, t in zip(visible, truths, strict=True):
        t["values"] = t["indices"]
        p["values"] = p["indices"]
    source = "(program (return $index))"
    sha = source_identity(source)
    banks = [
        dict(
            id=t["id"],
            truth=t,
            stratum=m["stratum"],
            selected=None,
            selected_verdict=None,
            gap=None,
            absent_sources=0,
            issued=1,
            expected=1,
            search_complete=True,
            programs=[],
        )
        for m, t in zip(members, truths, strict=True)
    ]
    banks[0].update(
        selected=sha,
        selected_verdict="solve",
        programs=[dict(source=source, sha256=sha, ordinals=[0], fit_completed_us=500)],
    )
    return dict(rows=banks), sha


def test_bank_reducer_independently_reopens_outcomes_and_counts_each_source_once(tmp_path):
    data, sha = specimen()
    client = Native()
    client.limits = object()
    req = seal(dict(method="ordinary-one", data=digest(data)))
    result = api["score_banks"](tmp_path, req, data, client, reference=reference)
    raw = load(tmp_path / "output" / result["artifact"]["path"])
    cells = api["bank_cost_cells"](req, data, raw)
    assert (
        client.calls == ["(program (return $index))"]
        and sum(c["cases"] for c in cells.values()) == 8
    )
    assert sum(c["any_solve"] for c in cells.values()) == 1
    assert sum(c["empty"] for c in cells.values()) == 7
    assert sum(c["programs"] for c in cells.values()) == 1
    assert "private/" not in json.dumps(cells) and sha not in json.dumps(cells)
    altered = deepcopy(raw)
    altered["cases"][data["rows"][0]["id"]]["programs"][sha]["reference"]["values"] = digest(
        ["changed"]
    )
    with pytest.raises(ValueError):
        api["bank_cost_cells"](req, data, seal(altered))


def test_missing_source_and_native_timeout_remain_unresolved(tmp_path):
    data, _ = specimen()
    data["rows"][1].update(gap="missing_discarded_source", absent_sources=1)
    client = Native()
    client.limits = object()
    client.execute = lambda *a, **kw: dict(outcome="timeout", values=[])
    req = seal(dict(method="ordinary-one", data=digest(data)))
    artifact = api["score_banks"](tmp_path, req, data, client, reference=reference)["artifact"]
    cells = api["bank_cost_cells"](req, data, load(tmp_path / "output" / artifact["path"]))
    assert sum(c["unresolved"] for c in cells.values()) == 2
    assert sum(c["empty"] for c in cells.values()) == 6
    assert sum(c["any_solve"] for c in cells.values()) == 0


def test_bank_prices_keep_sample_coverage_separate_from_the_wave():
    paid = dict(known_expected_us=55, known_tail_us=90, missing=["matched_full_bank_scoring"])
    cell = dict(cases=2, unresolved=0, programs=1, scoring_us=10, max_us=8)
    blocks = {b: dict(cells={"s": cell}, elapsed_us=25) for b in BLOCKS}
    report = api["bank_sample_forecast"](blocks, paid)
    assert report["observed_sample_us"] == 125
    assert report["matched_sample_expected_us"] == 225  # two ordinary, six learner, one D
    assert report["matched_sample_tail_us"] == 279
    assert report["known_wave"] == paid and report["full_wave_us"] is None
    blocks[BLOCKS[0]]["cells"] = {"s": dict(cell, unresolved=1)}
    assert api["bank_sample_forecast"](blocks, paid)["matched_sample_expected_us"] is None
    assert api["bank_sample_forecast"]({}, paid)["matched_sample_tail_us"] is None


def test_bank_cli_requires_original_receipt_and_new_zip(tmp_path):
    target = tmp_path / "bank.zip"
    with pytest.raises(SystemExit):
        api["cli"](["a" * 40, "20", "bank-prices", "--output", str(target)])
    assert not target.exists()


def test_five_banks_share_one_grant_and_resume_only_the_failed_role(tmp_path, monkeypatch):
    import zipfile
    from oeis_learn.generalization.progressive.contracts import atomic_write
    from oeis_learn.pipeline_validation import progressive_round028_rehearsal

    fn = api["bank_price_matrix"]
    state = fn.__globals__
    # This test exercises role failure/retry with a tiny fixture wheel.
    monkeypatch.setitem(state, "control_capacity", lambda *args, **kwargs: [
        dict(deficit_bytes=0, deficit_inodes=0)])
    source = tmp_path / "tree"
    operator = source / Path(state["LAUNCHER"]).with_name("qualify_continuation.py")
    operator.parent.mkdir(parents=True)
    operator.write_text("# synthetic operator")
    folder = tmp_path / "state"
    folder.mkdir()
    wheel = tmp_path / "test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("test/worker.py", "pass")
    runtime = dict(
        path=wheel, wheel=digest("wheel"), image=digest("image"), source=digest("source")
    )
    host = SimpleNamespace(policy=dict(free_disk_floor_bytes=0))
    monkeypatch.setattr(
        progressive_round028_rehearsal, "physical_scope", lambda *a, **kw: digest("fixture machine")
    )
    monkeypatch.setitem(state, "reopen_control_costs", lambda *a: dict(known_expected_us=3))
    monkeypatch.setitem(
        state,
        "paid_bank_boundary",
        lambda *a: (tmp_path, digest("reg"), digest("cut"), {}, dict.fromkeys(BLOCKS, {})),
    )
    data, _ = specimen()
    opened, calls = [], []

    def open_inputs(area, reg, index, name, rows, prompts, truth, released):
        opened.append(name)
        return data, dict(search=digest(name + "/search"), score=digest(name + "/score"))

    monkeypatch.setitem(state, "paid_bank_inputs", open_inputs)
    fail = {BLOCKS[1]}

    def stage(host, source, image, out, argv, *, gpu, seconds, readonly_inputs):
        assert not gpu and opened[-len(BLOCKS) :] == list(BLOCKS)
        inputs = {dest: path for path, dest in readonly_inputs}
        assert set(inputs) == {
            "/work/inputs/" + n
            for n in ("control-worker.py", "test.whl", "request.json", "data.json")
        }
        req = load(inputs["/work/inputs/request.json"])
        assert req["parents"]["score"] == digest(req["method"] + "/score")
        payload = load(inputs["/work/inputs/data.json"])
        calls.append(req["method"])
        atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
        if req["method"] in fail:
            fail.clear()
            raise ValueError("fixture interrupted one role")
        local = tmp_path / ("case" + str(len(calls)))
        client = Native()
        client.limits = object()
        artifact = api["score_banks"](local, req, payload, client, reference=reference)["artifact"]
        dest = out / artifact["path"]
        dest.parent.mkdir()
        dest.write_bytes((local / "output" / artifact["path"]).read_bytes())
        atomic_write(
            out / "producer.json",
            seal(
                dict(
                    artifact=artifact, request=req["identity"], native=digest("native"), worker_us=1
                )
            ),
        )
        mounts = [
            dict(Type="bind", Source=str(p), Destination=k, RW=False) for k, p in inputs.items()
        ]
        mounts.append(dict(Type="bind", Source=str(out), Destination="/work/output", RW=True))
        atomic_write(
            out / "container-inspect.json",
            [
                dict(
                    Image=image,
                    Mounts=mounts,
                    HostConfig=dict(
                        ReadonlyRootfs=True,
                        NetworkMode="none",
                        Privileged=False,
                        CapDrop=["ALL"],
                        Devices=[],
                    ),
                )
            ],
        )
        atomic_write(
            out / "process.json", dict(outcome="COMPLETE", returncode=0, cleanup_failure=False)
        )

    members, prompts, truths = panel()
    args = (
        tmp_path,
        source,
        folder,
        "a" * 40,
        20,
        runtime,
        (members, prompts, truths, {}, "fixture"),
        dict(checkpoint_receipt=digest("checkpoint")),
        host,
        stage,
        digest("controls"),
    )
    partial = fn(*args)
    assert partial["errors"] == {BLOCKS[1]: dict(phase="bank-score", error="ValueError")}, partial[
        "errors"
    ]
    assert (
        len(calls) == 5 and partial["files"]["forecast.json"]["matched_sample_expected_us"] is None
    )
    done = fn(*args)
    assert done["status"] == "COMPLETE" and done["grant"] == partial["grant"] and len(calls) == 6
    assert (
        done["charged_work"]["native_work"] == 6
        and done["charged_work"]["tokens"] == done["gpu_dispatches"] == 0
    )
    assert done["elapsed_us"] + done["remaining_us"] == 20 * 60_000_000
    assert done["files"]["forecast.json"]["matched_sample_expected_us"] > 0
    again = fn(*args)
    assert (
        again["receipt"] == done["receipt"]
        and again["charged_work"] == done["charged_work"]
        and len(calls) == 6
    )
    assert "private/" not in json.dumps(again) and "return $index" not in json.dumps(again)
    reopened = api["previous_bank_prices"](tmp_path, folder, again["receipt"], args[7], args[6], runtime, host)
    assert reopened["projection"] == done["files"]["forecast.json"]


    import subprocess
    import sys

    output = tmp_path / "bank-prices.zip"
    api["write_readiness_zip"](output, deepcopy(again))
    portable = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(OPERATOR), "--verify-bank-zip", str(output)],
        cwd=tmp_path,
        check=True,
        text=True,
        capture_output=True,
    )
    assert json.loads(portable.stdout)["status"] == "COMPLETE"
    changed = deepcopy(again)
    changed["files"]["forecast.json"]["observed_sample_us"] += 1
    api["write_readiness_zip"](tmp_path / "changed.zip", changed)
    with pytest.raises(ValueError, match="balance disagree"):
        api["verify_bank_zip"](tmp_path / "changed.zip")


def test_timing_chain_reopens_only_the_committed_block_on_its_original_panel(tmp_path):
    from oeis_learn.generalization.progressive.contracts import atomic_write, reference as bound
    from oeis_learn.pipeline_validation.pilot_clocks import DESCRIPTIVE, clock_record
    from oeis_learn.pipeline_validation.progressive_timing_worker import observations

    members, prompts, truths = panel()
    model, recipe, registration = map(digest, ("model", "recipe", "registration"))
    clock = [
        clock_record(
            prompt=r["task"],
            mode="correct",
            snapshot=model,
            start=1,
            finish=2,
            scope_deadline=64,
            expected=1,
            started=1,
            terminal=1,
            policy=DESCRIPTIVE,
        )
        for r in members
    ]
    rec = dict(
        costs=dict(
            correct=dict(
                equal_budget=None,
                complete=True,
                archive=dict(retained=[], omitted=[]),
                details=[
                    dict(
                        position=0, generation_seconds=0, fitting=dict(complete=True, admission=[])
                    )
                ],
            )
        ),
        selections=dict(equal_budget=None),
        arm_evidence=dict(equal_budget=dict(complete=True)),
    )
    gen = seal(
        dict(
            version="discovery-candidates/v1",
            methods=["correct"],
            model=model,
            recipe=recipe,
            prompts=digest(sorted(prompts, key=lambda p: p["id"])),
            clock=seal(
                dict(
                    policy=DESCRIPTIVE,
                    work=model,
                    source=recipe,
                    expected_portfolios=8,
                    missing_portfolios=0,
                    clocks=clock,
                )
            ),
            records={r["task"]: rec for r in members},
        )
    )
    index = {}
    name = "ordinary-one"

    def stage(role, data, record, result_name, path, clocks):
        data = seal(dict(data, block=name))
        req = seal(dict(registration=registration, data=data["identity"], role=role))
        folder = tmp_path / "roles" / req["identity"][7:]

        def put(rel, obj):
            dest = folder / rel
            atomic_write(dest, obj)
            index[str(dest.relative_to(tmp_path))] = bound(tmp_path, dest)
            return bound(folder, dest)

        atomic_write(
            tmp_path / "assignments" / (name + "-" + role + ".json"),
            seal(dict(request=req["identity"])),
        )
        put("request.json", req)
        put("data.json", data)
        dr = "attempt-0000/output/"
        ref = put(dr + path, record)
        producer = put(
            dr + "producer.json",
            seal(
                dict(
                    version="round028-timing-output/v1",
                    request=req["identity"],
                    result={result_name: dict(ref, path=path), **clocks},
                )
            ),
        )
        put("boundary.json", seal(dict(producer=producer, request=req["identity"])))
        return ref["sha256"]

    times = {r["task"]: 1000000 for r in members}
    gen_hash = stage(
        "search",
        dict(rows=prompts),
        gen,
        "candidates",
        "generation/candidates.json",
        dict(case_us=times),
    )
    score = dict(
        candidate_sha256=None,
        success=False,
        upstream=dict(complete=True),
        scoring=dict(outcome="no_selected_candidate", complete=True, elapsed_seconds=0),
    )
    scoring = seal(
        dict(
            version="discovery-scoring/v1",
            denominator=8,
            candidates=gen_hash,
            outcomes={r["task"]: dict(equal_budget=score) for r in members},
        )
    )
    stage(
        "score",
        dict(rows=truths, candidates=dict(sha256=gen_hash)),
        scoring,
        "scores",
        "scored/scores.json",
        {},
    )
    expected = observations(
        gen, scoring, {r["task"]: r["stratum"] for r in members}, times, gen_hash, DESCRIPTIVE
    )
    args = (tmp_path, registration, index, name, members, prompts, truths, dict(cells=expected))
    data, refs = api["paid_bank_inputs"](*args)
    assert len(data["rows"]) == 8 and all(
        r["programs"] == [] and r["gap"] is None for r in data["rows"]
    )
    assert refs["search"] == gen_hash
    with pytest.raises(ValueError):
        api["paid_bank_inputs"](*args[:-1], dict(cells={}))
    altered = next(p for p in index if p.endswith("generation/candidates.json"))
    (tmp_path / altered).write_text("{}")
    with pytest.raises(ValueError):
        api["paid_bank_inputs"](*args)


def test_independent_reference_executes_the_full_original_suffix(tmp_path):
    from oeis_learn.synthesis.contracts import ResourceProfile

    data, _ = specimen()
    req = seal(dict(method="ordinary-full", data=digest(data)))
    client = Native()
    client.limits = ResourceProfile()
    artifact = api["score_banks"](tmp_path, req, data, client)["artifact"]
    result = api["bank_cost_cells"](req, data, load(tmp_path / "output" / artifact["path"]))
    assert sum(c["any_solve"] for c in result.values()) == 1


def test_historical_private_cut_is_pinned_by_the_independent_workstation(tmp_path, monkeypatch):
    import io
    import zipfile
    from oeis_learn.generalization.progressive.contracts import atomic_write, reference as bound
    from oeis_learn.generalization.progressive.program_search_timing import PANEL
    from oeis_learn.pipeline_validation import (
        progressive_round028_pilot,
        progressive_velocity_donor,
    )
    from oeis_learn.pipeline_validation.progressive_round028_audit import SESSION
    from oeis_learn.pipeline_validation.progressive_round028_timing import AREA

    rows, _, _ = panel()
    original = tmp_path / SESSION
    area = original / AREA
    reg = seal(
        dict(
            roster=rows,
            blocks=list(BLOCKS),
            panel=PANEL,
            scope="previously_exposed_timing_only",
            attempts=dict(one=1, full=8),
        )
    )
    public = seal(dict(registration=reg["identity"], blocks={b: dict(cells={}) for b in BLOCKS}))
    for key, val in (("registration", reg), ("public", public)):
        atomic_write(area / (key + ".json"), val)
    cut = seal(
        dict(
            registration=bound(area, area / "registration.json"),
            public=bound(area, area / "public.json"),
            panel=PANEL,
            files=[],
        )
    )
    atomic_write(area / "boundary.json", cut)
    replay = seal(
        dict(
            timing_boundary=cut["identity"],
            portfolios=40,
            timing=public["identity"],
            registration=reg["identity"],
        )
    )
    closure = original / "private/terminal/closure.json"
    atomic_write(closure, seal(dict(replay=replay)))
    workstation = seal(dict(private_replay=replay["identity"]))
    parent = io.BytesIO()
    with zipfile.ZipFile(parent, "w") as z:
        z.writestr("workstation.json", json.dumps(workstation))
    pilot = tmp_path / "pilot.zip"
    with zipfile.ZipFile(pilot, "w") as z:
        z.writestr("round028-parent.zip", parent.getvalue())
    monkeypatch.setattr(progressive_velocity_donor, "_archives", lambda *a: (dict(pilot=pilot), {}))
    monkeypatch.setattr(progressive_round028_pilot, "PARENT_WORKSTATION", workstation["identity"])
    actual = api["paid_bank_boundary"](tmp_path, rows)
    assert actual[:4] == (area, reg["identity"], cut["identity"], {})
    closure.write_text(json.dumps(seal(dict(replay=seal(dict(replay, portfolios=32))))))
    with pytest.raises(ValueError, match="independently verified terminal"):
        api["paid_bank_boundary"](tmp_path, rows)
