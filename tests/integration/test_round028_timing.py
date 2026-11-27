"""Candidate pricing on historical families and the workstation spending boundary."""

import copy

import pytest

from oeis_learn.generalization.progressive.cohorts import STRATA
from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.generalization.progressive.program_search_timing import BLOCKS, project
from oeis_learn.pipeline_validation.pilot_clocks import DESCRIPTIVE, DRAIN, clock_record
from oeis_learn.pipeline_validation.progressive_round028_timing import timing_cases
from tests.integration.test_round028_receipts import original as original  # noqa: PLC0414 - shared fixture


def test_timing_fixtures_are_distinct_historical_families_independent_of_outcomes():
    rows = [
        {
            "task": f"{s}-{i}",
            "group": f"family-{s}-{i // 2}",
            "stratum": s,
            "observations": {"trial-05": i},
        }
        for s in STRATA
        for i in range(32)
    ]
    cut = {"cases": rows}
    first = timing_cases(cut)
    assert len(first) == len({r["group"] for r in first}) == 8
    assert {s: sum(r["stratum"] == s for r in first) for s in STRATA} == dict.fromkeys(STRATA, 2)
    for row in rows:
        row["observations"] = {"incumbent": "unresolved"}
    assert [r["task"] for r in first] == [r["task"] for r in timing_cases({"cases": rows[::-1]})]
    with pytest.raises(ValueError):
        timing_cases({"cases": rows[:80]})


def test_one_attempt_clock_is_a_real_registered_portfolio():
    def clock(policy, count):
        return clock_record(
            prompt="p",
            mode="correct",
            snapshot=digest("m"),
            start=0,
            scope_deadline=80,
            finish=3,
            expected=count,
            started=1,
            terminal=1,
            policy=policy,
        )

    assert clock(DESCRIPTIVE, 1)["cost_complete"]
    assert not clock(DRAIN, 8)["cost_complete"]
    with pytest.raises(ValueError):
        clock(DESCRIPTIVE, 8)


def measured():
    counts = {
        "cases": 2,
        "closed": 2,
        "started": 2,
        "selected": 1,
        "scored": 2,
        "executed": 1,
        "search_us": 20,
        "max_search_us": 15,
        "score_us": 4,
        "max_score_us": 3,
        "portfolio_us": 19,
        "decode_us": 12,
        "native_us": 7,
    }
    return {
        name: {
            "cells": {s: dict(counts) for s in STRATA},
            "gpu_wall_us": 120,
            "cpu_wall_us": 30,
            "lease_us": 110,
            "gpu_worker_us": 95,
            "cpu_worker_us": 22,
            "telemetry": {"samples": 3, "readable_samples": 2, "covered_us": 50, "busy_us": 30},
            "peak_device_bytes": 1024,
        }
        for name in BLOCKS
    }


def plan():
    return {
        "training_us": 1000,
        "shift_us": 200000,
        "reserve_us": 100000,
        "checkpoint_cycle_us": 0,
        "training_role_overhead_us": 0,
        "probes": dict.fromkeys(STRATA, 20),
        "score_floor_us": {arm: dict.fromkeys(STRATA, 2) for arm in ("learner", "incumbent")},
    }


def test_timing_forecast_prices_each_arm_and_preserves_unknown_work():
    blocks = measured()
    rates = project(blocks, plan())
    # Two observed search and scorer operations per cell, 40 + 14 us of batch overhead.
    assert rates["pilot_us"] == 1000 + (4 * (80 * 18 + 54)) + 3 * (160 * 18 + 54)
    assert rates["repeat_us"] == 1000 + 5 * (160 * 18 + 54)
    assert rates["remaining_us"] == 200000 - rates["pilot_us"] - rates["repeat_us"]
    assert rates["action"] == "QUALIFY_PROSPECTIVE_PILOT"
    slow = copy.deepcopy(blocks)
    slow["first_divergence-full"]["cells"]["parameter"]["closed"] = 1
    assert project(slow, plan())["action"] == "RESOLVE_INCOMPLETE_TIMING"
    short = dict(plan(), shift_us=2000)
    assert project(blocks, short)["action"] == "REPORT_PILOT_COST_DEFICIT"
    with pytest.raises(ValueError):
        project(dict(blocks, other={}), plan())


@pytest.mark.parametrize(("research", "count"), [(True, 1), (False, 8)])
def test_real_weight_reader_registered_sampler_clock_and_scorer(
    tmp_path, monkeypatch, research, count
):
    import torch

    from oeis_learn.decoder import bigint_generation
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.baselines import load_random
    from oeis_learn.generalization.progressive.contracts import load, reference
    from oeis_learn.generalization.progressive.models import construct, resolve_profile
    from oeis_learn.pipeline_validation.pilot_clocks import CORRECT_FULL
    from oeis_learn.pipeline_validation.progressive_evaluation import generate, score
    from oeis_learn.pipeline_validation.progressive_timing_worker import observations

    recipe = resolve({"version": "bigint-bootstrap/v1"})
    if research:
        recipe = resolve_profile(recipe)
        encoder, decoder, _ = construct(recipe, device="cpu")
    else:
        encoder, decoder, _ = load_random(recipe, seed=13)
    torch.save(
        {"encoder": encoder.state_dict(), "decoder": decoder.state_dict()}, tmp_path / "weights.pt"
    )
    draws = []

    def sample(decoder, requests):
        draws.extend((r.seed, r.temperature) for r in requests)
        return [
            {
                "tokens": [],
                "source": "",
                "complete": False,
                "generation_batch_size": 1,
                "generation_seconds": 0.0,
            }
            for _ in requests
        ]

    monkeypatch.setattr(bigint_generation, "sample_batch", sample)
    truth = {"id": "original", "indices": list(map(str, range(100))), "values": ["42"] * 100}
    prompt = {k: v if k == "id" else v[:20] for k, v in truth.items()}
    policy = DESCRIPTIVE if count == 1 else CORRECT_FULL
    elapsed = {}
    ref = generate(
        tmp_path,
        "generation",
        [prompt],
        recipe,
        client=None,
        methods=("correct",),
        model_control="earlier_checkpoint" if research else "retained_D",
        reference_recipe=recipe if research else None,
        checkpoint=reference(tmp_path, "weights.pt"),
        evaluation_policy=policy,
        seconds=60,
        case_elapsed_us=elapsed,
    )
    gen = load(tmp_path / ref["path"])
    result = score(tmp_path, ref, [truth], client=None, output="scored", selected_only=True)
    cells = observations(gen, result, {"original": "composition"}, elapsed, ref["sha256"], policy)
    c = cells["composition"]
    assert gen["clock"]["cost_complete"] and gen["clock"]["missing_portfolios"] == 0
    assert c["started"] == len(draws) == count and c["closed"] == c["scored"] == 1
    assert c["selected"] == c["executed"] == 0 and draws[0][1] == 0
    assert c["search_us"] >= c["portfolio_us"] > 0


def test_timing_role_boundaries_and_original_budget_survive_interruption(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.contracts import atomic_write, load, seal
    from oeis_learn.pipeline_validation import progressive_handoff, progressive_round028_rehearsal
    from oeis_learn.pipeline_validation import progressive_round028_panels as panels
    from oeis_learn.pipeline_validation.progressive_round028_timing import (
        TimingRole,
        cost,
        diagnostic_roles,
        timed_role,
    )
    from tests.integration.test_round028_receipts import SOURCE

    image = digest("test HIP")
    host = SimpleNamespace(
        engine=SimpleNamespace(name="podman", image=lambda v: v),
        policy={"free_disk_floor_bytes": 0},
    )
    monkeypatch.setattr(panels, "candidate_image", lambda *a: (image, 4, True))
    monkeypatch.setattr(
        progressive_round028_rehearsal, "physical_scope", lambda *a: digest("boot-and-render-node")
    )
    calls = []

    def invoke(
        host, source, immutable, output, argv, *, gpu, seconds, readonly_inputs, record_lease
    ):
        assert gpu and record_lease and immutable == image
        paths = {d: p for p, d in readonly_inputs}
        assert set(paths) == {
            "/work/inputs/request.json",
            "/work/inputs/data.json",
            "/work/inputs/weights.pt",
        }
        req, data = (load(paths["/work/inputs/" + key + ".json"]) for key in ("request", "data"))
        assert req["data"] == data["identity"]
        calls.append(data["block"])
        output.mkdir()
        if data["block"] == "first_divergence-one":
            raise RuntimeError("interrupted sampler")
        atomic_write(
            output / "producer.json",
            seal(
                {
                    "version": "round028-timing-output/v1",
                    "request": req["identity"],
                    "result": {},
                    "native": digest("native"),
                    "elapsed_us": 4,
                }
            ),
        )
        atomic_write(output / "gpu-lease.json", {"elapsed_us": 5})
        mounts = [(p, d, False) for p, d in readonly_inputs] + [(output, "/work/output", True)]
        atomic_write(
            output / "container-inspect.json",
            [
                {
                    "Image": image,
                    "HostConfig": {
                        "NetworkMode": "none",
                        "ReadonlyRootfs": True,
                        "Privileged": False,
                    },
                    "Mounts": [
                        {"Type": "bind", "Source": str(p), "Destination": d, "RW": w}
                        for p, d, w in mounts
                    ],
                }
            ],
        )

    monkeypatch.setattr(progressive_handoff, "stage", invoke)
    reg = seal({"source": digest("candidate")})
    (tmp_path / "weights.pt").write_bytes(b"weights")
    caps = {k: 2 * v for k, v in cost("ordinary-one", "search").items()}
    journal = BudgetJournal(tmp_path / "accounting", reg["identity"], caps)

    def start(name):
        runner = TimingRole(None, SOURCE, tmp_path, "1" * 40, tmp_path, reg, lambda *a, **k: host)
        runner.phase = "hip"
        return timed_role(
            runner,
            journal,
            name,
            "search",
            {"block": name},
            [(tmp_path / "weights.pt", "/work/inputs/weights.pt")],
        )

    assert start("ordinary-one") == {}
    (cut,) = tmp_path.glob("roles/*/boundary.json")
    cut.unlink()
    assert start("ordinary-one") == {}
    with pytest.raises(RuntimeError, match="interrupted"):
        start("first_divergence-one")
    with pytest.raises(ValueError, match="original timing attempt"):
        start("first_divergence-one")
    assert calls == ["ordinary-one", "first_divergence-one"]
    assert journal.snapshot()["consumed"]["draws"] == 16 and len(journal.snapshot()["pending"]) == 1
    diag = diagnostic_roles(tmp_path)
    assert diag["ordinary-one-search"]["completed"] and diag["ordinary-one-search"]["lease_us"] == 5
    assert not diag["first_divergence-one-search"]["completed"]


def test_transport_recomputes_changed_prices_and_refuses_substituted_panel(
    original, tmp_path, monkeypatch
):
    import json
    import zipfile

    from oeis_learn.generalization.progressive import program_search_panel_report as panels
    from oeis_learn.generalization.progressive import program_search_timing as timing
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.generalization.progressive.program_search_rehearsal_report import (
        encode,
        publish,
        verify_transport,
    )
    from oeis_learn.pipeline_validation.progressive_round028 import ARCHIVE
    from oeis_learn.pipeline_validation.progressive_round028_receipts import recover
    from tests.integration.test_round028_panels import historical_clocks, summary
    from tests.integration.test_round028_receipts import REVISION, SOURCE

    o = original
    weights = {}
    old = recover(o.workspace, o.folder, SOURCE, REVISION, o.store, weight_slices=weights)
    assert set(weights) == {"ordinary", "first_divergence"} and all(
        p.is_file() for p, _ in weights.values()
    )
    previous = tmp_path / "rehearsal.zip"
    trainer = publish(previous, old, o.folder / ARCHIVE)
    registration = seal({"candidate": "1" * 40})
    panel = panels.make_report(
        registration,
        trainer,
        summary(deficit=False),
        historical_clocks(),
        timings={"elapsed_us": 10, "role_us": 5, "unknown_role_intervals": 0, "image_build_us": 1},
    )
    parent = tmp_path / "panels.zip"
    panels.publish(parent, panel, previous)
    monkeypatch.setattr(timing, "PANEL", panel["identity"])
    exercise_collection(o, parent, panel, weights, monkeypatch)
    blocks = measured()
    result = seal(
        {
            "version": timing.VERSION,
            "panel": panel["identity"],
            "candidate": "2" * 40,
            "registration": digest("registration"),
            "scope": "previously_exposed_timing_only",
            "blocks": blocks,
            "forecast": timing.project(blocks, timing.parent_plan(panel, old)),
            "image_build_us": {"hip": 4, "cpu": 3},
            "space_before": {"free_bytes": 10, "free_inodes": 5},
            "space_after": {"free_bytes": 9, "free_inodes": 4},
        }
    )
    output = tmp_path / "timing.zip"
    receipt = panels.publish_child(output, result, parent, timing.MEMBERS, timing.verify_timing)
    assert (
        verify_transport(output) == receipt
        and receipt["portfolios"] == 40
        and receipt["new_training_updates"] == 0
    )
    with zipfile.ZipFile(output) as archive:
        assert b"original" not in archive.read("timing.json")
        rep = json.loads(archive.read("timing.json"))
        prior = archive.read("round028-panels.zip")
    rep["forecast"]["pilot_us"] += 1
    changed = tmp_path / "changed.zip"
    with zipfile.ZipFile(changed, "w") as archive:
        archive.writestr("timing.json", encode(seal(rep)))
        archive.writestr("round028-panels.zip", prior)
    with pytest.raises(ValueError):
        timing.verify_timing(changed)
    monkeypatch.setattr(timing, "PANEL", digest("other valid supply"))
    with pytest.raises(ValueError):
        timing.verify_timing(output)


def exercise_collection(o, parent, panel, weights, monkeypatch):
    """Drive the actual custodian, journal, collection and private replay with isolated role doubles."""
    import shutil

    from oeis_learn.generalization.progressive.contracts import atomic_write, load, reference, seal
    from oeis_learn.pipeline_validation import progressive_round028_timing as host
    from oeis_learn.pipeline_validation.pilot_clocks import CORRECT_FULL
    from tests.integration.test_round028_receipts import SOURCE

    with monkeypatch.context() as patch:
        prior = o.folder / host.PANEL_ARCHIVE
        prior.parent.mkdir(parents=True)
        shutil.copyfile(parent, prior)
        private = o.folder / "private/panel-supply" / host.PANEL_REVISION
        private.mkdir(parents=True)
        atomic_write(private / "boundary.json", seal({}))
        patch.setattr(host, "PANEL", panel["identity"])
        patch.setattr(host, "bind_private", lambda area, report: None)
        o.store.append(dict(o.store.state, route="MATCHED_GENERATOR"))
        calls = []

        def channels(root, rows):
            truth = [
                {"id": r["task"], "indices": list(map(str, range(100))), "values": ["1"] * 100}
                for r in rows
            ]
            return [{k: v if k == "id" else v[:20] for k, v in r.items()} for r in truth], truth

        patch.setattr(host, "original_channels", channels)
        all_weights = dict(weights, incumbent=weights["ordinary"])
        patch.setattr(
            host,
            "original_weights",
            lambda *a: ({n: {"model": n} for n in all_weights}, all_weights),
        )

        from oeis_learn.pipeline_validation.progressive_round028_panels import NativeRoles
        from oeis_learn.pipeline_validation.preparation_recovery import retain

        class Roles:
            phase = "cpu"
            build_us = 2

            def __init__(self, args, source, root, revision, area, registration, host_factory):
                self.area, self.registration = area, registration
                self.native = None

            def __call__(self, role, data, *, seconds, inputs):
                block = data["block"]
                calls.append((role, block))
                physical = digest(["workstation", self.phase])
                image_id = digest(["candidate", self.phase])
                opts = {
                    "revision": self.registration["candidate"],
                    "phase": self.phase,
                    "engine": "podman",
                }
                retain(
                    self.area / ("image-" + digest(opts)[7:] + ".json"),
                    seal({"options": opts, "image": image_id, "elapsed_us": 2}),
                )
                retain(self.area / ("physical-" + self.phase + ".json"), seal({"scope": physical}))
                req = seal(
                    {
                        "version": "round028-timing-role/v1",
                        "image": image_id,
                        "engine": "podman",
                        "registration": self.registration["identity"],
                        "source": self.registration["source"],
                        "role": role,
                        "data": data["identity"],
                        "physical": physical,
                        "candidate_native": self.native,
                    }
                )
                retain(
                    self.area / "assignments" / (block + "-" + role + ".json"),
                    seal({"request": req["identity"]}),
                )
                folder = self.area / "roles" / req["identity"][7:]
                folder.mkdir(parents=True)
                out = folder / "attempt-0000/output"
                model = digest(block.split("-")[0])
                keys = [r["id"] for r in data["rows"]]
                if role == "search":
                    assert self.phase == "hip" and set(data) == {
                        "block",
                        "rows",
                        "weights",
                        "recipe",
                        "identity",
                    }
                    assert all(len(r["values"]) == 20 for r in data["rows"])
                    assert inputs[0][1] == "/work/inputs/weights.pt"
                    count = 1 if block.endswith("-one") else 8
                    policy = DESCRIPTIVE if count == 1 else CORRECT_FULL
                    clocks = [
                        clock_record(
                            policy=policy,
                            prompt=k,
                            mode="correct",
                            snapshot=model,
                            start=0,
                            scope_deadline=90,
                            finish=40.0,
                            expected=count,
                            started=count,
                            terminal=count,
                        )
                        for k in keys
                    ]
                    records = {
                        k: {
                            "costs": {
                                "correct": {"details": [{"generation_seconds": 0.0}] * count}
                            },
                            "selections": {"equal_budget": None},
                        }
                        for k in keys
                    }
                    gen = seal(
                        {
                            "version": "discovery-candidates/v1",
                            "model": model,
                            "methods": ["correct"],
                            "checkpoint": data["weights"]["sha256"],
                            "recipe": digest(data["recipe"]),
                            "model_control": "retained_D"
                            if block == "incumbent-full"
                            else "earlier_checkpoint",
                            "prompts": digest(sorted(data["rows"], key=lambda r: r["id"])),
                            "records": records,
                            "clock": seal(
                                {
                                    "policy": policy,
                                    "clocks": clocks,
                                    "work": model,
                                    "source": digest(data["recipe"]),
                                    "expected_portfolios": 8,
                                    "missing_portfolios": 0,
                                }
                            ),
                        }
                    )
                    atomic_write(out / "generation/candidates.json", gen)
                    result = {
                        "candidates": reference(out, "generation/candidates.json"),
                        "case_us": dict.fromkeys(keys, 40_000_001),
                        "peak_device_bytes": 32,
                        "telemetry": {
                            "samples": 0,
                            "readable_samples": 0,
                            "covered_us": 0,
                            "busy_us": None,
                        },
                    }
                else:
                    assert self.phase == "cpu" and set(data) == {
                        "block",
                        "rows",
                        "candidates",
                        "identity",
                    }
                    assert all(len(r["values"]) == 100 for r in data["rows"])
                    path, dest = inputs[0]
                    assert path.is_file() and dest == "/work/inputs/candidates.json"
                    assert set(load(path)["records"]) == set(keys)
                    raw = {
                        "complete": True,
                        "outcome": "no_selected_candidate",
                        "elapsed_seconds": 0.000002,
                    }
                    scores = seal(
                        {
                            "version": "discovery-scoring/v1",
                            "denominator": 8,
                            "candidates": data["candidates"]["sha256"],
                            "outcomes": {k: {"equal_budget": {"scoring": raw}} for k in keys},
                        }
                    )
                    atomic_write(out / "scored/scores.json", scores)
                    result = {"scores": reference(out, "scored/scores.json")}
                self.last_output = out
                self.last_wall_us = 340_000_000 if role == "search" else 2000
                self.last_lease_us = 330_000_000 if role == "search" else 0
                self.native = digest(["native", self.phase])
                self.last_producer = seal(
                    {
                        "version": "round028-timing-output/v1",
                        "request": req["identity"],
                        "native": self.native,
                        "result": result,
                        "elapsed_us": 325_000_000 if role == "search" else 1400,
                    }
                )
                atomic_write(out / "producer.json", self.last_producer)
                atomic_write(folder / "request.json", req)
                atomic_write(folder / "data.json", data)
                atomic_write(out.parent / "start.json", seal({"request": req["identity"]}))
                atomic_write(
                    out.parent / "end.json",
                    seal(
                        {
                            "request": req["identity"],
                            "success": True,
                            "elapsed_us": self.last_wall_us,
                        }
                    ),
                )
                if self.phase == "hip":
                    atomic_write(out / "gpu-lease.json", {"elapsed_us": self.last_lease_us})
                mounts = [
                    (folder / "request.json", "/work/inputs/request.json", False),
                    (folder / "data.json", "/work/inputs/data.json", False),
                    (out, "/work/output", True),
                    *((p, d, False) for p, d in inputs),
                ]
                atomic_write(
                    out / "container-inspect.json",
                    [
                        {
                            "Image": image_id,
                            "HostConfig": {
                                "NetworkMode": "none",
                                "ReadonlyRootfs": True,
                                "Privileged": False,
                            },
                            "Mounts": [
                                {"Type": "bind", "Source": str(path), "Destination": dest, "RW": rw}
                                for path, dest, rw in mounts
                            ],
                        }
                    ],
                )
                self.last_boundary = folder / "boundary.json"
                NativeRoles.bind(self, folder, out.parent, req)
                return result

        patch.setattr(host, "TimingRole", Roles)

        def collect(rev):
            return host.timing_command(None, SOURCE, o.workspace, o.folder, rev, o.store, None)

        raw, code = collect("2" * 40)
        assert code == 0 and raw["receipt"]["portfolios"] == 40
        assert calls == [(role, block) for block in BLOCKS for role in ("search", "score")]
        assert collect("3" * 40) == (raw, code) and len(calls) == 10
        publication = o.folder / raw["archive"]
        publication.unlink()
        assert collect("2" * 40) == (raw, code) and publication.is_file() and len(calls) == 10
        from types import SimpleNamespace
        from oeis_learn.pipeline_validation import progressive_round028 as commands
        from oeis_learn.pipeline_validation.progressive_round028_audit import ORIGINAL

        patch.setattr(
            commands,
            "metadata",
            lambda *a: {
                "qualified_checkout": True,
                "files": {ORIGINAL + "/prepare-research/frozen/campaign.json": True},
            },
        )
        patch.setattr(commands, "candidate_revision", lambda *a: "3" * 40)

        def action(name):
            return commands.execute(
                SimpleNamespace(action=name, archive=None, workspace=str(o.workspace)),
                SOURCE,
                host_factory=lambda *a, **k: pytest.fail("terminal replay opened a device"),
            )

        # Interrupt after private replay and publication, exactly where the old journal charged
        # the saved checkpoints. A pending close-out must never fall back to fresh timing roles.
        from oeis_learn.generalization.progressive import storage
        with (o.folder / "private/retained-checkpoint").open("xb") as stream:
            stream.truncate(1_257_820_683)
        journal_write = storage.journal_write
        with patch.context() as previous:
            previous.setattr(storage, "journal_write", lambda root, path, value, **kw: journal_write(root, path, value))
            stopped, code = action("run")
        assert code == 2 and stopped["spending_reason"] == "REPORT_PILOT_COST_DEFICIT"
        assert action("status")[0]["mechanical_status"] == "PUBLICATION_PENDING"
        assert action("preflight")[0]["admission"] == "TERMINAL_PUBLICATION_PENDING"
        assert action("verify")[0]["reason"] == "TERMINAL_PUBLICATION_PENDING"
        assert action("rehearse")[0]["gpu_dispatches"] == 0
        cut = o.folder / host.AREA / "boundary.json"
        moved = cut.rename(cut.with_suffix(".saved"))
        for name in ("finalize", "run", "timing"):
            unavailable, code = action(name)
            assert code == 2 and unavailable["reason"] == "SAVED_TIMING_BOUNDARY_REQUIRED"
            assert unavailable["gpu_dispatches"] == 0
        moved.rename(cut)
        terminal, exit_code = action("run")
        assert (
            exit_code == 0
            and terminal["next_action"] == "RETAIN_BASELINE"
            and terminal["gpu_dispatches"] == 0
        )
        assert action("verify") == action("finalize") == action("run") == (terminal, 0)
        assert action("status")[0]["deficit_us"] == action("preflight")[0]["deficit_us"] > 0
        assert action("rehearse")[0]["mechanical_status"] == "SKIPPED"
        assert len(calls) == 10
        first = next(
            (o.folder / host.AREA).glob("roles/*/attempt-*/output/generation/candidates.json")
        )
        with first.open("ab") as stream:
            stream.write(b" ")
        with pytest.raises(ValueError):
            collect("2" * 40)
        failed, code = action("verify")
        assert code == 2 and failed["reason"] == "TERMINAL_VERIFICATION_FAILED"
        assert len(calls) == 10


def test_supply_deficit_prevents_device_creation(tmp_path, monkeypatch):
    from oeis_learn.pipeline_validation import progressive_round028_timing as host

    parent = tmp_path / host.PANEL_ARCHIVE
    parent.parent.mkdir(parents=True)
    parent.touch()
    monkeypatch.setattr(
        host,
        "verify_panels",
        lambda path: {"action": "EXTEND_FRESH_SUPPLY", "deficits": {"oeis": 1}},
    )
    raw, code = host.timing_command(
        None, None, tmp_path, tmp_path, "1" * 40, None, lambda *a: pytest.fail("device opened")
    )
    assert code == 2 and raw["reason"] == "EXTEND_FRESH_SUPPLY" and raw["gpu_dispatches"] == 0
    assert not (tmp_path / host.AREA).exists()
