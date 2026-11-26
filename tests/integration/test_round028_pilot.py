"""Operator-facing child pilot contracts, independent of a physical GPU."""

import copy

import pytest
from tests.integration.test_round028_terminal import collection as collection  # noqa: F401
from tests.integration.test_round028_receipts import original as original  # noqa: F401

from oeis_learn.generalization.progressive.contracts import digest, seal
from oeis_learn.generalization.progressive.program_search_pilot import (
    allocation,
    pilot_registration,
)
from oeis_learn.pipeline_validation.progressive_round028_pilot import (
    outcome_channels,
    research_rosters,
)


def parent():
    return dict(
        status="VERIFIED",
        evidence_verdict="INCOMPLETE",
        next_action="RETAIN_BASELINE",
        report=digest("old"),
        audit=digest("audit"),
        timing=digest("timing"),
        workstation={"status": "VERIFIED", "identity": digest("receipt")},
        comparison={"fresh_scored_cases": 0, "assessment": "SEALED"},
        funding={
            "pilot_us": 37296952335,
            "eight_case_batch_pilot_us": 40167927767,
            "repeat_if_earned_us": 40308089268,
        },
    )


def test_larger_allocation_uses_measured_pilot_alone_and_keeps_original_decision():
    p = parent()
    saved = copy.deepcopy(p)
    assert allocation(p)["headroom_us"] == 2303047665
    assert allocation(p)["eight_case_batch_headroom_us"] == -567927767
    assert allocation(p)["repeat_funded"] is False
    a = pilot_registration(p, "1" * 40, digest("source"))
    assert a["predecessor"] == p["report"] and a == seal(a) and p == saved
    assert pilot_registration(p, "2" * 40, digest("other"))["identity"] != a["identity"]
    bad = copy.deepcopy(p)
    bad["funding"]["pilot_us"] = 11 * 3600 * 1000000 + 1
    with pytest.raises(ValueError, match="exceeds"):
        pilot_registration(bad, "1" * 40, digest("source"))


def test_training_scoring_and_visible_inputs_have_disjoint_authority():
    from oeis_learn.generalization.progressive.graph import generator_look
    from tests.integration.test_progressive_lifecycle import generator_design_and_cuts

    panels = generator_design_and_cuts(0)[0]
    source = {
        name: [dict(id=row["task"], group=row["group"], stratum=row["stratum"]) for row in rows]
        for name, rows in panels.items()
    }
    source["assessment"] = [{"id": "never-open"}]
    assert research_rosters(source) == panels
    look = generator_look(
        panels, registration=digest("child"), pair=0, arm="ordinary", target=0, split="development"
    )
    witnesses = {
        r["task"]: {
            "id": r["task"],
            "indices": list(map(str, range(100))),
            "values": list(map(str, range(100))),
            "source": "PRIVATE",
        }
        for r in look["members"]
    }
    visible, truth = outcome_channels(look, witnesses)
    assert all(set(r) == {"id", "indices", "values"} and len(r["values"]) == 20 for r in visible)
    assert all(set(r) == {"id", "indices", "values"} and len(r["values"]) == 100 for r in truth)
    assert "never-open" not in {r["id"] for r in truth}


def test_cpu_training_boundary_reopens_all_state_for_the_same_registered_stream(tmp_path):
    import torch
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.generalization.progressive.training_rows import matched_stream
    from oeis_learn.generalization.progressive.trials import paired_generator_exposure
    from oeis_learn.generalization.progressive.contracts import atomic_write, reference
    from oeis_learn.pipeline_validation.progressive_pilot_worker import train
    from oeis_learn.pipeline_validation.progressive_round028_pilot import checkpoint_inputs
    from tests.unit.test_round028_training import training_rows
    from tests.unit.test_progressive_training_policy import profile

    torch.set_num_threads(1)
    rows = training_rows()
    reg = digest("pilot")
    stream = matched_stream([rows], {"probes": {}}, reg, 0)
    base = profile()
    model = resolve_profile(
        base["base_recipe"], shape=base["shape"], seed=stream["seeds"]["initialization"], dropout=0
    )
    package = seal(dict(rows=rows, stream=stream, profile=model, microbatch=4))
    from shutil import copyfile

    outputs = {}
    resumed = {}
    for arm in ("ordinary", "first_divergence"):
        work = tmp_path / arm
        (work / "output").mkdir(parents=True)
        path = work / "inputs/training.json"
        atomic_write(path, package)
        data = dict(
            training=reference(work, path),
            package=package["identity"],
            arm=arm,
            target=0,
            previous=None,
            registration=reg,
        )
        result = train(work, data, device="cpu")
        mounts, weight = checkpoint_inputs(work / "output", result)
        assert len(mounts) > 8 and weight.is_file()
        outputs[arm] = result["milestone"]
        # Simulate work charged after the last persisted model, then run one real 32-row update.
        next_work = tmp_path / (arm + "-recovered")
        (next_work / "output").mkdir(parents=True)
        for source_file, mount in [(path, "/work/inputs/training.json"), *mounts]:
            destination = next_work / mount.removeprefix("/work/")
            destination.parent.mkdir(parents=True, exist_ok=True)
            copyfile(source_file, destination)
        updated = train(
            next_work,
            dict(data, target=1_000_000, previous=result["checkpoint"], resume_charge=999_999),
            device="cpu",
        )
        checkpoint_inputs(next_work / "output", updated)
        assert updated["updates"] == 1
        assert updated["milestone"]["discarded"] == 999_999
        assert updated["milestone"]["charged"] == 999_999 + updated["milestone"]["useful_tokens"]
        resumed[arm] = updated
    assert paired_generator_exposure(*outputs.values()) == "MATCHED"
    assert (
        resumed["ordinary"]["weights"]["sha256"] != resumed["first_divergence"]["weights"]["sha256"]
    )


def test_pilot_projection_and_independent_spending_match_established_reducer():
    from oeis_learn.generalization.progressive.program_search_pilot import (
        family_counts,
        summary,
        spending,
    )
    from oeis_learn.pipeline_validation.progressive_study import generator_observation
    from tests.integration.test_progressive_lifecycle import generator_design_and_cuts

    panels, reg, cuts = generator_design_and_cuts(4_000_000)
    roster = panels["development"] + panels["development_retention"]
    outputs = {}
    for arm in ("ordinary", "first_divergence", "incumbent"):
        outputs[arm] = {
            r["task"]: dict(
                outcome="success"
                if arm == "first_divergence" and r["stratum"] == "composition"
                else "failure",
                complete=True,
            )
            for r in roster
        }
    for unknown in (False, True):
        if unknown:
            for r in roster:
                if r["stratum"] == "retention":
                    outputs["ordinary"][r["task"]] = dict(outcome="unresolved", complete=False)
        obs = generator_observation(
            panels,
            outputs,
            cuts,
            registration=reg,
            pair=0,
            target=4_000_000,
            funded=False,
            comparable_cost=True,
        )
        projection = family_counts(roster, outputs, list(outputs))
        action = spending(projection, True, True)
        assert (
            action
            == obs["action"]
            == ("UNRESOLVED_SPENDING_CONTRAST" if unknown else "REPLICATION_UNFUNDED")
        )
        assert summary(projection, 3)["first_divergence_minus_ordinary"]["gain"] == 0.5
        assert not any(row["task"] in str(projection) for row in roster)


def test_interrupt_keeps_completed_choices_and_never_reopens_a_started_case(tmp_path):
    from oeis_learn.generalization.progressive.contracts import atomic_write
    from oeis_learn.pipeline_validation.pilot_clocks import DESCRIPTIVE, clock_record
    from oeis_learn.pipeline_validation.progressive_pilot_recovery import recover_choices

    rows = [
        dict(id=k, indices=list(map(str, range(20))), values=["0"] * 20)
        for k in ("finished", "mid-fit", "future")
    ]
    output = tmp_path / "output"
    model = digest("weights")
    for p in rows[:2]:
        folder = output / "case-boundaries" / digest(p["id"])[7:]
        start = seal(dict(prompt=digest(p), model=model, policy=DESCRIPTIVE))
        atomic_write(folder / "start.json", start)
        if p is rows[0]:
            clock = clock_record(
                prompt=p["id"],
                mode="correct",
                snapshot=model,
                start=1.0,
                scope_deadline=400.0,
                finish=3.0,
                expected=1,
                started=1,
                terminal=1,
                policy=DESCRIPTIVE,
            )
            atomic_write(
                folder / "choice.json", seal(dict(start, record={"selected": None}, clock=clock))
            )
    part, pending, unknown = recover_choices(
        tmp_path, output, "original", rows, {"sha256": model}, {}, "ordinary", 0
    )
    assert unknown == ["mid-fit"] and pending == [rows[2]]
    assert list(part[2]["records"]) == ["finished"]
    assert (
        part
        == recover_choices(
            tmp_path, output, "original", rows, {"sha256": model}, {}, "ordinary", 0
        )[0]
    )


def test_archive_recomputes_all_three_looks_and_catches_changed_counts(collection, tmp_path):
    import io
    import json
    from oeis_learn.generalization.progressive.program_search_outcome import (
        build_outcome,
        verify_outcome,
        archive_bytes,
        workstation_receipt,
    )
    from oeis_learn.generalization.progressive.program_search_rehearsal_report import (
        encode,
        verify_transport,
    )
    from oeis_learn.generalization.progressive.program_search_pilot import (
        ARMS,
        LOOKS,
        family_counts,
        pilot_archive,
        spending,
        summary,
        verify_pilot,
    )
    from oeis_learn.generalization.progressive.graph import generator_look
    from oeis_learn.generalization.progressive.training_rows import pair_seeds
    from tests.integration.test_progressive_lifecycle import generator_design_and_cuts

    _, builder = collection
    result = build_outcome(builder("prospective", cold_overhead_us=4_300_000_000))
    receipt = verify_outcome(io.BytesIO(result))
    workstation = workstation_receipt(receipt, digest("all inspected"), digest("original journal"))
    prev = archive_bytes({"workstation.json": encode(workstation), "round028.zip": result})
    reg = pilot_registration(
        dict(receipt, workstation=workstation), "1" * 40, digest("child source")
    )
    looks, costs = [], []
    for target in LOOKS:
        panels, _, cuts = generator_design_and_cuts(target)
        cuts = {
            a: seal(dict(c, registration=reg["identity"], seeds=pair_seeds(reg["identity"], 0)))
            for a, c in cuts.items()
        }
        roster = generator_look(
            panels,
            registration=reg["identity"],
            pair=0,
            arm=ARMS[0],
            target=target,
            split="development",
        )["members"]
        arms = (*ARMS, "incumbent") if target == LOOKS[-1] else ARMS
        outcomes = {
            a: {r["task"]: {"outcome": "failure", "complete": True} for r in roster} for a in arms
        }
        for arm in arms:
            for role in ("search", "score") if arm == "incumbent" else ("train", "search", "score"):
                costs.append(
                    seal(
                        dict(
                            block=f"{arm}-{target}",
                            role=role,
                            lease_us=13 if role != "score" else 0,
                            wall_us=23,
                            charged_us=23,
                            covered_us=0,
                            busy_us=None,
                        )
                    )
                )
        stats = family_counts(roster, outcomes, arms)
        looks.append(
            dict(
                target=target,
                cuts=cuts,
                counts=stats,
                comparable=True,
                zero_reused=False,
                contrasts=summary(stats, len(arms)),
                action=spending(stats, True, True)
                if target == LOOKS[-1]
                else "CONTINUE_TO_RESERVED_ENDPOINT",
            )
        )
    public = seal(
        dict(
            version="round028-generator-pilot/v1",
            registration=reg,
            freeze=digest("rosters"),
            acceptance=digest("hardware"),
            looks=looks,
            action=looks[-1]["action"],
            research_gpu_us=sum(c["lease_us"] for c in costs),
            research_role_us=sum(c["charged_us"] for c in costs),
            costs=costs,
            verification="private_choices_and_checkpoints",
        )
    )
    packed = pilot_archive(public, prev)
    path = tmp_path / "round028-pilot.zip"
    path.write_bytes(packed)
    assert verify_transport(path) == verify_pilot(io.BytesIO(packed))
    assert verify_pilot(path)["next_action"] == "RETAIN_REFERENCE"
    with pytest.raises(ValueError):
        verify_pilot(io.BytesIO(packed + b"additional payload"))
    forged = json.loads(encode(public))
    forged["costs"][0] = seal(dict(forged["costs"][0], charged_us=17))
    with pytest.raises(ValueError, match="cost"):
        verify_pilot(io.BytesIO(pilot_archive(seal(forged), prev)))
    changed = json.loads(encode(public))
    changed["looks"][-1]["counts"][0]["counts"] = {"SSS": 1}
    with pytest.raises(ValueError):
        verify_pilot(io.BytesIO(pilot_archive(seal(changed), prev)))


def test_child_owner_runs_three_looks_reuses_boundaries_and_leaves_parent_unchanged(
    collection, tmp_path, monkeypatch
):
    import io
    from pathlib import Path
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive.program_search_outcome import (
        build_outcome,
        verify_outcome,
        workstation_receipt,
        archive_bytes,
        verify_workstation,
    )
    from oeis_learn.generalization.progressive.program_search_rehearsal_report import encode
    from oeis_learn.generalization.progressive.program_search_pilot import LOOKS, verify_pilot
    from oeis_learn.generalization.progressive.training_rows import pair_seeds
    from oeis_learn.generalization.progressive.contracts import reference, atomic_write
    from oeis_learn.pipeline_validation import (
        progressive_round028_pilot as pilot,
        progressive_round028_rehearsal as probe,
    )
    from oeis_learn.pipeline_validation.pilot_clocks import clock_record, CORRECT_FULL, DESCRIPTIVE
    from tests.integration.test_progressive_lifecycle import generator_design_and_cuts

    original, builder = collection
    old = build_outcome(builder("repriced", cold_overhead_us=4_300_000_000))
    older = verify_outcome(io.BytesIO(old))
    prev = archive_bytes(
        {
            "workstation.json": encode(
                workstation_receipt(older, digest("replay"), digest("boundary"))
            ),
            "round028.zip": old,
        }
    )
    parent = verify_workstation(io.BytesIO(prev))
    monkeypatch.setattr(pilot, "parent_evidence", lambda *a: (prev, parent))
    panels = generator_design_and_cuts(0)[0]
    donors = {
        p: [dict(id=r["task"], group=r["group"], stratum=r["stratum"]) for r in members]
        for p, members in panels.items()
    }
    witnesses = {
        r["id"]: dict(id=r["id"], indices=list(map(str, range(100))), values=["0"] * 100)
        for values in donors.values()
        for r in values
    }
    private = seal(dict(allocations=dict(panels=donors), witnesses=witnesses))
    panel_area = tmp_path / "panel"
    atomic_write(panel_area / "supply.json", private)
    weights = tmp_path / "weights.pt"
    weights.write_bytes(b"identified weights")
    monkeypatch.setattr(
        pilot, "original_inputs", lambda *a: (private, {}, {"recipe": "D"}, weights, panel_area)
    )
    monkeypatch.setattr(
        pilot,
        "hardware_qualification",
        lambda *a: {p: {"image": digest(p)} for p in ("cpu", "hip")},
    )
    monkeypatch.setattr(
        pilot, "accept_pilot", lambda *a: {"installed": digest("on exposed fixtures")}
    )
    monkeypatch.setattr(probe, "candidate_image", lambda *a: (digest(a[-1]), 0, True))
    monkeypatch.setattr(probe, "physical_scope", lambda h, p, **scope: digest(p))
    monkeypatch.setattr(pilot, "checkpoint_inputs", lambda *a: ([], weights))

    def training(root, area, reg, *unused):
        from oeis_learn.pipeline_validation.preparation_recovery import retain

        pkg = seal(dict(profile={"recipe": "candidate"}, stream=seal({"seed": 1})))
        retain(area / "training.json", pkg)
        return pkg

    monkeypatch.setattr(pilot, "prepare_training", training)
    calls = []

    class Role:
        phase = "cpu"

        def __init__(self, args, source, root, revision, area, registration, factory):
            self.area, self.registration = area, registration

        def __call__(self, role, data, *, seconds, inputs):
            from oeis_learn.pipeline_validation.preparation_recovery import retain

            out = self.area / "test-output" / (data["block"] + "-" + role)
            if not out.is_dir():
                calls.append((data["block"], role))
            out.mkdir(parents=True, exist_ok=True)
            self.last_wall_us = 2000
            self.last_output, self.last_lease_us = out, 1000 if self.phase == "hip" else 0
            if role == "train":
                arm, target = data["arm"], data["target"]
                c = generator_design_and_cuts(target)[2][arm]
                cut = seal(
                    dict(
                        c,
                        registration=self.registration["identity"],
                        seeds=pair_seeds(self.registration["identity"], 0),
                    )
                )
                return dict(
                    milestone=cut,
                    checkpoint={"id": digest([arm, target])},
                    weights=reference(tmp_path, weights),
                )
            if role == "search":
                full = data["target"] == LOOKS[-1]
                policy = CORRECT_FULL if full else DESCRIPTIVE
                ordered = sorted(data["rows"], key=lambda r: r["id"])
                clocks = [
                    clock_record(
                        prompt=r["id"],
                        mode="correct",
                        snapshot=digest("model"),
                        start=1.0,
                        scope_deadline=400.0,
                        finish=3.0,
                        expected=8 if full else 1,
                        started=8 if full else 1,
                        terminal=8 if full else 1,
                        policy=policy,
                    )
                    for r in ordered
                ]
                output = seal(
                    dict(
                        version="discovery-candidates/v1",
                        checkpoint=data["weights"]["sha256"],
                        recipe=digest(data["recipe"]),
                        prompts=digest(ordered),
                        model=digest("model"),
                        records={r["id"]: {} for r in ordered},
                        clock=seal(
                            dict(
                                policy=policy,
                                work=digest("model"),
                                clocks=clocks,
                                cost_complete=True,
                            )
                        ),
                    )
                )
                retain(out / "candidates.json", output)
                return {"candidates": reference(out, out / "candidates.json")}
            output = seal(
                dict(
                    version="discovery-scoring/v1",
                    denominator=len(data["rows"]),
                    candidates=data["candidates"]["sha256"],
                    outcomes={
                        r["id"]: {
                            "equal_budget": {
                                "success": False,
                                "complete": True,
                                "scoring": {"complete": True},
                            }
                        }
                        for r in data["rows"]
                    },
                )
            )
            retain(out / "scores.json", output)
            return {"scores": reference(out, out / "scores.json")}

    monkeypatch.setattr(pilot, "PilotRole", Role)
    args = (
        None,
        Path(__file__).resolve().parents[2],
        original.workspace,
        original.folder,
        "a" * 40,
        original.store,
        lambda *a, **kw: SimpleNamespace(policy={"free_disk_floor_bytes": 0}),
    )
    initial = original.store.tip
    result, code = pilot.run_pilot(*args)
    assert code == 0 and result["receipt"]["next_action"] == "RETAIN_REFERENCE"
    assert len(calls) == 6 + 6 + 6 and len([v for v in calls if v[0].startswith("incumbent")]) == 2
    assert verify_pilot(original.folder / pilot.SHARE) == result["receipt"]

    def unexpected_host(*unused, **options):
        raise AssertionError("a completed pilot must re-export without acquiring the workstation")

    assert pilot.run_pilot(*args[:-1], unexpected_host) == (result, 0)
    assert len(calls) == 18 and original.store.tip == initial
    cost = next((original.folder / pilot.AREA / "costs").glob("*.json"))
    cost.write_text("{}")
    with pytest.raises(ValueError, match="publication boundary"):
        pilot.run_pilot(*args[:-1], unexpected_host)


def test_scorer_recovery_debits_original_time_and_reopens_the_same_sealed_choices(monkeypatch):
    from oeis_learn.pipeline_validation import progressive_pilot_recovery as recovery

    data = dict(
        block="ordinary-4000000-resume-1", candidates=digest("sealed choices"), rows=["case"]
    )
    mounts = [("original", "same")]
    runs, settled = [], []

    def prior(cpu, role, request, inputs):
        assert role == "score" and request["candidates"] == data["candidates"] and inputs == mounts
        return (None, 0, 17) if request["block"] == data["block"] else None

    def score(cpu, role, request, inputs):
        runs.append(request)
        return {"checked": True}

    monkeypatch.setattr(recovery, "interrupted", prior)
    assert recovery.measured_score(
        score, lambda *args: settled.append(args), object(), data, mounts
    ) == {"checked": True}
    assert settled == [(data["block"], "score", 0, 17)]
    assert runs == [dict(data, block=data["block"] + "-scoreretry-1")]
