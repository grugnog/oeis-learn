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
    outputs = {}
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
    assert paired_generator_exposure(*outputs.values()) == "MATCHED"


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
    looks = []
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
        stats = family_counts(roster, outcomes, arms)
        looks.append(
            dict(
                target=target,
                cuts=cuts,
                counts=stats,
                comparable=True,
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
            research_gpu_us=13,
            verification="private_choices_and_checkpoints",
        )
    )
    packed = pilot_archive(public, prev)
    path = tmp_path / "round028-pilot.zip"
    path.write_bytes(packed)
    assert verify_transport(path) == verify_pilot(io.BytesIO(packed))
    assert verify_pilot(path)["next_action"] == "RETAIN_REFERENCE"
    changed = json.loads(encode(public))
    changed["looks"][-1]["counts"][0]["counts"] = {"SSS": 1}
    with pytest.raises(ValueError):
        verify_pilot(io.BytesIO(pilot_archive(seal(changed), prev)))
