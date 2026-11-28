"""Signed synthetic history exercises the operator's multi-control receipt reader."""

import hashlib
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import zipfile

import pytest

from oeis_learn.generalization.progressive import campaign as campaigns
from oeis_learn.generalization.progressive.cohorts import STRATA
from oeis_learn.pipeline_validation.progressive_round028_audit import METHODS, ORIGINAL
from oeis_learn.generalization.progressive.contracts import atomic_write, reference, seal

OPERATOR = (
    Path(__file__).resolve().parents[2]
    / "specs/016-measure-learning-velocity/validation/qualify_continuation.py"
)


history = SimpleNamespace(**runpy.run_path(str(OPERATOR)))


def private_history(root):
    members, producers = {}, {}
    for block in range(8):
        keys = [f"private-{block}-{i}" for i in range(16)]
        members.update({k: STRATA[i % 4] for i, k in enumerate(keys)})
        for model in history.CONTROL_MODELS:
            tag = f"capability-{model}-{block:02d}"
            costs = {m: {"total_elapsed_us": 12, "attempts": 1} for m in METHODS}
            costs["template"] = dict(elapsed_seconds=0.00001, details=[{}] * 8)
            upstream = dict(complete=True, reason="synthetic")
            records = {
                k: dict(
                    costs=costs,
                    selections=dict.fromkeys(METHODS),
                    arm_evidence=dict.fromkeys(METHODS, upstream),
                )
                for k in keys
            }
            gen = seal(
                dict(
                    version="discovery-candidates/v1",
                    model=tag,
                    model_control=model,
                    methods=list(METHODS),
                    records=records,
                )
            )
            path = root / (tag + ".json")
            atomic_write(path, gen)
            gref = reference(root, path)
            score = dict(
                candidate_sha256=None,
                success=False,
                upstream=upstream,
                scoring=dict(elapsed_seconds=0),
            )
            scores = seal(
                dict(
                    version="discovery-scoring/v1",
                    candidates=gref["sha256"],
                    denominator=len(keys),
                    outcomes={k: dict.fromkeys(METHODS, score) for k in keys},
                )
            )
            path = root / (tag + "-scores.json")
            atomic_write(path, scores)
            producers[tag] = dict(generation=gref, scoring=reference(root, path))
    return members, producers


def test_complete_two_independent_cohorts_keep_three_clocks_and_no_case_values(tmp_path):
    members, refs = private_history(tmp_path)
    result = history.control_prices(tmp_path, members, refs)
    assert result["status"] == "COMPLETE"
    assert "private-" not in json.dumps(result)
    for method, cohort in result["controls"].items():
        assert set(cohort) == set(history.CONTROL_MODELS)
        for observation in cohort.values():
            assert observation["partitions"] == 8
            for cell in observation["strata"].values():
                assert (cell["expected_cases"], cell["seen"], cell["issued"], cell["empty"]) == (
                    32,
                ) * 4
                assert cell["paired_cases"] == 32
                assert cell["mean_us"] == cell["max_us"] == (10 if method == "template" else 12)
                assert cell["scoring_us"] == 0


def test_missing_file_retains_other_models_and_incomplete_clock_stays_unknown(tmp_path):
    members, refs = private_history(tmp_path)
    tag = "capability-retained_D-00"
    (tmp_path / refs[tag]["scoring"]["path"]).unlink()
    result = history.control_prices(tmp_path, members, refs)
    assert result["status"] == "PARTIAL"
    for row in result["controls"].values():
        assert row["retained_D"]["partitions"] == 7
        assert row["random"]["partitions"] == 8
        for cell in row["retained_D"]["strata"].values():
            assert cell["paired_cases"] == 28 and cell["mean_us"] is None and cell["max_us"] is None


def test_tampered_repeated_and_foreign_parent_cannot_supply_a_price(tmp_path):
    members, refs = private_history(tmp_path)
    tag = "capability-retained_D-01"
    original = refs[tag]
    refs[tag] = refs["capability-retained_D-00"]
    with pytest.raises(ValueError):
        history.control_prices(tmp_path, members, refs)
    refs[tag] = dict(original, scoring=refs["capability-retained_D-02"]["scoring"])
    with pytest.raises(ValueError):
        history.control_prices(tmp_path, members, refs)
    refs[tag] = original
    (tmp_path / original["generation"]["path"]).write_text("{}")
    with pytest.raises(ValueError):
        history.control_prices(tmp_path, members, refs)


def test_absent_and_short_clocks_keep_partial_totals_without_inventing_a_rate():
    cell = history.new_cell(2)
    score = dict(candidate_sha256=None, success=False, scoring={}, upstream={"complete": False})
    gen = dict(
        costs={"template": dict(elapsed_seconds=1, details=[{}])},
        selections={"template": None},
        arm_evidence={"template": score["upstream"]},
    )
    history.observe(cell, "template", gen, score)
    gen["costs"]["template"] = dict(elapsed_seconds=0, details=[{}] * 8)
    score["scoring"] = dict(elapsed_seconds=0)
    history.observe(cell, "template", gen, score)
    history.finish_cell(cell)
    assert (
        cell["search_us"] == 1_000_000 and cell["search_clocks"] == 2 and cell["score_clocks"] == 1
    )
    assert cell["issued"] == cell["paired_cases"] == 1 and cell["mean_us"] is None
    score["scoring"]["elapsed_seconds"] = -1
    with pytest.raises(ValueError):
        history.observe(history.new_cell(1), "template", gen, score)


def test_operator_packages_three_controls_and_the_verified_wave(tmp_path, monkeypatch):
    members, refs = private_history(tmp_path)
    operator = runpy.run_path(str(OPERATOR))
    monkeypatch.setitem(
        operator["donor_control_history"].__globals__,
        "read_history",
        lambda root: history.control_prices(root, members, refs),
    )
    calls = []

    def action(name, *, at=None):
        calls.append(name)
        if name == "verify":
            assert at == "b" * 40
            return dict(
                mechanical_status="VERIFIED",
                verification_scope="private_checkpoint_matrix",
                scientific_verdict=None,
                report_ref="matrix",
                last_committed_ref="matrix",
            )
        assert name == "preflight" and at is None
        return dict(
            reason="fixture",
            next_action="inspect",
            inventory=dict(
                gpu_dispatches=0,
                capacity=dict(registration=None, mounts=None),
                mount=dict(available_bytes=100),
                prices=dict(
                    supported_wave=dict(
                        methods=["00", "04", "05", "06"], checkpoint_receipt="matrix"
                    ),
                    checkpoint_cycles=dict(
                        candidate="b" * 40,
                        receipt="matrix",
                        methods={"00": dict(checkpoint_bytes=4, checkpoint_inodes=2)},
                    ),
                ),
            ),
        )

    report = operator["donor_control_history"](action, tmp_path, "a" * 40)
    assert report["status"] == "COMPLETE" and sorted(calls) == ["preflight", "verify"]
    out = tmp_path / "out.zip"
    operator["write_readiness_zip"](out, report)
    with zipfile.ZipFile(out) as archive:
        summary = json.loads(archive.read("summary.json"))
        for name, sha in summary["sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == sha
        assert set(summary["sha256"]) == {"control-" + m + ".json" for m in METHODS} | {
            "coverage.json",
            "forecast.json",
            "capacity.json",
            "verification.json",
        }
        assert summary["gpu_dispatches"] == 0 and summary["grant_minutes"] == 0

    def stale(name, *, at=None):
        reply = action(name, at=at)
        if name == "verify":
            reply["report_ref"] = "unrelated"
        return reply

    report = operator["donor_control_history"](stale, tmp_path, "a" * 40)
    assert report["status"] == "PARTIAL" and "forecast.json" not in report["files"]
    assert "capacity.json" in report["files"] and "control-template.json" in report["files"]


def test_frozen_study_and_capability_cut_bind_the_real_operator_read_path(tmp_path, monkeypatch):
    members, producers = private_history(tmp_path)

    def publish(name, data):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, data)
        return reference(tmp_path, path)

    roster = [dict(task=task, stratum=s, group=task) for task, s in members.items()]
    study = seal(dict(evaluations={"trial-05-1000000-development-score": dict(roster=roster)}))
    binding = dict(
        name="study-policy", role="proposal", kind="policy", artifact=publish("study.json", study)
    )
    campaign = seal(
        dict(payload=dict(references=[binding], tasks=[dict(trial=dict(study=study["identity"]))]))
    )
    publish(ORIGINAL + "/prepare-research/frozen/campaign.json", campaign)
    outputs = {}
    for tag, metrics in producers.items():
        for operation, key, metric in (
            ("generate", "generation", "candidates"),
            ("score", "scoring", "scores"),
        ):
            task = tag + "-" + operation
            receipt = seal(
                dict(
                    operation=operation,
                    mechanical_status="COMPLETE",
                    task=task,
                    campaign=campaign["identity"],
                    metrics={metric: metrics[key]},
                )
            )
            outputs[task] = dict(
                task=task, operation=operation, result=publish(task + "-receipt.json", receipt)
            )
    name = "original.json"
    bound = publish(
        "run-state/discovery-control/" + campaign["identity"][7:] + "/" + name,
        seal(dict(state=dict(outputs=outputs))),
    )
    policy = dict(
        campaign=campaign["identity"],
        capability_cut=dict(name=name, bytes=bound["bytes"], sha256=bound["sha256"]),
    )
    monkeypatch.setattr(campaigns, "learned_search_policy", lambda: policy)
    actual = history.read_history(tmp_path)
    assert (
        actual == history.control_prices(tmp_path, members, producers)
        and actual["status"] == "COMPLETE"
    )
    policy["capability_cut"]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError):
        history.read_history(tmp_path)


def test_selected_case_requires_bound_choice_and_finished_scoring_for_a_whole_rate():
    from oeis_learn.fresh.synthesis_policy import select_visible
    from oeis_learn.synthesis.admission import source_identity

    body = "(program (return $index))"
    candidate = dict(
        source=body, source_sha256=source_identity(body), verified=True, visible_count=20, cost=0
    )
    record = dict(
        costs={"polynomial": dict(total_elapsed_us=2, attempts=1)},
        selections={"polynomial": dict(selected=select_visible([candidate]))},
        arm_evidence={"polynomial": dict(complete=True)},
    )
    score = dict(
        candidate_sha256=candidate["source_sha256"],
        success=False,
        scoring=dict(elapsed_seconds=0.125),
    )
    cell = history.new_cell(1)
    history.observe(cell, "polynomial", record, score)
    history.finish_cell(cell)
    assert cell["mean_us"] is None and cell["empty"] == 0 and cell["score_clocks"] == 1
    score["scoring"]["complete"] = True
    cell = history.new_cell(1)
    history.observe(cell, "polynomial", record, score)
    history.finish_cell(cell)
    assert cell["mean_us"] == cell["max_us"] == 125002
    score["candidate_sha256"] = None
    with pytest.raises(ValueError):
        history.observe(history.new_cell(1), "polynomial", record, score)
