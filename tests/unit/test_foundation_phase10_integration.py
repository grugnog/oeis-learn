"""Executable phase10 integration; these CPU fixtures cannot qualify GPU measurements."""

import json
import pytest
import torch
from oeis_learn.experiments.pilot_models import pilot_profile, construct_pilot_models
from oeis_learn.experiments.artifacts import (
    canonical_bytes,
    compute_file_hash,
    compute_canonical_digest as digest,
)
from oeis_learn.decoder.program_codec import encode_body, BOS_ID, PAD_ID


def models():
    torch.manual_seed(7001)
    encoder, decoder = construct_pilot_models(pilot_profile(), "cpu")
    encoder.eval()
    decoder.eval()
    return encoder, decoder


def test_cached_batched_sdpa_frozen_streams_are_equivalent():
    from oeis_learn.evaluation.engineering import inference_stream, equivalent_logits

    e, d = models()
    rows = [
        dict(tokens=encode_body(f"i256.const {i}"), entry=dict(conditioning_values=[str(i)] * 20))
        for i in (1, 12, -6, 100)
    ]
    baseline = inference_stream(e, d, rows, cached=False, backend="eager", batch=1)
    for cached, backend, batch in ((True, "eager", 1), (True, "eager", 4), (True, "sdpa", 4)):
        actual = inference_stream(e, d, rows, cached=cached, backend=backend, batch=batch)
        assert actual[1] == baseline[1]
        for left, right in zip(baseline[0], actual[0]):
            equivalent_logits(left, right, (torch.ones(1), torch.ones(1)))
        assert actual[2]["encoder"] > 0 and actual[2]["decoder"] > 0
    assert d.attention_backend == "eager"


def test_actual_supervised_update_measures_entropy_phases_and_changes_parameters():
    from oeis_learn.rl.sft_trainer import supervised_step

    e, d = models()
    before = next(d.parameters()).detach().clone()
    optimizer = torch.optim.AdamW([*e.parameters(), *d.parameters()], lr=3e-4)
    result = supervised_step(
        e,
        d,
        optimizer,
        [[7] * 20],
        torch.tensor([[BOS_ID] + encode_body("i256.const 7")]),
        device="cpu",
        pad_id=PAD_ID,
    )
    assert result["entropy"] > 0 and result["loss"] > 0
    assert set(result["phase_timings_ns"]) == {
        "encoder",
        "decoder_forward",
        "objective",
        "backward_optimizer",
    }
    assert all(v > 0 for v in result["phase_timings_ns"].values())
    assert not torch.equal(before, next(d.parameters()))
    assert result["sensors"]["thermal"]["state"] == "unavailable"


def test_pilot_checkpoint_tampering_and_resume_same_update(tmp_path):
    from oeis_learn.tracking.role_worker import train, load_state
    from oeis_learn.evaluation.live_measurement import write_view

    settings = dict(seed=7001, device="cpu", profile=pilot_profile(), seconds=6, update_cap=1)
    row = dict(
        tokens=encode_body("i256.const 7"),
        source="i256.const 7",
        entry=dict(conditioning_values=["7"] * 20),
    )
    inputs = write_view(
        tmp_path / "input", {"settings.json": settings, "examples.json": {"rows": [row]}}
    )
    output = tmp_path / "output"
    output.mkdir()
    result = train(inputs, output)
    assert result["completed_updates"] == 1
    state = load_state(output, "cpu")
    assert state["optimizer"]["state"] and "python_rng" in state and "torch_rng" in state
    resumed = write_view(
        tmp_path / "resume-input",
        {"settings.json": settings, "examples.json": {"rows": [row]}},
        {"state.pt": output / "state.pt", "state-manifest.json": output / "state-manifest.json"},
    )
    second = tmp_path / "second"
    second.mkdir()
    assert train(resumed, second)["completed_updates"] == 1
    for key, value in state["decoder"].items():
        assert torch.equal(value, load_state(second, "cpu")["decoder"][key])
    with (second / "state.pt").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="final bytes"):
        load_state(second, "cpu")


def test_failed_diagnostic_is_immediately_charged_and_never_reissued(tmp_path):
    from oeis_learn.evaluation.live_measurement import run_pilot
    from oeis_learn.experiments.decisions import PilotSchedule
    from oeis_learn.evaluation.experiment_runner import fixture_manifest

    class Coordinator:
        def run(self, *args, **kwargs):
            raise RuntimeError("fixture containment failure")

    manifest = fixture_manifest()
    with pytest.raises(RuntimeError, match="containment"):
        run_pilot(manifest, {"examples": {"rows": []}}, tmp_path, Coordinator())
    schedule = PilotSchedule(tmp_path / "pilot-events", digest(manifest))
    assert schedule.interrupted and schedule.charged > 0 and schedule.pending is None
    with pytest.raises(ValueError, match="interrupted"):
        schedule.issue()


def test_source_release_replays_original_files_and_rejects_changes(tmp_path):
    from oeis_learn.evaluation.release_source import prepare_release, audit_release
    from oeis_learn.discovery.program_archive import ProgramArchive
    from oeis_learn.data.structural_splits import StructuralHoldouts
    from oeis_learn.evaluation.foundation_cohort import load_cohort, read_ref

    source = tmp_path / "indexed"
    source.mkdir()
    records = []
    for number in range(2):
        terms = source / f"b{number}.txt"
        terms.write_text("\n".join(f"{i - 3} {i ** (number + 2)}" for i in range(120)))
        meta = source / f"m{number}.json"
        meta.write_text(
            json.dumps(
                dict(
                    offset=-3,
                    keywords=["sign"],
                    finite=False,
                    table=False,
                    locator=f"https://example.invalid/source/{number}",
                )
            )
        )
        records.append(
            dict(
                record_id=f"fixture:A{number:06}",
                terms=dict(path=terms.name, sha256=compute_file_hash(terms)),
                metadata=dict(path=meta.name, sha256=compute_file_hash(meta)),
            )
        )
    manifest = source / "manifest.json"
    manifest.write_bytes(
        canonical_bytes(
            dict(version="indexed-source/v1", release="explicit test fixture", records=records)
        )
    )
    config = tmp_path / "cohort.yaml"
    config.write_text(
        "schema_version: foundation/v1\nprofile: prefix20_total100_affine_v2\nseed: 3\ndev_count: 1\nfinal_count: 1\n"
    )
    output = tmp_path / "release"
    assert prepare_release(manifest, config, output)["status"] == "pending"
    cohort = load_cohort(output / "cohort")
    archive = ProgramArchive(
        tmp_path / "archive",
        holdouts=StructuralHoldouts(
            reserved_prefixes=frozenset(read_ref(output / "cohort", cohort["prefix_membership"]))
        ),
    )
    with pytest.raises(ValueError, match="empty"):
        audit_release(output / "cohort", archive=archive)
    audit = audit_release(output / "cohort")
    assert audit["metadata_verified"] and audit["grouping_replayed"]
    (output / "cohort/source-inputs/b0.txt").write_text("0 999")
    with pytest.raises(ValueError, match="digest"):
        audit_release(output / "cohort")


def test_baseline_labels_and_empty_junit_cannot_qualify(tmp_path):
    from oeis_learn.evaluation.baseline_gate import validate_preflight, require_junit

    with pytest.raises(ValueError, match="actual-device"):
        validate_preflight({"status": "ready", "gate": "passed"})
    junit = tmp_path / "suite.xml"
    junit.write_text("<testsuites><testsuite/></testsuites>")
    with pytest.raises(ValueError, match="empty"):
        require_junit(tmp_path, dict(path=junit.name, sha256=compute_file_hash(junit)), "G5")


def test_live_qualification_rejects_status_only_and_missing_role_evidence(tmp_path):
    from oeis_learn.evaluation.live_measurement import validate_live_evidence
    from oeis_learn.evaluation.baseline_gate import require_containment

    with pytest.raises(ValueError, match="executed"):
        validate_live_evidence({"qualified": True}, {}, tmp_path)
    with pytest.raises(ValueError, match="containment"):
        require_containment({"status": "complete"}, None)


def test_diagnostic_hypotheses_require_archive_admission_and_original_holdouts(tmp_path):
    from copy import deepcopy
    from oeis_learn.tracking.role_worker import admit_training_hypothesis
    from oeis_learn.evaluation.search import SearchSession
    from oeis_learn.sandbox.pipeline import Runtime
    from oeis_learn.data.structural_splits import structural_signature, split_for_structure

    source = "i256.const 7"
    seed = next(
        s for s in range(100) if split_for_structure(structural_signature(source), s) == "train"
    )
    policy = dict(seed=seed, reserved_compositions=[])

    class Runner:
        runtime = Runtime()

        def run_single(self, source, indices, **kwargs):
            return self.runtime.evaluate(source, indices)

    class Proposer:
        def stochastic(self, visible, seed, ledger, **kwargs):
            ledger.charge(expansions=1, tokens=len(encode_body(source)))
            return source, dict(
                method="greedy", encoder_sha256=digest("e"), decoder_sha256=digest("d")
            )

    runner = Runner()
    candidate = SearchSession([7] * 20, portfolio="greedy").run(Proposer(), runner)["results"][0]
    admitted = admit_training_hypothesis(
        candidate, [7] * 100, runner, tmp_path / "valid", holdouts=policy, request_id="valid"
    )
    assert admitted["entry"]["training_eligible"] and admitted["entry"][
        "validation_indices"
    ] == list(range(20, 100))
    for name, changed, values, holdouts in (
        ("mismatch", candidate, [7] * 20 + [8] * 80, policy),
        ("reserved", candidate, [7] * 100, dict(policy, reserved_prefixes=[digest(["7"] * 20)])),
        ("forged", deepcopy(candidate), [7] * 100, policy),
    ):
        if name == "forged":
            changed["provenance"]["hypothesis"]["encoder_sha256"] = digest("forged")
        assert (
            admit_training_hypothesis(
                changed, values, runner, tmp_path / name, holdouts=holdouts, request_id=name
            )
            is None
        )
        assert not (tmp_path / name / "index.json").exists()


def test_measurement_cli_reports_invalid_holdout_without_traceback(tmp_path, capsys):
    from oeis_learn.cli.foundation import cmd_prepare_measurement

    code = cmd_prepare_measurement(
        archive=str(tmp_path),
        cohort=str(tmp_path),
        gate_report="missing",
        learning_gate_report="missing",
        image="sha256:" + "a" * 64,
        output=str(tmp_path / "out"),
        baseline_report="missing",
        baseline_checkpoint="missing",
        holdouts=str(tmp_path / "absent"),
        as_json=True,
    )
    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "invalid"


def test_live_paired_scheduler_uses_one_contrast_and_isolated_evaluations(tmp_path):
    from oeis_learn.evaluation.live_measurement import run_pilot
    from oeis_learn.evaluation.experiment_runner import fixture_manifest

    tick = [1000]
    calls = []

    class Coordinator:
        def run(self, operation, *, inputs, output, seconds, **kwargs):
            calls.append(operation)
            output.mkdir(parents=True, exist_ok=True)
            if operation == "diagnostic":
                tick[0] += 120 * 10**9
                return {"diagnostic": {"encoder_collisions": 1}}
            if operation == "train":
                settings = json.loads((inputs / "settings.json").read_text())
                tick[0] += int(settings["seconds"] * 10**9)
                (output / "state.pt").write_bytes(b"test role transport placeholder")
                (output / "state-manifest.json").write_text("{}")
                return {"state_sha256": compute_file_hash(output / "state.pt")}
            if operation == "generate":
                assert set(p.name for p in inputs.iterdir()) == {
                    "settings.json",
                    "visible.json",
                    "state.pt",
                    "state-manifest.json",
                }
                assert (
                    len(json.loads((inputs / "visible.json").read_text())["groups"][0]["values"])
                    == 20
                )
                tick[0] += 10 * 10**9
                return {"fixture_seal": True}
            assert operation == "evaluate" and set(p.name for p in inputs.iterdir()) == {
                "seal.json",
                "development.json",
            }
            tick[0] += 10 * 10**9
            return dict(
                selection_split="development",
                final_truth_available=False,
                solved_groups=["group"],
                top_one_groups=[],
            )

    result = run_pilot(
        fixture_manifest(),
        dict(
            examples={"rows": []},
            development={"groups": [{"group_id": "group", "values": ["1"] * 100}]},
        ),
        tmp_path,
        Coordinator(),
        clock=lambda: tick[0],
        utc_clock=lambda: tick[0],
    )
    assert result["selected"] == "encoder" and len(result["arms"]) == 2
    assert result["pairs"] == [[1, 1]] and calls.count("diagnostic") == 1
    assert all([e["minute"] for e in a["evaluations"]] == [15, 30, 45, 55] for a in result["arms"])
    assert result["decisions"][0]["status"] == "baseline_retained"


def test_review_dispositions_bind_current_package_implementation():
    from pathlib import Path
    from oeis_learn.evaluation.experiment_runner import implementation_identity

    report = json.loads(Path("specs/007-experiment-foundation/validation/phase-10-dispositions.json").read_text())
    assert report["original_inputs"] == len(report["items"]) == 68
    assert report["implementation_sha256"] == implementation_identity()


@pytest.mark.parametrize(
    "question,arm,binding",
    [("encoder", "baseline", False), ("encoder", "challenger", False),
     ("position", "challenger", False), ("position", "challenger", True),
     ("capacity", "challenger", False)],
)
def test_closed_pilot_variants_execute_finite_forward_backward(question, arm, binding, record_property):
    profile = pilot_profile(question, arm, binding_scope_cause=binding)
    encoder, decoder = construct_pilot_models(profile, "cpu")
    memory = encoder.forward_from_sequences([[7] * 20], device="cpu")
    tokens = torch.tensor([[BOS_ID] + encode_body("i256.const 7")])
    logits = decoder(tokens, memory)
    loss = logits.square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    assert all(torch.isfinite(p.grad).all() for p in parameters if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.parameters())
    count = sum(p.numel() for p in parameters)
    record_property("parameter_count", count)
    if question == "capacity":
        assert profile["d_ff"] == 2048 and 25_000_000 <= count <= 26_000_000


def test_acceptance_workspaces_do_not_resume_or_delete_previous_runs(tmp_path):
    from oeis_learn.evaluation.cli_acceptance import learning_workspace

    phase = tmp_path / "learning"
    first = learning_workspace(phase)
    prior = first / "training/qualified-00000001.json"
    prior.parent.mkdir()
    prior.write_text("earlier accepted checkpoint")
    second = learning_workspace(phase)
    assert first != second and first.parent == second.parent == phase
    assert prior.read_text() == "earlier accepted checkpoint"
    assert not (second / "training").exists()
