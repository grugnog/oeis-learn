"""Actual-weight, information-boundary and sealed scoring regressions."""

import pytest
import torch


def test_strict_loader_rejects_missing_checkpoint(tmp_path):
    from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint

    with pytest.raises((ValueError, FileNotFoundError)):
        load_foundation_checkpoint(tmp_path / "missing.json")


def test_seeds_ignore_hidden_identity_and_metadata():
    from oeis_learn.evaluation.foundation_synthesis import attempt_seed, default_protocol

    protocol = default_protocol()
    terms = ["0"] * 20
    seed = attempt_seed(protocol, terms, 1)
    other = dict(protocol, checkpoint_sha256="private-A", cohort_sha256="private-B")
    assert attempt_seed(other, terms, 1) == seed
    assert attempt_seed(protocol, terms, 2) != seed


def test_actual_checkpoint_parameters_drive_generation_and_scores(tmp_path):
    from tests.helpers.foundation_fixture import make_checkpoint, make_cohort, protocol_file
    from oeis_learn.evaluation.foundation_synthesis import evaluate_foundation

    checkpoint = make_checkpoint(tmp_path / "zero")
    different = make_checkpoint(tmp_path / "one", "i256.const 1")
    cohort = make_cohort(tmp_path / "dataset")
    protocol = protocol_file(tmp_path)
    good = evaluate_foundation(checkpoint, cohort, protocol, tmp_path / "good")
    bad = evaluate_foundation(different, cohort, protocol, tmp_path / "bad")
    assert good["N"] == bad["N"] == 1
    assert good["success_any"] == good["success_top1"] == 1
    assert bad["success_any"] == bad["success_top1"] == 0
    assert good["checkpoint_sha256"] != bad["checkpoint_sha256"]
    assert not good["qualified"] and good["proof_status"] == "not_claimed"


def test_hidden_metadata_and_nonce_never_change_proposals(tmp_path):
    from tests.helpers.foundation_fixture import make_checkpoint, make_cohort, protocol_file, record
    from oeis_learn.evaluation.foundation_synthesis import evaluate_foundation
    from oeis_learn.experiments.artifacts import load_json

    checkpoint = make_checkpoint(tmp_path / "model")
    protocol = protocol_file(tmp_path)
    a = make_cohort(tmp_path / "a", [record("first", [0] * 100, 7, {"family": "secret-A"})])
    b = make_cohort(
        tmp_path / "b", [record("other", [0] * 20 + [9] * 80, -8, {"family": "secret-B"})]
    )
    first = evaluate_foundation(checkpoint, a, protocol, tmp_path / "out-a")
    second = evaluate_foundation(checkpoint, b, protocol, tmp_path / "out-b")

    def proposals(root):
        seal = load_json(next(root.glob("targets/*/seal.json")).read_bytes())
        return [(a["seed"], a["source"], a["tokens"]) for a in seal["attempts"]]

    assert proposals(tmp_path / "out-a") == proposals(tmp_path / "out-b")
    assert first["success_any"] == 1 and second["success_any"] == 0


def test_fixed_denominator_duplicates_and_no_reference_substitution(tmp_path):
    from tests.helpers.foundation_fixture import make_checkpoint, make_cohort, protocol_file, record
    from oeis_learn.evaluation.foundation_synthesis import evaluate_foundation
    from oeis_learn.experiments.artifacts import load_json

    checkpoint = make_checkpoint(tmp_path / "model")
    cohort = make_cohort(
        tmp_path / "data", [record(str(i), [i] * 100) for i in range(3)], dev_count=3
    )
    result = evaluate_foundation(
        checkpoint, cohort, protocol_file(tmp_path, attempts=3), tmp_path / "out"
    )
    assert result["N"] == 3 and result["success_any"] == result["success_top1"] == 1
    assert result["rates"]["success_any"] == 1 / 3
    assert sum(result["prefix_outcomes"].values()) == 9
    for path in (tmp_path / "out").glob("targets/*/seal.json"):
        seal = load_json(path.read_bytes())
        assert len(seal["attempts"]) == 3  # duplicate source still consumes attempts
        assert len({a["source"] for a in seal["attempts"]}) == 1
        if seal["selected"] is not None:
            assert seal["selected"] == 0


def test_token_cap_is_a_recorded_failure_not_a_repaired_candidate(tmp_path):
    from tests.helpers.foundation_fixture import make_checkpoint, make_cohort, protocol_file
    from oeis_learn.evaluation.foundation_synthesis import evaluate_foundation

    checkpoint = make_checkpoint(tmp_path / "model", "i256.const 1")
    result = evaluate_foundation(
        checkpoint,
        make_cohort(tmp_path / "data"),
        protocol_file(tmp_path, max_body_tokens=1, attempts=1),
        tmp_path / "out",
    )
    assert result["N"] == 1 and result["success_any"] == 0
    assert result["prefix_outcomes"] == {"execution_limit": 1}


def test_checkpoint_rejects_weights_only_corrupt_and_mismatched_state(tmp_path):
    import torch
    from tests.helpers.foundation_fixture import make_checkpoint
    from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint
    from oeis_learn.experiments.artifacts import load_json, canonical_bytes, compute_file_hash

    path = make_checkpoint(tmp_path / "model")
    original = path.read_bytes()
    manifest = load_json(original)
    blob = path.parent / "model.pt"
    payload = torch.load(blob, weights_only=True)
    for mutate in [
        lambda p: p.pop("optimizer"),
        lambda p: p["counters"].__setitem__("completed_update", 2),
        lambda p: p["model"]["decoder"].pop("lm_head.weight"),
        lambda p: p["data_order"].__setitem__("accumulation_step", 1),
    ]:
        import copy

        bad = copy.deepcopy(payload)
        mutate(bad)
        torch.save(bad, blob)
        manifest["blob_sha256"] = compute_file_hash(blob)
        path.write_bytes(canonical_bytes(manifest))
        with pytest.raises((ValueError, RuntimeError)):
            load_foundation_checkpoint(path)
    blob.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="digest"):
        load_foundation_checkpoint(path)


@pytest.mark.parametrize(
    "bodies", [("nop i256.zero", "i256.zero"), ("nop i256.zero", "i256.zero nop")]
)
def test_selector_uses_length_then_hash_and_seal_rejects_changed_choice(
    tmp_path, monkeypatch, bodies
):
    from tests.helpers.foundation_fixture import make_checkpoint, make_cohort, protocol_file
    from oeis_learn.evaluation.foundation_synthesis import evaluate_foundation, ModelProcess
    from oeis_learn.decoder.program_codec import encode_body
    from oeis_learn.experiments.artifacts import (
        load_json,
        canonical_bytes,
        compute_canonical_digest as digest,
    )

    checkpoint = make_checkpoint(tmp_path / "model")
    cohort = make_cohort(tmp_path / "data")
    protocol = protocol_file(tmp_path)
    sources = iter(bodies)

    def generate(*args, **kwargs):
        source = next(sources)
        return {"source": source, "tokens": encode_body(source), "outcome": None, "reason": None}

    monkeypatch.setattr(ModelProcess, "generate", generate)
    out = tmp_path / "out"
    evaluate_foundation(checkpoint, cohort, protocol, out)
    path = next(out.glob("targets/*/seal.json"))
    seal = load_json(path.read_bytes())
    import hashlib

    expected = min(
        range(2),
        key=lambda i: (len(encode_body(bodies[i])), hashlib.sha256(bodies[i].encode()).hexdigest()),
    )
    assert seal["selected"] == expected
    seal["selected"] = 1 - expected
    seal["seal_id"] = digest(seal, "seal_id")
    path.write_bytes(canonical_bytes(seal))
    with pytest.raises(ValueError):
        evaluate_foundation(checkpoint, cohort, protocol, out)


def test_missing_truth_invalidates_before_generation(tmp_path, monkeypatch):
    from tests.helpers.foundation_fixture import make_checkpoint, make_cohort, protocol_file
    from oeis_learn.evaluation.foundation_synthesis import evaluate_foundation, ModelProcess
    from oeis_learn.evaluation.foundation_cohort import load_cohort

    checkpoint = make_checkpoint(tmp_path / "model")
    cohort = make_cohort(tmp_path / "data")
    group = load_cohort(cohort)["groups"][0]
    (cohort / group["truth"]["path"]).unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("generation without complete truth")

    monkeypatch.setattr(ModelProcess, "generate", forbidden)
    with pytest.raises(FileNotFoundError):
        evaluate_foundation(checkpoint, cohort, protocol_file(tmp_path), tmp_path / "out")
    assert not (tmp_path / "out/reports/evaluation.json").exists()


@pytest.mark.skipif(
    torch.version.hip is None or not torch.cuda.is_available(),
    reason="requires AMD HIP GPU via lab-gpu; CPU fixture is not device evidence",
)
def test_gpu_checkpoint_loading_and_prefix_only_generation(tmp_path):
    import torch
    import uuid
    from tests.helpers.foundation_fixture import make_checkpoint
    from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint
    from oeis_learn.evaluation.foundation_synthesis import ModelProcess
    from oeis_learn.experiments.profiles import profile_digests

    checkpoint = make_checkpoint(tmp_path / "gpu-fixture")
    loaded = load_foundation_checkpoint(checkpoint, device="cuda")
    assert next(loaded.encoder.parameters()).is_cuda and next(loaded.decoder.parameters()).is_cuda
    assert all(p.dtype == torch.float32 for p in loaded.decoder.parameters())
    prompt = {
        "kind": "visible_prompt",
        "schema_version": "foundation/v1",
        "request_nonce": str(uuid.uuid4()),
        "language_profile": profile_digests()["language"],
        "observed_terms": [str(-(2**255)), str(2**255 - 1)] * 10,
    }
    import time

    with ModelProcess(checkpoint, device="cuda") as model:
        proposal = model.generate(
            prompt,
            seed=11,
            temperature=0,
            max_body_tokens=32,
            deadline_ns=time.monotonic_ns() + 20_000_000_000,
        )
    assert proposal["source"] == "i256.zero" and proposal["outcome"] is None


def test_training_checkpoint_requires_resolved_provenance(tmp_path):
    from tests.helpers.foundation_fixture import make_checkpoint
    from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint
    from oeis_learn.experiments.artifacts import (
        load_json,
        canonical_bytes,
        compute_canonical_digest,
    )

    path = make_checkpoint(tmp_path)
    contract_path = path.parent / "contract.json"
    contract = load_json(contract_path.read_bytes())
    contract["purpose"] = "training"
    contract["contract_id"] = compute_canonical_digest(contract, "contract_id")
    contract_path.write_bytes(canonical_bytes(contract))
    manifest = load_json(path.read_bytes())
    manifest["contract_sha256"] = compute_canonical_digest(contract)
    path.write_bytes(canonical_bytes(manifest))
    with pytest.raises(ValueError, match="base_image_digest: digest must not be null"):
        load_foundation_checkpoint(path)


def test_ignored_weights_are_rejected(tmp_path, monkeypatch):
    from tests.helpers.foundation_fixture import make_checkpoint
    from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint
    from oeis_learn.decoder.wat_decoder import WatTransformerDecoder

    checkpoint = make_checkpoint(tmp_path)
    monkeypatch.setattr(WatTransformerDecoder, "load_state_dict", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="silently changed or ignored"):
        load_foundation_checkpoint(checkpoint)
