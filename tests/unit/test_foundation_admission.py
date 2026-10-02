"""Admission rejects unsupported provenance before consulting private membership."""

import copy
import pytest


@pytest.fixture(scope="module")
def executed_sample():
    from oeis_learn.data.generic_programs import default_config, GenericSampler
    from oeis_learn.sandbox.pipeline import Runtime

    sampler = GenericSampler(default_config())
    runtime = Runtime()
    for i in range(128):
        sample = sampler.sample(i)
        if sample["sampling_failure"] is None:
            evidence = runtime.evaluate(sample["canonical_source"], list(range(100)))
            if evidence.outcome is None and len(evidence.outputs) == 100:
                return sampler, sample, evidence
    pytest.fail("fixed generic fixture did not produce any executable sample")


@pytest.mark.parametrize("origin", ["oeis", "loda", "named_teacher", "legacy_replay", "pretrained"])
def test_imported_and_assisted_samples_never_reach_execution(origin):
    from oeis_learn.data.generic_programs import default_config, GenericSampler
    from oeis_learn.data.program_admission import validate_sample

    sampler = GenericSampler(default_config())
    sample = sampler.sample(0)
    sample["origin"] = origin
    assert validate_sample(sample, sampler) == "prohibited_origin"


def test_provenance_and_lossless_tokens_cannot_be_forged():
    from oeis_learn.data.generic_programs import default_config, GenericSampler
    from oeis_learn.data.program_admission import validate_sample

    sampler = GenericSampler(default_config())
    sample = next(
        sampler.sample(i) for i in range(100) if sampler.sample(i)["sampling_failure"] is None
    )
    assert validate_sample(sample, sampler) is None
    for change in (
        lambda s: s["body_tokens"].pop(),
        lambda s: s["body_tokens"].append(999999),
        lambda s: s.__setitem__("canonical_source", "i256.zero"),
        lambda s: s.__setitem__("sample_counter", s["sample_counter"] + 1),
        lambda s: s.__setitem__("generator_revision", "sha256:" + "0" * 64),
    ):
        altered = copy.deepcopy(sample)
        change(altered)
        assert validate_sample(altered, sampler) is not None


def test_reserved_collision_returns_no_matched_identity_or_continuation(tmp_path, executed_sample):
    from oeis_learn.data.program_admission import AdmissionService
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    sampler, sample, evidence = executed_sample
    reserved = [digest(evidence.outputs[:20])]
    service = AdmissionService(tmp_path, sampler, reserved, digest(reserved))
    result = service.consider(sample, evidence)
    assert result.record is None
    assert result.decision["reason"] == "reserved_prefix_collision"
    assert result.decision["reserved_prefix_check"] == "collision"
    assert (
        not {"target_id", "values", "outputs", "matched_id", "continuation"}
        & result.decision.keys()
    )


@pytest.mark.parametrize("field", ["outputs", "reference_outputs"])
def test_disagreement_blocks_admission(tmp_path, executed_sample, field):
    from oeis_learn.data.program_admission import AdmissionService, AdmissionGateError
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    sampler, sample, original = executed_sample
    evidence = copy.deepcopy(original)
    getattr(evidence, field)[99] = str(int(getattr(evidence, field)[99]) + 1)
    service = AdmissionService(tmp_path, sampler, [], digest([]))
    with pytest.raises(AdmissionGateError, match="disagreement"):
        service.consider(sample, evidence)
    assert not (tmp_path / "programs").exists()


@pytest.mark.parametrize(
    "mutation",
    ["missing_conditioning", "wrong_conditioning", "cycle", "stale_id", "token", "self_admission"],
)
def test_corrupt_records_cannot_be_loaded_even_with_new_file_hash(
    tmp_path, executed_sample, mutation
):
    from oeis_learn.data.program_admission import AdmissionService, validate_program
    from oeis_learn.experiments.artifacts import (
        canonical_bytes,
        compute_file_hash,
        compute_canonical_digest as digest,
    )

    sampler, sample, evidence = executed_sample
    membership = digest([])
    admitted = AdmissionService(tmp_path, sampler, [], membership).consider(sample, evidence)
    assert admitted.record is not None
    record = copy.deepcopy(admitted.record)
    if mutation == "missing_conditioning":
        record.pop("visible_terms")
    elif mutation == "wrong_conditioning":
        record["visible_terms"][0] = str(int(record["visible_terms"][0]) + 1)
    elif mutation == "cycle":
        record["outputs_ref"] = admitted.record_ref
    elif mutation == "stale_id":
        record["program_id"] = "sha256:" + "0" * 64
    elif mutation == "token":
        record["body_tokens"][-1] = 999999
    else:
        record["admission_decision_ref"] = admitted.decision_ref
    path = tmp_path / admitted.record_ref["path"]
    path.write_bytes(canonical_bytes(record))
    new_ref = {"path": admitted.record_ref["path"], "sha256": compute_file_hash(path)}
    with pytest.raises(ValueError):
        validate_program(tmp_path, new_ref, admitted.decision_ref, sampler, membership)


def test_missing_conditioning_outputs_are_not_fabricated(tmp_path, executed_sample):
    from oeis_learn.data.program_admission import AdmissionService
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    sampler, sample, original = executed_sample
    evidence = copy.deepcopy(original)
    evidence.outputs = evidence.outputs[:19]
    evidence.reference_outputs = evidence.reference_outputs[:19]
    result = AdmissionService(tmp_path, sampler, [], digest([])).consider(sample, evidence)
    assert result.record is None and result.decision["reason"] == "incomplete_output"


def test_asymmetric_resource_limits_are_rejections_not_arithmetic_disagreements(
    tmp_path, executed_sample
):
    from oeis_learn.data.program_admission import AdmissionService
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    sampler, sample, original = executed_sample
    evidence = copy.deepcopy(original)
    evidence.outputs = evidence.outputs[:23]
    evidence.reference_outputs = evidence.reference_outputs[:24]
    evidence.outcome, evidence.reason = "execution_limit", "wasmtime_fuel"
    result = AdmissionService(tmp_path, sampler, [], digest([])).consider(sample, evidence)
    assert result.record is None and result.decision["reason"] == "execution_limit"


def test_duplicate_source_is_rejected_without_second_execution(tmp_path, executed_sample):
    from oeis_learn.data.program_admission import AdmissionService
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    sampler, sample, _ = executed_sample
    result = AdmissionService(tmp_path, sampler, [], digest([])).consider(
        sample, None, duplicate=True
    )
    assert result.record is None and result.decision["reason"] == "duplicate_source"


def test_membership_includes_nonrepresentatives_without_reading_private_truth(tmp_path):
    import json
    import yaml
    from oeis_learn.evaluation.foundation_cohort import freeze_cohort
    from oeis_learn.data.program_admission import load_membership
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    source = tmp_path / "source"
    source.mkdir()
    rows = [
        {
            "record_id": "fixture:" + name,
            "first_index": 0,
            "indices": list(range(100)),
            "values": [str(i + shift) for i in range(100)],
            "metadata": {},
        }
        for name, shift in (("a", 0), ("b", 10))
    ]
    (source / "records.json").write_text(json.dumps(rows))
    config = tmp_path / "cohort.yaml"
    config.write_text(
        yaml.safe_dump(
            dict(
                schema_version="foundation/v1",
                profile="prefix20_total100_v1",
                seed=1,
                dev_count=1,
                final_count=0,
            )
        )
    )
    out = tmp_path / "cohort"
    freeze_cohort(source, config, out)
    (out / "private").rename(out / "private-not-mounted")
    reserved, identity = load_membership(out)
    assert reserved == {digest(row["values"][:20]) for row in rows}
    assert identity.startswith("sha256:")
