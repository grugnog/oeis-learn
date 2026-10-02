"""Evaluator-owned generic admission and immutable acyclic program evidence.

The membership interface returns a collision reason, never a target identity or
continuation. The trainer receives only separately exported approved examples.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
import uuid

from oeis_learn.data.generic_programs import GenericSampler
from oeis_learn.decoder.program_codec import decode_body, encode_body, CodecError
from oeis_learn.evaluation.foundation_cohort import exact_keys, read_ref, write_json
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest, load_json
from oeis_learn.experiments.models import check_digest, parse_integer_text
from oeis_learn.experiments.profiles import I256_MIN, I256_MAX, profile_digests
from oeis_learn.sandbox.pipeline import ExecutionEvidence, digest_bytes, verify

CORE_KEYS = (
    "generator_revision",
    "generator_config_hash",
    "sample_seed",
    "sample_counter",
    "language_profile",
    "codec_profile",
    "source_sha256",
    "tokens_sha256",
    "origin",
)
RECORD_KEYS = (
    *CORE_KEYS,
    "program_id",
    "canonical_source",
    "body_tokens",
    "visible_terms",
    "outputs_ref",
    "production_evidence_ref",
    "reference_evidence_ref",
)
SAMPLE_KEYS = (
    "generator_revision",
    "generator_config_hash",
    "sample_seed",
    "sample_counter",
    "origin",
    "language_profile",
    "codec_profile",
    "canonical_source",
    "body_tokens",
    "statistics",
    "sampling_failure",
)


class AdmissionGateError(ValueError):
    """Independent disagreement/corrupt evidence blocks pool publication."""


def validate_sample(sample, sampler):
    """Reject claimed provenance unless the exact generic draw is reproducible."""
    if not isinstance(sample, dict) or sample.get("origin") != "generic_sample":
        return "prohibited_origin"
    try:
        exact_keys(sample, SAMPLE_KEYS, "generic sample")
        if digest(sample) != digest(sampler.sample(sample["sample_counter"])):
            return "provenance_mismatch"
        if sample["sampling_failure"]:
            return sample["sampling_failure"]
        source, tokens = sample["canonical_source"], sample["body_tokens"]
        if (
            not isinstance(tokens, list)
            or any(type(t) is not int for t in tokens)
            or decode_body(tokens) != source
            or encode_body(source) != tokens
        ):
            return "invalid_roundtrip"
    except (ValueError, TypeError, KeyError, CodecError):
        return "invalid_sample"
    return None


def program_core(sample):
    source = sample["canonical_source"]
    return {
        **{key: sample[key] for key in CORE_KEYS if key not in ("source_sha256", "tokens_sha256")},
        "source_sha256": digest_bytes(source.encode()),
        "tokens_sha256": digest(sample["body_tokens"]),
    }


def load_membership(cohort_root):
    """Read only the evaluator's manifest and exact membership set, not truth."""
    root = Path(cohort_root)
    manifest = load_json((root / "manifest.json").read_bytes())
    if (
        manifest.get("schema_version") != "foundation/v1"
        or manifest.get("kind") != "cohort_manifest"
        or manifest.get("profile") != "prefix20_total100_v1"
        or manifest.get("cohort_id") != digest(manifest, "cohort_id")
    ):
        raise ValueError("missing/corrupt frozen cohort identity")
    values = read_ref(root, manifest["prefix_membership"])
    if (
        not isinstance(values, list)
        or any(not isinstance(v, str) for v in values)
        or values != sorted(set(values))
    ):
        raise ValueError("invalid reserved-prefix membership set")
    for value in values:
        check_digest(value, "reserved prefix")
    return frozenset(values), manifest["prefix_membership"]["sha256"]


def _verified_outputs(sample, evidence):
    if evidence.source_sha256 != digest_bytes(sample["canonical_source"].encode()):
        raise AdmissionGateError("execution_source_mismatch")
    if any(a != b for a, b in zip(evidence.outputs, evidence.reference_outputs)):
        raise AdmissionGateError("production_reference_disagreement")
    if evidence.reason and "disagreement" in evidence.reason:
        raise AdmissionGateError("production_reference_disagreement")
    if evidence.outcome is not None:
        return evidence.outcome
    if len(evidence.outputs) != 100 or len(evidence.reference_outputs) != 100:
        return "incomplete_output"
    # Independent outputs are the synthetic truth, not an OEIS continuation.
    try:
        result = verify(
            evidence,
            evidence.reference_outputs,
            "full",
            run_id=str(uuid.UUID(int=0)),
            attempt_index=sample["sample_counter"],
            prompt_sha256=digest(evidence.outputs[:20]),
            purpose="conformance",
        )
    except (ValueError, TypeError) as exc:
        raise AdmissionGateError("invalid_independent_evidence") from exc
    if result["outcome"] != "full_horizon_match":
        return result["outcome"]
    return None


def independent_record(program_id, evidence):
    return {
        "program_id": program_id,
        "source_sha256": evidence.source_sha256,
        "outputs": evidence.reference_outputs,
        "terms": evidence.reference_terms,
        "steps": evidence.reference_steps,
        "evidence_sha256": evidence.reference_sha256,
    }


@dataclass(frozen=True)
class AdmissionResult:
    decision: dict
    decision_ref: dict
    record: dict | None
    record_ref: dict | None


class AdmissionService:
    def __init__(self, root, sampler: GenericSampler, reserved_prefixes, membership_sha256):
        self.root, self.sampler = Path(root), sampler
        self._reserved = frozenset(reserved_prefixes)
        check_digest(membership_sha256, "membership identity")
        for value in self._reserved:
            check_digest(value, "reserved prefix")
        self.membership_sha256 = membership_sha256

    def consider(self, sample, evidence: ExecutionEvidence | None, *, duplicate=False):
        reason = validate_sample(sample, self.sampler)
        if reason is None and duplicate:
            reason = "duplicate_source"
        counter = sample.get("sample_counter") if isinstance(sample, dict) else None
        core = program_core(sample) if reason is None else None
        identity = digest(core) if core else None
        record, record_ref, evidence_ref = None, None, None
        membership = "not_checked"
        if reason is None:
            if evidence is None:
                reason = "missing_execution"
            else:
                evidence_ref = write_json(
                    self.root,
                    f"evidence/{identity[7:]}/production.json",
                    {"program_id": identity, "execution": evidence.to_dict()},
                )
                reason = _verified_outputs(sample, evidence)
        if reason is None:
            membership = "collision" if digest(evidence.outputs[:20]) in self._reserved else "clear"
            if membership == "collision":
                reason = "reserved_prefix_collision"
        if reason is None:
            outputs = write_json(
                self.root,
                f"evidence/{identity[7:]}/outputs.json",
                {"program_id": identity, "values": evidence.outputs},
            )
            reference = write_json(
                self.root,
                f"evidence/{identity[7:]}/reference.json",
                independent_record(identity, evidence),
            )
            record = {
                **core,
                "program_id": identity,
                "canonical_source": sample["canonical_source"],
                "body_tokens": sample["body_tokens"],
                "visible_terms": evidence.outputs[:20],
                "outputs_ref": outputs,
                "production_evidence_ref": evidence_ref,
                "reference_evidence_ref": reference,
            }
            record_ref = write_json(self.root, f"programs/{identity[7:]}.json", record)
        decision = {
            "schema_version": "foundation/v1",
            "kind": "admission_decision",
            "program_id": identity,
            "sample_counter": counter,
            "status": "admitted" if reason is None else "rejected",
            "reason": reason,
            "reserved_prefix_check": membership,
            "membership_sha256": self.membership_sha256,
            "evidence_ref": evidence_ref,
        }
        decision["decision_id"] = digest(decision)
        decision_ref = write_json(
            self.root, f"decisions/{decision['decision_id'][7:]}.json", decision
        )
        return AdmissionResult(decision, decision_ref, record, record_ref)


def validate_program(root, record_ref, decision_ref, sampler, membership_sha256):
    """Controller-side strict loader; private continuations never enter its return value."""
    record = read_ref(root, record_ref)
    exact_keys(record, RECORD_KEYS, "program record")
    core = {key: record[key] for key in CORE_KEYS}
    identity = digest(core)
    if record["program_id"] != identity:
        raise ValueError("stale program identity")
    sample = sampler.sample(record["sample_counter"])
    if (
        validate_sample(sample, sampler) is not None
        or digest(core) != digest(program_core(sample))
        or record["canonical_source"] != sample["canonical_source"]
        or record["body_tokens"] != sample["body_tokens"]
    ):
        raise ValueError("invalid generic provenance or token roundtrip")
    outputs = read_ref(root, record["outputs_ref"])
    exact_keys(outputs, ("program_id", "values"), "synthetic outputs")
    if (
        outputs["program_id"] != identity
        or not isinstance(outputs["values"], list)
        or len(outputs["values"]) != 100
    ):
        raise ValueError("missing exact100 synthetic outputs")
    for value in outputs["values"]:
        parse_integer_text(value, I256_MIN, I256_MAX, "synthetic output")
    if record["visible_terms"] != outputs["values"][:20]:
        raise ValueError("missing/mismatched conditioning prefix")
    production = read_ref(root, record["production_evidence_ref"])
    exact_keys(production, ("program_id", "execution"), "production evidence")
    if production["program_id"] != identity:
        raise ValueError("stale production identity")
    evidence = ExecutionEvidence(**production["execution"])
    if _verified_outputs(sample, evidence) is not None or evidence.outputs != outputs["values"]:
        raise ValueError("unverified synthetic outputs")
    reference = read_ref(root, record["reference_evidence_ref"])
    if reference != independent_record(identity, evidence):
        raise ValueError("stale/mismatched independent evidence")
    decision = read_ref(root, decision_ref)
    exact_keys(
        decision,
        (
            "schema_version",
            "kind",
            "program_id",
            "sample_counter",
            "status",
            "reason",
            "reserved_prefix_check",
            "membership_sha256",
            "evidence_ref",
            "decision_id",
        ),
        "admission decision",
    )
    if (
        decision["schema_version"] != "foundation/v1"
        or decision["kind"] != "admission_decision"
        or decision["decision_id"] != digest(decision, "decision_id")
        or decision["program_id"] != identity
        or decision["sample_counter"] != record["sample_counter"]
        or decision["status"] != "admitted"
        or decision["reason"] is not None
        or decision["reserved_prefix_check"] != "clear"
        or decision["membership_sha256"] != membership_sha256
        or decision["evidence_ref"] != record["production_evidence_ref"]
    ):
        raise ValueError("missing/stale admission decision")
    return {
        "program_id": identity,
        "codec_profile": profile_digests()["codec"],
        "visible_terms": record["visible_terms"],
        "body_tokens": record["body_tokens"],
    }
