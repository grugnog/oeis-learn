"""Immutable final decision lock; seals are separate, committed before scoring."""

from __future__ import annotations

from pathlib import Path
import datetime
from oeis_learn.experiments.artifacts import (
    compute_canonical_digest as digest,
    compute_file_hash,
    load_json,
)
from oeis_learn.evaluation.foundation_cohort import exact_keys, load_cohort, write_json


def create_finalization(
    run_dir, checkpoint_path, protocol_path, cohort_root, stopping_record, output
):
    from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint
    from oeis_learn.evaluation.foundation_synthesis import validate_protocol

    run_dir = Path(run_dir).resolve()
    checkpoint = load_foundation_checkpoint(checkpoint_path)
    cohort = load_cohort(cohort_root)
    protocol = validate_protocol(load_json(Path(protocol_path).read_bytes()))
    stop = load_json(Path(stopping_record).read_bytes())
    exact_keys(
        stop,
        ("decision", "reason", "checkpoint_sha256", "protocol_sha256", "cohort_id"),
        "stopping record",
    )
    if (
        stop["decision"] != "stop"
        or not isinstance(stop["reason"], str)
        or not stop["reason"].strip()
        or stop["checkpoint_sha256"] != checkpoint.manifest["blob_sha256"]
        or stop["protocol_sha256"] != digest(protocol)
        or stop["cohort_id"] != cohort["cohort_id"]
    ):
        raise ValueError("stopping decision does not bind this checkpoint/protocol/cohort")
    ledger_path = run_dir / "ledger.json"
    ledger = load_json(ledger_path.read_bytes())
    exact_keys(
        ledger, ("run_id", "sequence", "charged_budget_ns", "status"), "ledger high-water record"
    )
    if (
        ledger["run_id"] != checkpoint.manifest["run_id"]
        or ledger["status"] not in ("paused", "completed")
        or type(ledger["sequence"]) is not int
        or ledger["sequence"] < checkpoint.manifest["ledger_sequence"]
        or type(ledger["charged_budget_ns"]) is not int
        or ledger["charged_budget_ns"] < checkpoint.manifest["charged_budget_ns"]
    ):
        raise ValueError("invalid or stale run ledger high-water mark")
    core = {
        "schema_version": "foundation/v1",
        "kind": "finalization",
        "run_dir": str(run_dir),
        "run_id": ledger["run_id"],
        "contract_sha256": checkpoint.manifest["contract_sha256"],
        "checkpoint_path": str(Path(checkpoint_path).resolve()),
        "checkpoint_manifest_sha256": checkpoint.manifest_sha256,
        "checkpoint_sha256": checkpoint.manifest["blob_sha256"],
        "cohort_path": str(Path(cohort_root).resolve()),
        "cohort_id": cohort["cohort_id"],
        "protocol_path": str(Path(protocol_path).resolve()),
        "protocol_sha256": digest(protocol),
        "selector": protocol["selector"],
        "stopping_record_path": str(Path(stopping_record).resolve()),
        "stopping_record_sha256": compute_file_hash(Path(stopping_record)),
        "ledger_sha256": compute_file_hash(ledger_path),
        "ledger_sequence": ledger["sequence"],
        "implementation_sha256": implementation_identity(),
    }
    anchor = run_dir / "finalization.json"
    if anchor.exists():
        record = load_json(anchor.read_bytes())
        if {k: record.get(k) for k in core} != core or record.get("finalization_id") != digest(
            record, "finalization_id"
        ):
            raise ValueError(
                "finalization decision is immutable; changed inputs are exploratory only"
            )
    else:
        record = dict(core, created_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        record["finalization_id"] = digest(record)
        write_json(run_dir, "finalization.json", record)
    output = Path(output)
    write_json(output.parent, output.name, record)
    return record


def final_output(lock):
    return Path(lock["run_dir"]) / "final-evaluation" / lock["finalization_id"][7:]


def exposure_path(lock, group_id):
    return (
        Path(lock["run_dir"])
        / "final-exposure"
        / lock["finalization_id"][7:]
        / (group_id[7:] + ".json")
    )


def validate_finalization(path, checkpoint_path, cohort_root, protocol_path):
    record = load_json(Path(path).read_bytes())
    if (
        record.get("schema_version") != "foundation/v1"
        or record.get("kind") != "finalization"
        or record.get("finalization_id") != digest(record, "finalization_id")
    ):
        raise ValueError("invalid finalization lock")
    anchor = load_json((Path(record["run_dir"]) / "finalization.json").read_bytes())
    if anchor != record:
        raise ValueError("uncommitted/replaced finalization lock")
    for field, supplied in [
        ("checkpoint_path", checkpoint_path),
        ("cohort_path", cohort_root),
        ("protocol_path", protocol_path),
    ]:
        if str(Path(supplied).resolve()) != record[field]:
            raise ValueError("final inputs may not be overridden")
    if record.get("implementation_sha256") != implementation_identity():
        raise ValueError("final implementation changed")
    if compute_file_hash(Path(checkpoint_path)) != record["checkpoint_manifest_sha256"]:
        raise ValueError("final checkpoint changed")
    if (
        load_cohort(cohort_root)["cohort_id"] != record["cohort_id"]
        or digest(load_json(Path(protocol_path).read_bytes())) != record["protocol_sha256"]
    ):
        raise ValueError("final cohort/protocol changed")
    if (
        compute_file_hash(Path(record["stopping_record_path"])) != record["stopping_record_sha256"]
        or compute_file_hash(Path(record["run_dir"]) / "ledger.json") != record["ledger_sha256"]
    ):
        raise ValueError("stopping decision or ledger changed after finalization")
    return record


def implementation_identity():
    from oeis_learn.sandbox.pipeline import runtime_identity

    return digest(
        {
            "execution": runtime_identity(),
            "evaluation": {
                name: compute_file_hash(Path(__file__).parent / name)
                for name in (
                    "foundation_synthesis.py",
                    "foundation_cohort.py",
                    "checkpoint.py",
                    "finalization.py",
                )
            },
        }
    )
