"""Portable, diagnostic-only receipts for the donated ordinary next-update comparison."""

import json
from pathlib import Path
import re

from oeis_learn.generalization.budgets import integer
from .contracts import _atomic_write, seal
from .program_search_outcome import archive_bytes, read_canonical
from .program_search_rehearsal_report import encode

QUALIFICATION = "round029-qualification.zip"
REPORT = "round029-report.zip"


def closed(value, keys, *, sealed=False):
    if not isinstance(value, dict) or set(value) != set(keys.split()) | (
        {"identity"} if sealed else set()
    ):
        raise ValueError("unexpected fields in public diagnostic record")
    if sealed and value != seal(value):
        raise ValueError("public diagnostic seal differs")


def hashes(*values):
    if any(not isinstance(v, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", v) for v in values):
        raise ValueError("opaque digest required in public diagnostic record")


def donation_qualification(binding, request, grant):
    return seal(
        dict(
            version="round029-donation-qualification/v1",
            pilot=binding["pilot"],
            candidate=binding["candidate"],
            source=binding["source"],
            grant=grant["identity"],
            total_us=grant["total_us"],
            holdback_us=grant["holdback_us"],
            image=request["image"],
            physical=request["physical"],
            request=request["identity"],
            role_policy=request["role_policy"],
            plan=request["plan"],
        )
    )


def validate_qualification(value):
    from oeis_learn.pipeline_validation.progressive_velocity_donor import PILOT_REPORT
    from oeis_learn.pipeline_validation.progressive_roles import ROUND029_ROLE_POLICY

    closed(
        value,
        "version pilot candidate source grant total_us holdback_us image physical request role_policy plan",
        sealed=True,
    )
    if (
        value["version"] != "round029-donation-qualification/v1"
        or value["pilot"] != PILOT_REPORT
        or value["role_policy"] != ROUND029_ROLE_POLICY
        or not re.fullmatch(r"[0-9a-f]{40}", value["candidate"])
        or integer(value["total_us"], 1) < integer(value["holdback_us"])
    ):
        raise ValueError("original donation qualification and authority required")
    hashes(*(value[k] for k in ("source", "grant", "image", "physical", "request")))
    plan = value["plan"]
    closed(
        plan,
        "parent checkpoint package old_policy new_policy parent_updates prior_useful prior_charge update_tokens rows next_rows",
        sealed=True,
    )
    hashes(
        *(
            plan[k]
            for k in (
                "parent",
                "checkpoint",
                "package",
                "old_policy",
                "new_policy",
                "rows",
                "next_rows",
            )
        )
    )
    if (
        integer(plan["prior_charge"]) < integer(plan["prior_useful"], 4_000_000)
        or not integer(plan["parent_updates"], 1)
        or not integer(plan["update_tokens"], 1)
        or plan["old_policy"] == plan["new_policy"]
    ):
        raise ValueError("entire 4M original cut and a separately identified next update required")
    return value


def donation_report(qualification, record, snapshot):
    consumed = snapshot["consumed"]
    if snapshot["pending"]:
        raise ValueError("unresolved qualification debits")
    return seal(
        dict(
            version="round029-donation-report/v1",
            qualification=qualification["identity"],
            receipt=record["identity"],
            comparison=record["comparison"],
            lease_us=record["lease_us"],
            peak_device_bytes=record["peak_device_bytes"],
            charged_tokens=consumed["tokens"],
            attempts=consumed["attempts"],
            elapsed_us=consumed["elapsed_us"],
            remaining_us=snapshot["remaining"]["elapsed_us"],
            scientific_verdict=None,
        )
    )


def validate_donation(report, qualification):
    from oeis_learn.pipeline_validation.progressive_velocity_continuation import verify_comparison

    q = validate_qualification(qualification)
    closed(
        report,
        "version qualification receipt comparison lease_us peak_device_bytes charged_tokens attempts elapsed_us remaining_us scientific_verdict",
        sealed=True,
    )
    cmp = report["comparison"]
    closed(cmp, "plan parent_updates reference candidate charged_tokens", sealed=True)
    for name in ("reference", "candidate"):
        row = cmp[name]
        closed(row, "parent_weights weights optimizer rng tokens updates useful_tokens next_rows")
        hashes(*(row[k] for k in ("parent_weights", "weights", "optimizer", "rng", "next_rows")))
        for k in ("tokens", "updates", "useful_tokens"):
            integer(row[k], 1)
    hashes(report["receipt"])
    for k in ("parent_updates", "charged_tokens"):
        integer(cmp[k], 1)
    verify_comparison(cmp, q["plan"])
    if (
        report["version"] != "round029-donation-report/v1"
        or report["qualification"] != q["identity"]
        or report["scientific_verdict"] is not None
        or integer(report["attempts"]) != 2
        or integer(report["charged_tokens"]) != cmp["charged_tokens"]
        or integer(report["elapsed_us"]) + integer(report["remaining_us"]) != q["total_us"]
        or report["remaining_us"] < q["holdback_us"]
        or integer(report["lease_us"], 1) > report["elapsed_us"]
        or not integer(report["peak_device_bytes"], 1)
    ):
        raise ValueError("donation parity report conflicts with pre-score inputs or debits")
    return report


def install(path, member, value):
    """Canonical immutable ZIP includes only the validated public projection."""
    path = Path(path)
    raw = encode(value)
    if not path.exists():
        _atomic_write(path, archive_bytes({member: raw}))
    if read_canonical(path, (member,))[member] != raw:
        raise ValueError("previously installed public diagnostic differs")


def read_member(path, member):
    raw = read_canonical(path, (member,))[member]
    record = json.loads(raw)
    if encode(record) != raw:
        raise ValueError("public diagnostic requires canonical bytes")
    return record


def verify_archives(report, qualification):
    q = read_member(qualification, "donation-qualification.json")
    r = read_member(report, "donation-report.json")
    return validate_donation(r, q)
