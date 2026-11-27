"""A terminal resource decision, verifiable from the four original public projections."""

import io
import json
import re
import zipfile
from pathlib import Path

from .contracts import seal
from .program_search_rehearsal_report import encode
from .program_search_timing import BLOCKS, verify_timing
from .report import MAX_BYTES

VERSION = "round028-program-search-outcome/v1"
LAYERS = (
    ("timing.json", "round028-panels.zip"),
    ("panels.json", "round028-rehearsal.zip"),
    ("rehearsal.json", "round028.zip"),
    ("audit.json",),
)
MEMBERS = ("outcome.json", *(layer[0] for layer in LAYERS))


def archive_bytes(records):
    """Fixed ZIP metadata and stored JSON make the share portable and byte-reproducible."""
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as archive:
        for name, raw in records.items():
            member = zipfile.ZipInfo(name)
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(member, raw)
    return out.getvalue()


def released_records(timing_path):
    """Read only the named, verified public JSON from the producer's original chain."""
    with Path(timing_path).open("rb") as stream:
        parent = verify_timing(stream)
        records = {}
        layer_source = stream
        for layer in LAYERS:
            with zipfile.ZipFile(layer_source) as archive:
                records[layer[0]] = archive.read(layer[0])
                if len(layer) == 2:
                    layer_source = io.BytesIO(archive.read(layer[1]))
    return parent, records


def outcome(verified, records):
    forecast = verified["forecast"]
    if verified["action"] != "REPORT_PILOT_COST_DEFICIT" or forecast["pilot_remaining_us"] >= 0:
        raise ValueError("a verified registered pilot cost deficit is required")
    timing, panels, rehearsal, audit = (json.loads(records[layer[0]]) for layer in LAYERS)
    if audit["decision"]["action"] != "MATCHED_GENERATOR":
        raise ValueError("historical audit did not select the matched generator")
    blocks = timing["blocks"]
    cells = [cell for block in blocks.values() for cell in block["cells"].values()]
    return seal(
        {
            "version": VERSION,
            "audit": audit["identity"],
            "rehearsal": rehearsal["identity"],
            "panels": panels["identity"],
            "timing": verified["evidence"],
            "timing_candidate": verified["candidate"],
            "timing_registration": verified["registration"],
            "diagnostic_route": audit["decision"],
            "mechanical_status": "COMPLETE",
            "evidence_verdict": "INCOMPLETE",
            "next_action": "RETAIN_BASELINE",
            "reason": verified["action"],
            "remaining_authorized_gpu_us": 0,
            "funding": {
                "basis": "slower_observed_case_per_phase_and_stratum_plus_measured_role_overhead",
                "shift_us": forecast["shift_us"],
                "training_us": forecast["training_us"],
                "descriptive_us": forecast["descriptive_us"],
                "endpoint_us": forecast["endpoint_us"],
                "pilot_us": forecast["pilot_us"],
                "deficit_us": -forecast["pilot_remaining_us"],
                "repeat_if_earned_us": forecast["repeat_us"],
                "pilot_and_repeat_us": forecast["pilot_us"] + forecast["repeat_us"],
                "eight_case_batch_pilot_us": forecast["eight_case_batches"]["pilot_us"],
            },
            "measurement": {
                "basis": "exposed_development_timing",
                "distinct_cases": 8,
                "gpu_roles": len(blocks),
                "portfolios": sum(c["cases"] for c in cells),
                "closed": sum(c["closed"] for c in cells),
                "scorer_dispositions": sum(c["scored"] for c in cells),
                "selections": sum(c["selected"] for c in cells),
                "independently_executed": sum(c["executed"] for c in cells),
                "draws_started": sum(c["started"] for c in cells),
                "draws_reserved": sum(
                    (1 if name.endswith("-one") else 8)
                    * sum(c["cases"] for c in blocks[name]["cells"].values())
                    for name in BLOCKS
                ),
                "gpu_role_wall_us": sum(b["gpu_wall_us"] for b in blocks.values()),
                "gpu_lease_us": sum(b["lease_us"] for b in blocks.values()),
                "cpu_role_wall_us": sum(b["cpu_wall_us"] for b in blocks.values()),
                "gpu_covered_us": sum(b["telemetry"]["covered_us"] for b in blocks.values()),
                "gpu_busy_observed_us": sum(b["telemetry"]["busy_us"] or 0 for b in blocks.values())
                if any(b["telemetry"]["covered_us"] for b in blocks.values())
                else None,
                "fresh_reservations": sum(
                    c["cases"]
                    for role in panels["summary"]["panels"].values()
                    for c in role.values()
                ),
            },
            "comparison": {
                "contrast": "first_divergence_minus_ordinary_at_4M",
                "reference": "retained_D",
                "status": "PILOT_UNMEASURED",
                "new_4m_runs": 0,
                "new_pilot_training_tokens": 0,
                "fresh_scored_cases": 0,
                "primary_gain": None,
                "retention_change": None,
                "adjusted_interval": None,
                "replication": "NOT_EARNED",
                "long_confirmation": "NOT_AUTHORIZED",
                "assessment": "SEALED",
            },
        }
    )


def build_outcome(timing_path):
    parent, records = released_records(timing_path)
    return archive_bytes({"outcome.json": encode(outcome(parent, records)), **records})


def read_canonical(path, members):
    with zipfile.ZipFile(path) as archive:
        if (
            tuple(archive.namelist()) != members
            or sum(i.file_size for i in archive.infolist()) > MAX_BYTES
        ):
            raise ValueError("unexpected terminal share membership or size")
        if any(i.compress_type != zipfile.ZIP_STORED for i in archive.infolist()):
            raise ValueError("terminal share requires canonical stored projections")
        records = {name: archive.read(name) for name in members}
    # Reject even unreferenced bytes, ZIP comments, private metadata and alternate encodings.
    canonical = archive_bytes(records)
    if isinstance(path, (str, Path)):
        with Path(path).open("rb") as stream:
            identical = stream.read(len(canonical) + 1) == canonical
    else:
        path.seek(0)
        identical = path.read(len(canonical) + 1) == canonical
    if not identical:
        raise ValueError("terminal share encoding differs")
    return records


def verify_outcome(path):
    # Rebuild each parent in the existing archive-only reader. Every old seal, cross-parent
    # reference, forecast, calibration count and historical routing bound is checked there.
    records = read_canonical(path, MEMBERS)
    parent_bytes = None
    for layer in reversed(LAYERS):
        members = {layer[0]: records[layer[0]]}
        if len(layer) == 2:
            members[layer[1]] = parent_bytes
        parent_bytes = archive_bytes(members)
    checked = verify_timing(io.BytesIO(parent_bytes))
    decision = outcome(checked, records)
    if records["outcome.json"] != encode(decision):
        raise ValueError("terminal decision differs from verified evidence")
    return {
        "status": "VERIFIED",
        "scope": "archive_only_terminal",
        "report": decision["identity"],
        **{
            key: decision[key]
            for key in (
                "evidence_verdict",
                "reason",
                "next_action",
                "remaining_authorized_gpu_us",
                "audit",
                "timing",
                "funding",
                "measurement",
                "comparison",
            )
        },
    }


def workstation_receipt(public, replay_id, boundary):
    """The private reader issues this receipt only after installing and committing the share."""
    for identity in (replay_id, boundary):
        if not isinstance(identity, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", identity):
            raise ValueError("invalid workstation receipt reference")
    return seal(
        {
            "version": "round028-terminal-workstation/v1",
            "scope": "original_weights_choices_and_scorer_clocks",
            "status": "VERIFIED",
            "report": public["report"],
            "audit": public["audit"],
            "timing": public["timing"],
            "portfolios": public["measurement"]["portfolios"],
            "private_replay": replay_id,
            "journal_boundary": boundary,
            "new_gpu_dispatches": 0,
        }
    )


def verify_workstation(path):
    """Recompute the public decision and check what the private reader recorded at its boundary."""
    records = read_canonical(path, ("workstation.json", "round028.zip"))
    public = verify_outcome(io.BytesIO(records["round028.zip"]))
    value = json.loads(records["workstation.json"])
    if not isinstance(value, dict):
        raise ValueError("workstation receipt must be an object")
    expected = workstation_receipt(
        public, value.get("private_replay"), value.get("journal_boundary")
    )
    if encode(expected) != records["workstation.json"]:
        raise ValueError("workstation receipt differs from the verified public decision")
    return dict(public, scope="archive_only_with_workstation_receipt", workstation=expected)
