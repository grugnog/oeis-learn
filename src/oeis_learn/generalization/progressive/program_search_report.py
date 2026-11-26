"""Disclosure-limited historical audit archive; each released sum has a checked denominator."""

import json
import os
from pathlib import Path
import re
import tempfile
import zipfile

from .campaign import diagnostic_registration, learned_search_policy
from .cohorts import STRATA
from .contracts import seal
from .program_search_routing import route

VERSION = "round028-audit-report/v1"


def _methods(policy):
    return [*(f"trial-{t:02d}-{n // 1000000}M" for t, n in policy["historical_looks"]),
            *(label for model in ("retained_D", "random") for label in (
                model, *(model + "/" + m for m in ("template", "polynomial", "linear_recurrence"))))]


def public_audit(registration, private, *, elapsed_us, private_identity):
    checkpoints = [{k: row[k] for k in ("arm", "seed", "target", "disposition", "reason")}
                   for row in private["checkpoints"]]
    phases = ("startup", "training", "inference", "search", "verification")
    old_costs = {name: {k: row["costs"][k] for k in phases} for name, row in private["original_trials"].items()}
    followup = private["followup_training"]
    units = {}
    if followup and "charges" in followup:
        charges = followup["charges"]
        units = {k: sum(task[k] for task in charges.values()) for k in ("tokens", "elapsed_us", "draws", "native_work")}
    value = seal(dict(version=VERSION, candidate=registration["candidate"], registration=registration["identity"],
                      private_cut=private_identity, scope="historical_development", denominator=128,
                      methods=private["methods"], pairs=private["pairs"], checkpoints=checkpoints,
                      projection=private["projection"], decision=private["decision"],
                      audit_elapsed_us=elapsed_us, gpu_dispatches=0, historical_1m_cost_us=old_costs, followup_charge=units,
                      trial05_unfinished_composition=sum(r["stratum"] == "composition"
                          and r["diagnosis"]["unfinished_fit"] for r in private["cases"] if r["stratum"] in
                          {"composition", "unseen_structure"})))
    validate_public(value)
    return value


def validate_public(value):
    fields = {"version", "identity", "registration", "candidate", "private_cut", "scope", "denominator",
              "methods", "pairs", "checkpoints", "projection", "decision", "audit_elapsed_us",
              "gpu_dispatches", "trial05_unfinished_composition", "historical_1m_cost_us", "followup_charge"}
    def require(ok):
        if not ok:
            raise ValueError("inconsistent public audit evidence")
    def count(n, maximum):
        return type(n) is int and 0 <= n <= maximum

    require(set(value) == fields and value["version"] == VERSION and seal(value) == value
            and diagnostic_registration(value["candidate"])["identity"] == value["registration"]
            and re.fullmatch("sha256:[a-f0-9]{64}", value["private_cut"]) and value["scope"] == "historical_development"
            and value["denominator"] == 128 and count(value["audit_elapsed_us"], 2**63 - 1)
            and count(value["gpu_dispatches"], 0) and count(value["trial05_unfinished_composition"], 32))
    require(value["decision"] == route(value["projection"]))
    require(not set(value["historical_1m_cost_us"]) - {f"trial-{t:02d}-1M" for t in range(6)})
    for phases in value["historical_1m_cost_us"].values():
        require(set(phases) == {"startup", "training", "inference", "search", "verification"}
                and all(count(v, 2**63 - 1) for v in phases.values()))
    followup = value["followup_charge"]
    require((not followup or set(followup) == {"tokens", "elapsed_us", "draws", "native_work"})
            and all(count(v, 2**63 - 1) for v in followup.values()))
    methods = _methods(learned_search_policy())
    require(set(value["methods"]) == set(methods))
    for name, strata in value["methods"].items():
        require(set(strata) == set(STRATA))
        rules = {"frozen", "companion"} if name.startswith("trial-") else {"frozen"}
        for cells in strata.values():
            require(set(cells) == rules)
            for frequencies in cells.values():
                require(not (set(frequencies) - {"success", "failure", "unresolved", "unavailable"})
                        and all(count(n, 32) for n in frequencies.values()) and sum(frequencies.values()) == 32)
            if len(rules) == 2:
                require(cells["frozen"].get("success", 0) == cells["companion"].get("success", 0))
    comparisons = {(m, "retained_D") for m in methods if m != "retained_D"} | {("trial-00-2M", "trial-00-1M")}
    require(set(value["pairs"]) == {f"{a} vs {b}" for a, b in comparisons})
    for a, b in comparisons:
        pair = value["pairs"][f"{a} vs {b}"]
        require(set(pair) == set(STRATA))
        for s, cells in pair.items():
            require(set(cells) == {"frozen", "companion"})
            for rule, row in cells.items():
                require(set(row) == {"both", "first_only", "second_only", "neither", "unresolved",
                                     "unavailable", "denominator"} and all(count(n, 32) for n in row.values()))
                counts = [value["methods"][k][s][rule if k.startswith("trial-") else "frozen"] for k in (a, b)]
                require(row["denominator"] == sum(row[k] for k in ("both", "first_only", "second_only", "neither")) == 32
                        and row["both"] + row["first_only"] == counts[0].get("success", 0)
                        and row["both"] + row["second_only"] == counts[1].get("success", 0))
                for overlay in ("unresolved", "unavailable"):
                    x, y = (c.get(overlay, 0) for c in counts)
                    require(max(x, y) <= row[overlay] <= min(32, x + y))
    checkpoints = value["checkpoints"]
    require(len(checkpoints) == 4 and len({(c["arm"], c["seed"]) for c in checkpoints}) == 4)
    seeds = {}
    for c in checkpoints:
        require(set(c) == {"arm", "seed", "target", "disposition", "reason"} and c["arm"] in {"static", "refreshed"}
                and count(c["seed"], 2**63 - 1) and c["target"] == 25000000
                and (c["disposition"], c["reason"]) in {("unavailable", "missing_checkpoint_index"),
                    ("unavailable", "missing_checkpoint_closure"), ("incompatible", "protected_panels_only")})
        seeds.setdefault(c["arm"], set()).add(c["seed"])
    require(seeds.get("static") == seeds.get("refreshed") and len(seeds["static"]) == 2)
    return value


def verify_archive(path):
    from .report import MAX_BYTES
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["audit.json"]:
            raise ValueError("unexpected public archive member")
        member = archive.getinfo("audit.json")
        if member.file_size > MAX_BYTES:
            raise ValueError("public audit exceeds discovery report allowance")
        if archive.comment or member.comment or member.extra:
            raise ValueError("unexpected public archive metadata")
        raw = archive.read(member)
        value = validate_public(json.loads(raw))
        if raw != json.dumps(value, sort_keys=True, allow_nan=False).encode():
            raise ValueError("public payload differs from its canonical projection")
    return dict(status="VERIFIED", scope="archive_only", report=value["identity"], decision=value["decision"],
                denominator=value["denominator"], private_cut=value["private_cut"])


def publish_archive(path, public):
    path = Path(path)
    validate_public(public)
    if path.exists():
        if verify_archive(path)["report"] != public["identity"]:
            raise ValueError("conflicting public audit")
        return verify_archive(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent) as stage:
        with zipfile.ZipFile(stage, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("audit.json", json.dumps(public, sort_keys=True, allow_nan=False))
        stage.flush()
        os.fsync(stage.fileno())
        receipt = verify_archive(stage.name)
        os.link(stage.name, path)
        fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return receipt
