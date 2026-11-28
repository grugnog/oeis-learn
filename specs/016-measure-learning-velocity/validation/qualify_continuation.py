"""Run a pinned donor preparation or qualification and save its verified result."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

DONOR = "77667e67176ae32fa16457d52565bae1dbe41afc"
CACHE_HELPER = "specs/015-learned-program-search/validation/candidate_source.py"
LAUNCHER = "specs/016-measure-learning-velocity/validation/launch-round029.sh"


def checked_json(argv, cwd, action):
    from oeis_learn.generalization.progressive.velocity_records import validate_result

    # Older launchers inherit stdout for Podman builds and finish with one JSON line.
    with tempfile.TemporaryFile() as output:
        proc = subprocess.run(argv, cwd=cwd, stdout=output)
        output.seek(0)
        terminal = b""
        for line in output:
            if line.strip():
                terminal = line
    try:
        value = validate_result(json.loads(terminal))
        if value["action"] != action:
            raise ValueError("foreign action")
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Missing or invalid terminal {action} result (exit {proc.returncode})"
        ) from exc
    if proc.returncode:
        raise SystemExit(json.dumps(value))
    return value


def child_rehearsal(action, root, revision, minutes):
    from oeis_learn.pipeline_validation.progressive_velocity_qualification import (
        authorize_learner_check,
    )

    grant = authorize_learner_check(root, int(minutes), revision, children=True)
    first = action("qualify", grant)
    if first["mechanical_status"] != "COMPLETE" or first["scope"] != "learner_child_restore":
        raise SystemExit(json.dumps(first))
    for _ in range(2):
        if action("qualify", grant) != first:
            raise SystemExit("Child rehearsal replay changed its receipt or debits.")
    record = first["receipt"]
    status, verification = action("status"), action("verify")
    if (
        status["last_committed_ref"] != record["identity"]
        or verification["mechanical_status"] != "VERIFIED"
        or verification["last_committed_ref"] != record["identity"]
        or first["research_dispatches"]
        or verification["scientific_verdict"] is not None
    ):
        raise SystemExit("Private child verification disagrees with committed session evidence.")
    return child_summary(
        revision, record, first["charged_tokens"], first["elapsed_us"], first["remaining_authority"]
    )


def child_summary(revision, record, charged_tokens, elapsed_us, remaining_us):
    return dict(
        candidate=revision,
        receipt=record["identity"],
        children={
            size: {
                k: row["child"][k]
                for k in (
                    "parent",
                    "checkpoint",
                    "useful_tokens",
                    "charged_tokens",
                    "discarded_tokens",
                    "incremental_charged_tokens",
                    "horizon",
                    "verified_replays",
                )
            }
            for size, row in record["batches"].items()
        },
        charged_tokens=charged_tokens,
        elapsed_us=elapsed_us,
        remaining_us=remaining_us,
        lease_us=record["lease_us"],
        peak_device_bytes=record["peak_device_bytes"],
        verified_replays=2,
        research_dispatches=0,
        scientific_verdict=None,
    )


def saved_children(root, source, revision, minutes):
    """Verify the original grant twice, even after a later action has replaced session status."""
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.contracts import digest, load, seal
    from oeis_learn.generalization.progressive.velocity_records import (
        SESSION,
        physical_root_id,
        validate_grant,
    )
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_roles import ROUND029_ROLE_POLICY
    from oeis_learn.pipeline_validation.progressive_velocity_qualification import (
        _receipt,
        qualification_parent,
    )
    from oeis_learn.pipeline_validation.progressive_velocity_worker import rehearsal_work

    folder = confined(root, "run-state/discovery/" + SESSION, exists=False)
    missing = f"No saved child qualification for {revision} with a {minutes}-minute grant"
    if not (folder / "run.lock").is_file():
        raise ValueError(missing)
    with confined(folder, "run.lock").open("rb") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                "A session owner is active; retry the read-only review once it finishes"
            ) from exc
        matches = []
        for path in folder.glob("qualification/*/binding.json"):
            binding = load(confined(folder, path))
            if binding.get("candidate") != revision or binding.get("child_restore") is not True:
                continue
            area = path.parent
            grant = load(confined(area, "grant.json"))
            if grant["total_us"] == minutes * 60_000_000:
                matches.append((area, binding, grant))
        if not matches:
            raise ValueError(missing)
        if len(matches) != 1:
            raise ValueError("Multiple child grants match this candidate and duration")
        area, binding, grant = matches[0]
        expected = seal(
            dict(
                version="round029-learner-rehearsal/v1",
                candidate=revision,
                role_policy=ROUND029_ROLE_POLICY,
                grant=grant["identity"],
                child_restore=True,
            )
        )
        auth = confined(root, "run-state/discovery-authorizations")
        if (
            binding != expected
            or area.name != grant["identity"][7:]
            or load(confined(auth, "grant-" + area.name + ".json")) != grant
        ):
            raise ValueError("Original child binding or retained grant differs")
        approval = load(confined(auth, grant["authorizing_record_ref"][7:] + ".json"))
        validate_grant(
            grant,
            approval,
            root=physical_root_id(root),
            parent=qualification_parent(revision, children=True),
            action="qualify",
        )
        if not (area / "receipt.json").is_file():
            first_error = confined(area, "first-error.json", exists=False)
            reason = (
                load(first_error)["reason"] if first_error.is_file() else "no committed receipt"
            )
            raise ValueError(f"Child grant found without a completed receipt: {reason}")
        if load(confined(area, "request.json"))["source"] != digest(collect_snapshot(source)):
            raise ValueError("Original learner source differs from the requested candidate")
        token_charge = rehearsal_work(children=True)["tokens"]
        reports = []
        for _ in range(2):
            prior = load(confined(area, "receipt.json"))
            record = _receipt(area, confined(area, prior["attempt"]), binding)
            journal = BudgetJournal.inspect_existing(
                confined(area, "accounting"), grant["identity"], grant["resources"]
            )
            if (
                record != prior
                or journal["pending"]
                or len(set(record["debits"])) != 2
                or not set(record["debits"]).issubset(journal["committed"])
                or journal["consumed"]["tokens"] != token_charge
            ):
                raise ValueError(
                    "Child evidence differs from the original receipt or settled debits"
                )
            reports.append(
                child_summary(
                    revision,
                    record,
                    journal["consumed"]["tokens"],
                    journal["consumed"]["elapsed_us"],
                    journal["remaining"]["elapsed_us"],
                )
            )
        if reports[0] != reports[1]:
            raise ValueError("Read-only replays disagree")
        return reports[0]


def donor_supply(action, root, revision, minutes):
    from oeis_learn.pipeline_validation.progressive_velocity_preparation import preparation_parent
    from oeis_learn.pipeline_validation.progressive_velocity_qualification import (
        authorize_qualification,
    )

    grant = authorize_qualification(root, int(minutes), preparation_parent(revision), {})
    try:
        paid = action("qualify", grant)
    except SystemExit as exc:
        # A failed second step can still leave a verified, paid first measurement.
        paid = json.loads(str(exc))
    verified = None
    if paid["mechanical_status"] == "COMPLETE":
        verified = action("verify")
        if (
            verified["verification_scope"] != "private_donor_supply"
            or verified["last_committed_ref"] != paid["last_committed_ref"]
        ):
            raise ValueError("private supply replay differs from the funded preparation")
    inspected = action("preflight")
    info = inspected["inventory"]
    if info["gpu_dispatches"] != 0:
        raise ValueError("donor supply preflight issued device work")
    return dict(
        candidate=revision,
        status=paid["mechanical_status"],
        scope="donor_supply",
        grant=json.loads(grant.read_bytes())["identity"],
        receipt=paid["last_committed_ref"],
        verified=verified and verified["report_ref"],
        elapsed_us=paid.get("elapsed_us"),
        remaining_us=paid["remaining_authority"],
        first_error=paid.get("first_error"),
        qualification_reason=paid["reason"],
        families=info["families"],
        matching_pairs=info["matching_pairs"],
        matching_scope="fixed_target_ordinary_stream",
        prices=info["prices"],
        mount=info["mount"],
        next_action=inspected["next_action"],
        reason=inspected["reason"],
        gpu_dispatches=0,
        research_dispatches=0,
    )


def check_price_links(prices, record, *, fresh):
    key = "fresh_start" if fresh else "broader_updates"
    if (
        prices.get(key) != record
        or (fresh and prices.get("broader_updates", {}).get("identity") != record["overlap"])
        or prices.get("timed_updates", {}).get("measurement") != record["ordinary_measurement"]
    ):
        raise ValueError("Preflight did not reopen the paid timing scopes and their shared calibration")


def donor_timings(action, root, revision, minutes, *, broader=False, fresh=False):
    from oeis_learn.pipeline_validation.progressive_velocity_timing_host import authorize_timing

    from oeis_learn.pipeline_validation.progressive_velocity_broad_host import (
        authorize as authorize_broader,
    )

    if fresh:
        authority = authorize_broader(root, int(minutes), revision, fresh=True)
    else:
        authority = (authorize_broader if broader else authorize_timing)(root, int(minutes), revision)
    try:
        first = action("qualify", authority)
    except SystemExit as exc:
        first = json.loads(str(exc))
    summary = dict(
        candidate=revision,
        scope="fresh_start_timings" if fresh else ("broader_update_timings" if broader else "ordinary_update_timings"),
        status=first["mechanical_status"],
        reason=first["reason"],
        grant=json.loads(authority.read_bytes())["identity"],
        first_error=first.get("first_error"),
        receipt=first["last_committed_ref"],
        elapsed_us=first.get("elapsed_us"),
        remaining_us=first["remaining_authority"],
        charged_tokens=first.get("charged_tokens"),
        research_dispatches=first.get("research_dispatches"),
    )
    if first["mechanical_status"] != "COMPLETE":
        from oeis_learn.pipeline_validation.progressive_velocity_image import failure_details

        current = first.get("diagnostics") or summary["first_error"]
        return dict(
            summary,
            problem=current,
            next_action=first["next_action"],
            diagnostics=failure_details(root, summary["grant"], current),
        )
    verification = action("verify")
    record = first["receipt"]
    expected_scope = (
        "private_fresh_start_timings"
        if fresh
        else ("private_broader_timings" if broader else "private_update_timings")
    )
    if (
        verification["verification_scope"] != expected_scope
        or verification["last_committed_ref"] != record["identity"]
        or first["research_dispatches"] != 0
    ):
        raise ValueError("original timing debit or private replay differs")
    inspected = action("preflight")
    prices = inspected["inventory"]["prices"]
    if broader or fresh:
        check_price_links(prices, record, fresh=fresh)
        return dict(summary, **{"fresh_start" if fresh else "broader": record}, prices=prices, next_action=inspected["reason"])
    if prices.get("timed_updates", {}).get("measurement") != record["measurement"]:
        raise ValueError("preflight did not reopen the original update matrix")
    return dict(
        summary,
        preparation_storage=record.get("storage"),
        image_acquisition=(
            "qualified_runtime_and_candidate_wheel" if record.get("runtime") else "candidate_image"
        ),
        units=record["units"],
        measurement=record["measurement"],
        worker_cpu_us=record["worker_cpu_us"],
        telemetry=record["telemetry"],
        lease_us=record["lease_us"],
        peak_device_bytes=record["peak_device_bytes"],
        prices=prices,
        next_action=inspected["reason"],
    )


def select_fields(record, *keys):
    return {key: record[key] for key in keys if key in record}


def readiness_preflight(reply):
    info = reply["inventory"]
    if info["gpu_dispatches"] != 0:
        raise ValueError("Read-only preflight issued device work")
    amounts = ("device", "available_bytes", "available_inodes", "required_bytes", "required_inodes",
               "reserved_bytes", "reserved_inodes", "deficit_bytes", "deficit_inodes")
    mounts = info["capacity"]["mounts"]
    return dict(
        next_action=reply["next_action"], reason=reply["reason"], gpu_dispatches=0,
        families=select_fields(info["families"], "count", "measurement", "registration", "by_role",
                               "deficits", "eligible_cases", "eligible_components", "excluded_cases",
                               "provenance_unavailable"),
        matching_pairs=select_fields(info["matching_pairs"], "count", "measurement", "registration",
                                     "eligible_families", "positive_rows", "excluded_rows", "dispositions"),
        mount=select_fields(info["mount"], *amounts),
        mounts=[select_fields(row, *amounts) for row in mounts] if mounts is not None else None,
        prices=select_fields(info["prices"], "basis", "common_probe", "wave", "followup",
                             "followup_family_available", "followup_reserve_us", "wave_with_reserve_us",
                             "additional_control_tokens", "missing_control_looks", "prior_score_credits",
                             "unit_prices"),
    )


def reopen_child_rehearsal(root, source):
    validation = source / Path(LAUNCHER).parent
    expected = json.loads((validation / "round029-child-qualification.json").read_bytes())
    total = expected["elapsed_us"] + expected["remaining_us"]
    minutes, remainder = divmod(total, 60_000_000)
    if remainder or minutes < 1:
        raise ValueError("Historical child grant duration differs")
    # The original child grant has its own candidate and source identity.
    command = [sys.executable, "-I", "-S", "-B", str(validation / "qualify_continuation.py"),
               expected["candidate"], str(minutes), "review-children"]
    reopened = json.loads(subprocess.check_output(command, cwd=root))
    if reopened != expected:
        raise ValueError("Reopened child rehearsal differs from the published receipt")
    return expected


def accounting_fixture():
    from oeis_learn.generalization.progressive.program_search_costing import (
        occupied_time, workstation_accounting,
    )

    failed = dict(receipt="first", state="failed", origin="incremental", role="learn",
                  host="fixture", boot="one", start_us=0, end_us=6, charged_us=6, beneficiaries=["00"])
    done = dict(failed, receipt="second", state="complete", start_us=3, end_us=9, beneficiaries=["02"])
    result = workstation_accounting([failed, done, failed])["incremental"]
    occupied = result["occupied"]
    reboot = occupied_time([failed, dict(done, boot="two")])
    if (result["ledger_us"] != 12 or occupied["union_us"] != 9
            or occupied["attributed_us"] != {"00": 4.5, "02": 4.5}
            or result["dispositions"] != {"complete": 1, "failed": 1}
            or (reboot["union_us"], reboot["lower_us"], reboot["upper_us"]) != (None, 6, 12)):
        raise ValueError("Reference workstation accounting differs")
    return dict(scope="reference_fixture", ledger_us=result["ledger_us"], occupied_us=occupied["union_us"],
                attributed_us=occupied["attributed_us"], dispositions=result["dispositions"],
                cross_boot_bounds_us=[reboot["lower_us"], reboot["upper_us"]])


def velocity_fixture():
    from oeis_learn.generalization.progressive.velocity_metrics import affordable_look, factorial, rates

    observed = rates(6, 3_000_000, 5_400_000_000)
    controls = factorial(dict(zip(("00", "01", "02", "03", "04", "05", "06"), (1, 2, 3, 6, 5, 7, 12))))
    interactions = {name: block["interaction"] for name, block in controls.items()}
    looks = [dict(look=n, verified=(n != 2), cost=n) for n in (1, 2, 4)]
    within = affordable_look(looks, "cost", 3, overlap=(1, 4))
    outside = affordable_look(looks, "cost", 5, overlap=(1, 4))
    if (observed != {"pp_per_million": 2, "pp_per_hour": 4}
            or interactions != {"composition_matching": 2, "batch_decay": 1}
            or within != looks[0] or outside is not None):
        raise ValueError("Reference velocity reduction differs")
    return dict(scope="reference_fixture", rates=observed, interactions=interactions,
                affordable_look=within["look"], outside_overlap=outside)


def donor_readiness(action, root, source, revision, minutes, supplied):
    """Collect an already issued qualification. Every launcher action here is read-only."""
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.generalization.progressive.velocity_records import SESSION
    from oeis_learn.pipeline_validation.identity import confined

    checks, errors, files = {}, {}, {}

    def inspect(name, operation):
        try:
            result = operation()
            checks[name] = "PASS"
            return result
        except (Exception, SystemExit) as exc:
            checks[name] = "FAILED"
            errors[name] = type(exc).__name__
            print(f"{name}: {exc}", file=sys.stderr)
            return None

    def read_result():
        saved = json.loads(supplied.read_bytes())
        if saved["candidate"] != revision:
            raise ValueError("Supplied qualification has a different candidate")
        if saved["status"] == "UNVERIFIED" and saved.get("mode") == "fresh-start":
            return dict(candidate=revision, status="UNVERIFIED")
        if (saved["scope"] != "fresh_start_timings" or saved["status"] not in {"COMPLETE", "BLOCKED", "FAILED"}
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", saved["grant"])
                or any(type(saved[key]) is not int or saved[key] < 0
                       for key in ("elapsed_us", "remaining_us", "charged_tokens"))
                or saved["elapsed_us"] + saved["remaining_us"] != int(minutes) * 60_000_000
                or saved["research_dispatches"] != 0):
            raise ValueError("Supplied qualification scope, grant or debits differ")
        if saved["status"] == "COMPLETE" and saved["receipt"] != saved["fresh_start"]["identity"]:
            raise ValueError("Supplied qualification receipt differs")
        return saved

    saved = inspect("fresh-start", read_result)
    if saved is not None:
        summary = select_fields(saved, "candidate", "scope", "status", "grant", "receipt", "elapsed_us",
                                "remaining_us", "charged_tokens", "research_dispatches")
        if saved["status"] != "COMPLETE":
            checks["fresh-start"] = saved["status"]
            problem = saved.get("problem") or saved.get("first_error") or {}
            summary.update(select_fields(problem, "deficit_bytes", "deficit_inodes"))
            if problem.get("stage") in {"custodian", "package", "prepare", "measure"}:
                summary["stage"] = problem["stage"]
            if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,64}", str(problem.get("error", ""))):
                summary["error"] = problem["error"]
            if saved.get("next_action") in {"inspect_capacity", "inspect_attempt"}:
                summary["next_action"] = saved["next_action"]
        files["fresh-start.json"] = summary

    status = inspect("status", lambda: action("status"))
    if status is not None:
        files["status.json"] = select_fields(status, "owner_active", "mechanical_status", "next_action",
                                             "last_committed_ref", "remaining_authority")
        if "progress" in status:
            files["status.json"]["progress"] = select_fields(
                status["progress"], "scope", "stage", "mechanical_status", "elapsed_us",
                "charged_tokens", "research_dispatches",
            )

    def preflight():
        reply = action("preflight")
        files["preflight.json"] = readiness_preflight(reply)
        return reply["inventory"]["prices"]

    prices = inspect("preflight", preflight)

    def verify():
        checked = action("verify")
        if (checked["mechanical_status"] != "VERIFIED"
                or checked["verification_scope"] != "private_fresh_start_timings"
                or checked["last_committed_ref"] != saved["receipt"]
                or checked["report_ref"] != saved["receipt"]
                or checked["scientific_verdict"] is not None):
            raise ValueError("Private verification did not reopen the supplied receipt")
        return select_fields(checked, "mechanical_status", "verification_scope", "last_committed_ref",
                             "report_ref", "scientific_verdict")

    def link_receipts():
        progress = status["progress"]
        path = f"run-state/discovery/{SESSION}/qualification/{saved['grant'][7:]}/receipt.json"
        binding = load(confined(root, Path(path).with_name("binding.json")))
        if (status["owner_active"] or status["last_committed_ref"] != saved["receipt"]
                or binding["candidate"] != revision or binding["grant"] != saved["grant"]
                or progress["receipt_path"] != path or progress["receipt"] != saved["fresh_start"]
                or progress["elapsed_us"] != saved["elapsed_us"]
                or progress["remaining_authority"] != saved["remaining_us"]
                or progress["charged_tokens"] != saved["charged_tokens"]
                or saved["fresh_start"]["charged_work"]["tokens"] != saved["charged_tokens"]):
            raise ValueError("Supplied grant, receipt or debits differ from the private session")
        check_price_links(prices, progress["receipt"], fresh=True)
        files["fresh-start.json"]["fresh_start"] = progress["receipt"]

    checks.update({"verification": "NOT_RUN", "receipt-links": "NOT_RUN"})
    if saved is not None and saved["status"] == "COMPLETE":
        verified = inspect("verification", verify)
        if verified is not None:
            files["verification.json"] = verified
            if prices is not None and status is not None:
                inspect("receipt-links", link_receipts)
    for name, op in (("recovery", lambda: reopen_child_rehearsal(root, source)),
                     ("accounting-fixture", accounting_fixture), ("velocity-fixture", velocity_fixture)):
        result = inspect(name, op)
        if result is not None:
            files[name + ".json"] = result
    return dict(candidate=revision, scope="round029-readiness/v1", grant_minutes=int(minutes),
                status="COMPLETE" if all(v == "PASS" for v in checks.values()) else "PARTIAL",
                checks=checks, errors=errors, research_run_requested=False, scientific_verdict=None, files=files)


def main(revision, minutes, mode="donation", supplied=None):
    if mode not in {"donation", "children", "review-children", "supply", "timings", "broader", "fresh-start", "readiness"}:
        raise SystemExit("Unknown continuation qualification mode.")
    root = Path.cwd().resolve()
    if not re.fullmatch(r"[a-f0-9]{40}", revision) or not re.fullmatch(r"[1-9][0-9]*", minutes):
        raise SystemExit("Supply a full candidate commit and a positive maximum number of minutes.")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root).decode().strip()

    if (
        git("rev-parse", "--show-toplevel") != str(root)
        or git("rev-parse", "HEAD") != DONOR
        or subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"], cwd=root, stdout=subprocess.DEVNULL
        ).returncode
        != 1
        or git("rev-parse", "--verify", revision + "^{commit}") != revision
    ):
        raise SystemExit(
            "Run from the physical root of the detached round027 donor with the candidate fetched."
        )
    helper = subprocess.check_output(["git", "show", revision + ":" + CACHE_HELPER], cwd=root)
    source = Path(
        subprocess.check_output(
            [sys.executable, "-I", "-S", "-B", "-", str(root), revision], input=helper, cwd=root
        )
        .decode()
        .strip()
    )
    sys.path.insert(0, str(source / "src"))
    if mode == "review-children":
        return saved_children(root, source, revision, int(minutes))
    launcher = ["bash", str(source / LAUNCHER), revision]

    def action(name, *args):
        return checked_json([*launcher, name, *map(str, args)], root, name)

    if mode == "readiness":
        return donor_readiness(action, root, source, revision, minutes, supplied)
    if mode in {"timings", "broader", "fresh-start"}:
        return donor_timings(action, root, revision, minutes, broader=mode == "broader", fresh=mode == "fresh-start")
    if mode == "supply":
        return donor_supply(action, root, revision, minutes)
    inventory = action("preflight")["inventory"]
    if inventory["gpu_dispatches"] != 0:
        raise SystemExit("A read-only preflight issued device work.")
    if mode == "children":
        return child_rehearsal(action, root, revision, minutes)
    ordinary = next(r for r in inventory["cuts"] if (r["method"], r["look"]) == ("00", 4_000_000))
    if not ordinary["full_state"] or inventory["gpu_dispatches"] != 0:
        raise SystemExit(
            "Preflight could not authenticate the ordinary 4M full state: " + ordinary["reason"]
        )

    from oeis_learn.pipeline_validation.progressive_velocity_donation import (
        authorize_donation_check,
    )

    authority = authorize_donation_check(root, int(minutes), revision)
    first = action("qualify", authority)
    if first["mechanical_status"] != "COMPLETE" or first["scope"] != "donated_ordinary_next_update":
        raise SystemExit(json.dumps(first))
    for _ in range(2):
        if action("qualify", authority) != first:
            raise SystemExit("Repeated qualification changed the original outcome or charges.")
    if action("status")["last_committed_ref"] != first["last_committed_ref"]:
        raise SystemExit("Committed session status differs from the qualification receipt.")
    private = action("verify")
    share = (root / first["receipt_path"]).parent / "share"
    grant_id = json.loads(authority.read_bytes())["identity"][7:]
    retained = root / "run-state/discovery-authorizations" / (grant_id + "-prescore.zip")
    with tempfile.TemporaryDirectory(prefix="round029-public-") as directory:
        out = Path(directory)
        report = shutil.copyfile(share / "round029-report.zip", out / "round029-report.zip")
        before = shutil.copyfile(retained, out / "round029-qualification.zip")
        public = checked_json(
            [
                sys.executable,
                "-S",
                "-B",
                str(source / "scripts/foundation_run.py"),
                "--engine",
                "podman",
                "discovery",
                "round029",
                "--action",
                "verify",
                "--archive",
                str(report),
                "--qualification-archive",
                str(before),
            ],
            out,
            "verify",
        )
    if (
        public["report_ref"] != private["report_ref"]
        or public["last_committed_ref"] != first["last_committed_ref"]
        or public["scientific_verdict"] is not None
    ):
        raise SystemExit("Independent diagnostic replay differs from the private workspace.")
    record = first["receipt"]
    return dict(
        candidate=revision,
        receipt=record["identity"],
        report=public["report_ref"],
        parent_cut=ordinary["full_state"],
        parent_useful_tokens=ordinary["useful_tokens"],
        next_update_tokens=record["comparison"]["charged_tokens"] // 2,
        charged_tokens=first["charged_tokens"],
        elapsed_us=first["elapsed_us"],
        remaining_us=first["remaining_authority"],
        lease_us=record["lease_us"],
        peak_device_bytes=record["peak_device_bytes"],
        verified_replays=2,
        research_dispatches=first["research_dispatches"],
        scientific_verdict=None,
    )


def write_readiness_zip(out, report):
    def encoded(value):
        return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")

    members = report.pop("files", {})
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        hashes = {}
        for name, value in members.items():
            data = encoded(value)
            archive.writestr(name, data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        archive.writestr("summary.json", encoded(dict(report, sha256=hashes)))


def cli(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", metavar="FULL_COMMIT")
    parser.add_argument("minutes", metavar="MAXIMUM_MINUTES")
    parser.add_argument(
        "mode",
        nargs="?",
        default="donation",
        choices=("donation", "children", "review-children", "supply", "timings", "broader", "fresh-start", "readiness"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="save the result to a new file (ZIP for readiness, JSON for other modes)",
    )
    parser.add_argument("--from-result", type=Path, help="completed fresh-start JSON for read-only readiness")
    args = parser.parse_args(argv)
    readiness = args.mode == "readiness"
    if readiness and (args.from_result is None or not args.output or args.output.suffix.lower() != ".zip"):
        parser.error("Readiness requires --from-result QUALIFICATION.json and --output NEW.zip")
    if args.from_result is not None and not readiness:
        parser.error("--from-result is only valid with readiness")
    if args.mode in {"supply", "timings", "broader", "fresh-start"} and args.output is None:
        parser.error("Donor preparation requires an output file (--output PATH)")
    request = (args.revision, args.minutes, args.mode)
    if readiness:
        request += (args.from_result,)
    if args.output is None:
        print(json.dumps(main(*request), indent=2))
        return 0
    if os.path.lexists(args.output):
        parser.error(f"Output already exists; choose a new file: {args.output}")
    # Open the destination directory before any funded work. Publish only complete bytes.
    with tempfile.NamedTemporaryFile(dir=args.output.parent, mode="w+b") as out:
        try:
            report = main(*request)
            code = int(report.get("status") in {"FAILED", "BLOCKED", "PARTIAL", "UNVERIFIED"})
        except (SystemExit, Exception) as exc:
            report = dict(
                candidate=args.revision,
                mode=args.mode,
                status="UNVERIFIED",
                error=type(exc).__name__,
            )
            if readiness:
                print(f"readiness: {exc}", file=sys.stderr)
                report.update(scope="round029-readiness/v1", research_run_requested=False,
                              scientific_verdict=None, checks={"bootstrap": "FAILED"})
            else:
                report["reason"] = str(exc)
            code = 1
        if readiness:
            write_readiness_zip(out, report)
        else:
            out.write((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        out.flush()
        os.fsync(out.fileno())
        os.link(out.name, args.output)
    print(f"Saved {args.output.resolve()}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))
