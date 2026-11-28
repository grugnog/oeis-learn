"""Run a pinned donor preparation or qualification and save its verified result."""

import argparse
import base64
import fcntl
import hashlib
import io
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
CONDITIONING = "round029-paired-conditioning-prices/v1"
BANK_PRICES = "round029-matched-bank-prices/v1"
LAUNCH_PRICES = "round029-launch-preparation/v1"
INITIALIZATION = "round029-ordinary-initialization/v1"
TRIAL_PREPARATION = "round029-supported-trial-preparation/v1"
INITIAL_WORKER = "3c954c15a93a3ba08264910a038efb926e38bfeb"
INITIAL_GRANT = "sha256:a38310c4081a609e16f9014f249363653a126ec2896c860d4861eefb65484fa5"
ZIP_SCOPES = {
    "initialization": INITIALIZATION,
    "readiness": "round029-readiness/v1",
    "checkpoints": "checkpoint_matrix",
    "control-history": "round029-control-history/v1",
    "isolated-controls": "round029-isolated-control-prices/v1",
    "bank-prices": BANK_PRICES,
    "conditioning": CONDITIONING,
    "launch-prices": LAUNCH_PRICES,
}
MODES = ("donation", "children", "review-children", "supply", "timings", "broader", "fresh-start", *ZIP_SCOPES)


def checked_json(argv, cwd, action, *, log=None):
    from oeis_learn.generalization.progressive.velocity_records import validate_result

    # Older launchers inherit stdout for Podman builds and finish with one JSON line.
    with tempfile.TemporaryFile() as output:
        proc = subprocess.run(argv, cwd=cwd, stdout=output, stderr=log)
        output.seek(0)
        terminal = b""
        for line in output:
            if log is not None:
                log.write(line)
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


def donor_checkpoints(action, root, source, revision, minutes):
    from oeis_learn.pipeline_validation.progressive_velocity_checkpoint_host import authorize, incomplete

    grant = authorize(root, source, revision, minutes)
    try:
        first = action("qualify", grant)
    except SystemExit as exc:
        first = json.loads(str(exc))
    files = {}
    report = dict(candidate=revision, scope="checkpoint_matrix", grant=json.loads(grant.read_bytes())["identity"],
                  status=first["mechanical_status"], research_run_requested=False, scientific_verdict=None, files=files,
                  **select_fields(first, "elapsed_us", "remaining_authority", "charged_tokens", "research_dispatches",
                                  "stage", "next_action", "deficit_bytes", "deficit_inodes"))
    if first["mechanical_status"] != "COMPLETE":
        problem = first.get("diagnostics") or first.get("first_error") or {}
        files["failure.json"] = select_fields(problem, "stage", "error", "capacity", "deficit_bytes", "deficit_inodes")
        if first.get("stage") == "measure":
            try:
                for key, observation in incomplete(root, source, grant).items():
                    files["unverified-" + key + ".json"] = observation
            except (OSError, ValueError, KeyError, TypeError) as exc:
                files["failure.json"]["partial_read_error"] = type(exc).__name__
        return report
    record = first["receipt"]
    files["receipt.json"] = record
    try:
        verify = action("verify")
        if (first["scope"] != "checkpoint_matrix" or first["research_dispatches"] != 0
                or verify["verification_scope"] != "private_checkpoint_matrix"
                or verify["mechanical_status"] != "VERIFIED" or verify["report_ref"] != record["identity"]
                or verify["last_committed_ref"] != first["last_committed_ref"] or verify["scientific_verdict"] is not None
                or report["elapsed_us"] + report["remaining_authority"] != int(minutes) * 60_000_000):
            raise ValueError("verified checkpoint matrix or grant balance differs")
        files["verification.json"] = select_fields(verify, "mechanical_status", "last_committed_ref", "verification_scope")
        files["capacity.json"] = dict(package=record["package_capacity"], device=record["capacity"])
        for method, value in record["methods"].items():
            files["recipe-" + method + ".json"] = value
    except (Exception, SystemExit) as exc:
        report.update(status="UNVERIFIED", error=type(exc).__name__)
        print(f"private checkpoint verification: {exc}", file=sys.stderr)
        return report
    try:
        flight = action("preflight")
        prices = flight["inventory"]["prices"]
        if (flight["inventory"]["gpu_dispatches"] != 0
                or prices.get("checkpoint_cycles", {}).get("receipt") != record["identity"]):
            raise ValueError("preflight did not reopen the verified checkpoint receipt")
        files["forecast.json"] = dict(checkpoint_receipt=record["identity"], **select_fields(
            prices, "common_probe", "wave", "followup", "wave_with_reserve_us", "followup_reserve_us",
            "followup_family_available"))
    except (Exception, SystemExit) as exc:
        report.update(status="PARTIAL", forecast_error=type(exc).__name__)
        print(f"checkpoint preflight: {exc}", file=sys.stderr)
    return report


CONTROL_MODELS = ("retained_D", "random")


def new_cell(expected):
    return dict(
        expected_cases=expected,
        seen=0,
        issued=0,
        empty=0,
        search_clocks=0,
        score_clocks=0,
        search_us=0,
        scoring_us=0,
        paired_cases=0,
        paired_us=0,
        observed_max_us=None,
    )


def microseconds(seconds):
    from math import ceil, isfinite

    if seconds is None:
        return None
    if type(seconds) not in (int, float) or not isfinite(seconds) or seconds < 0:
        raise ValueError("invalid original elapsed seconds")
    return ceil(seconds * 1_000_000)


def observe(cell, method, record, score):
    from oeis_learn.generalization.progressive.program_search_bank import selected_source

    selected = selected_source(record["selections"][method], score, None)
    if score.get("upstream") is not None and score["upstream"] != record["arm_evidence"][method]:
        raise ValueError("selected score differs from its upstream method")
    cost = record["costs"].get(method, {})
    if method == "template":
        elapsed = microseconds(cost.get("elapsed_seconds"))
        attempts = len(cost["details"]) if "details" in cost else None
        allowance = 8
    else:
        elapsed = cost.get("total_elapsed_us")
        attempts = cost.get("attempts")
        allowance = 1
    if elapsed is not None and (type(elapsed) is not int or elapsed < 0):
        raise ValueError("invalid original search elapsed time")
    if attempts is not None and (type(attempts) is not int or not 0 <= attempts <= allowance):
        raise ValueError("historical attempt count differs from the registered control")
    scorer = score.get("scoring") or {}
    scoring = microseconds(scorer.get("elapsed_seconds"))
    cell["seen"] += 1
    cell["issued"] += attempts == allowance
    cell["empty"] += selected is None
    if elapsed is not None:
        cell["search_clocks"] += 1
        cell["search_us"] += elapsed
    if scoring is not None:
        cell["score_clocks"] += 1
        cell["scoring_us"] += scoring
    if (
        elapsed is not None
        and scoring is not None
        and attempts == allowance
        and (selected is None or scorer.get("complete") is True)
    ):
        wall = elapsed + scoring
        cell["paired_cases"] += 1
        cell["paired_us"] += wall
        cell["observed_max_us"] = max(cell["observed_max_us"] or 0, wall)


def finish_cell(cell):
    full = cell["expected_cases"] > 0 and cell["paired_cases"] == cell["expected_cases"]
    cell["mean_us"] = (
        (cell["paired_us"] + cell["paired_cases"] - 1) // cell["paired_cases"] if full else None
    )
    cell["max_us"] = cell["observed_max_us"] if full else None


def control_prices(root, members, producers):
    from collections import Counter
    from oeis_learn.generalization.progressive.cohorts import STRATA
    from oeis_learn.generalization.progressive.contracts import read_bound_record
    from oeis_learn.pipeline_validation.progressive_round028_audit import METHODS

    counts = Counter(members.values())
    if counts != dict.fromkeys(STRATA, 32):
        raise ValueError("original exposed development roster differs")
    controls = {m: {} for m in METHODS}
    coverage = {}
    complete = True
    used = set()
    for model in CONTROL_MODELS:
        strata = {m: {s: new_cell(counts[s]) for s in STRATA} for m in METHODS}
        seen, missing = set(), []
        for block in range(8):
            pair = producers.get(f"capability-{model}-{block:02d}")
            if pair is None:
                missing.append(block)
                continue
            try:
                gen = read_bound_record(root, pair["generation"], version="discovery-candidates/v1")
                scores = read_bound_record(root, pair["scoring"], version="discovery-scoring/v1")
            except FileNotFoundError:
                missing.append(block)
                continue
            parent = pair["generation"]["sha256"]
            if (
                parent in used
                or gen["model_control"] != model
                or scores["candidates"] != parent
                or scores["denominator"] != len(scores["outcomes"])
                or gen["records"].keys() != scores["outcomes"].keys()
            ):
                raise ValueError("historical control partition or scorer parent differs")
            used.add(parent)
            cases = members.keys() & gen["records"].keys()
            if seen & cases:
                raise ValueError("repeated historical development case within a model")
            seen.update(cases)
            for case in cases:
                for method in METHODS:
                    if method in gen["methods"]:
                        observe(
                            strata[method][members[case]],
                            method,
                            gen["records"][case],
                            scores["outcomes"][case][method],
                        )
        coverage[model] = dict(
            expected_partitions=8,
            missing_partitions=missing,
            expected_cases=len(members),
            covered_cases=len(seen),
        )
        for method, cells in strata.items():
            for cell in cells.values():
                finish_cell(cell)
            controls[method][model] = dict(partitions=8 - len(missing), strata=cells)
            if missing or any(c["mean_us"] is None for c in cells.values()):
                complete = False
    return dict(
        status="COMPLETE" if complete else "PARTIAL",
        controls=controls,
        coverage=coverage,
        basis="closed historical development partitions; one independent observation per model and case",
        price_scope="historical search and selected-choice scoring only",
        current_isolated_control_price_us=None,
    )


def frozen_control_study(root):
    """Open only the original exposed study roster and its conventional-control producers."""
    from functools import partial
    from oeis_learn.generalization.progressive.campaign import learned_search_policy
    from oeis_learn.generalization.progressive.contracts import (
        CAMPAIGN_DOCUMENT_BYTES,
        read_bound_record,
        reference,
    )
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_round028_audit import ORIGINAL, roster_map

    policy = learned_search_policy()
    bound = partial(read_bound_record, root)
    path = confined(root, ORIGINAL + "/prepare-research/frozen/campaign.json")
    if path.stat().st_size > CAMPAIGN_DOCUMENT_BYTES:
        raise ValueError("original frozen campaign exceeds the registered size")
    campaign = bound(reference(root, path), identity=policy["campaign"])
    entries = [r for r in campaign["payload"]["references"] if r["name"] == "study-policy"]
    if len(entries) != 1 or (entries[0]["role"], entries[0]["kind"]) != ("proposal", "policy"):
        raise ValueError("original study policy binding differs")
    study = bound(entries[0]["artifact"])
    if any(
        plan["trial"]["study"] != study["identity"]
        for plan in campaign["payload"]["tasks"]
        if plan.get("trial")
    ):
        raise ValueError("original trial study differs")
    roster = roster_map(study["evaluations"]["trial-05-1000000-development-score"]["roster"])
    return policy, campaign, roster


def read_history(root):
    from functools import partial
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive.contracts import read_bound_record
    from oeis_learn.pipeline_validation.progressive_round028_audit import metric_ref

    policy, _, roster = frozen_control_study(root)
    bound = partial(read_bound_record, root)
    cut = policy["capability_cut"]
    cut_path = f"run-state/discovery-control/{policy['campaign'][7:]}/{cut['name']}"
    control = bound(dict(path=cut_path, bytes=cut["bytes"], sha256=cut["sha256"]))
    outputs = control["state"]["outputs"]
    donor = SimpleNamespace(root=root, policy=policy, bound=bound)
    producers = {}
    for model in CONTROL_MODELS:
        for block in range(8):
            tag = f"capability-{model}-{block:02d}"
            if tag + "-generate" not in outputs or tag + "-score" not in outputs:
                continue
            try:
                generation = metric_ref(donor, outputs[tag + "-generate"], "generate", "candidates")
                scoring = metric_ref(donor, outputs[tag + "-score"], "score", "scores")
            except FileNotFoundError:
                continue
            producers[tag] = dict(generation=generation, scoring=scoring)
    return control_prices(root, {k: value[0] for k, value in roster.items()}, producers)


CONTROL_METHODS = ("template", "polynomial", "linear_recurrence")
ISOLATED_CONTROLS = "round029-isolated-control-prices/v1"


BANK_SCORES = "round029-exposed-bank-scores/v1"


CONDITIONING_ROLE = "round029-conditioning-role/v1"
CONDITIONING_GROUPS = {"ordinary-common": ("00", "common"), "ordinary-full": ("00", "full"), "D-full": ("D", "full")}
CONDITIONING_MODES = ("true", "shuffled")


def conditioning_coordinates(grant, group, phase, seed):
    return dict(
        study=grant["identity"],
        stage="qualification",
        method=group,
        seed=seed,
        look=4_000_000,
        panel="exposed_development",
        profile=CONDITIONING_GROUPS.get(group, (None, "models"))[1],
        input_mode="paired",
        role=phase,
        ordinal=0,
    )


def conditioning_leases(area, grant, seed, balance):
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.generalization.progressive.velocity_records import operation_key
    from oeis_learn.pipeline_validation.identity import confined

    issued = leases = lease_us = 0
    for group in CONDITIONING_GROUPS:
        logical = operation_key(conditioning_coordinates(grant, group, "search", seed), 0)[
            "logical"
        ]
        for key in balance["committed"]:
            if key.startswith(logical + "/dispatch/"):
                issued += 1
                attempt = int(key.rsplit("/", 1)[-1])
                path = confined(
                    area,
                    f"operations/{logical[7:]}/attempt-{attempt:04d}/gpu-lease.json",
                    exists=False,
                )
                if path.is_file():
                    us = load(path)["elapsed_us"]
                    if type(us) is not int or us <= 0:
                        raise ValueError("invalid original GPU lease clock")
                    leases += 1
                    lease_us += us
    return issued, leases, lease_us


def conditioning_work(group, phase, cases, bank_programs=None):
    from oeis_learn.generalization.progressive.budgets import role_work_requirements

    if phase == "models":
        return {}
    n = cases * len(CONDITIONING_MODES)
    if phase == "search":
        work = role_work_requirements(
            "generate", [dict(kind="visible_prompt")] * n + [dict(kind="oeis_generation")]
        )
        if CONDITIONING_GROUPS[group][1] == "common":
            work.update(draws=n, attempts=n, rejections=n)
    else:
        work = role_work_requirements("score", [dict(kind="development_truth")] * n)
        if type(bank_programs) is not int or not 0 <= bank_programs <= 64 * n:
            raise ValueError("bank request exceeds the registered number of distinct admissions")
        work["native_work"] += bank_programs
    return {k: v for k, v in work.items() if k != "elapsed_us" and v}


def check_paired_search(record, out, request, data):
    from oeis_learn.generalization.progressive.contracts import digest, read_bound_record, seal
    from oeis_learn.pipeline_validation.pilot_clocks import CORRECT_FULL, DESCRIPTIVE, clock_record
    from oeis_learn.pipeline_validation.progressive_timing_worker import CLOCK_FIELDS

    if (
        record["request"] != request["identity"]
        or record["phase"] != "search"
        or set(record["modes"]) != set(CONDITIONING_MODES)
    ):
        raise ValueError("paired search has a different request or mode roster")
    total_programs = 0
    evidence = {}
    for mode in CONDITIONING_MODES:
        probe = request["probes"][mode]
        model, look = probe["model"], probe["look"]
        item = record["modes"][mode]
        ref = item["artifact"]
        if ref["path"] != mode + "/candidates.json":
            raise ValueError("unexpected paired candidate location")
        gen = read_bound_record(out, ref, version="discovery-candidates/v1")
        clock = gen["clock"]
        policy = DESCRIPTIVE if look["profile"] == "common" else CORRECT_FULL
        keys = {r["task"] for r in look["members"]}
        if (
            gen["model"] != model["weights"]
            or gen["recipe"] != model["profile"]
            or gen["checkpoint"] != model["checkpoint"]
            or gen["methods"] != ["correct"]
            or gen["model_control"]
            != ("retained_D" if look["method"] == "D" else "earlier_checkpoint")
            or gen["prompts"] != digest(sorted(data["prompts"], key=lambda p: p["id"]))
            or clock != seal(clock)
            or clock["policy"] != policy
            or clock["velocity_probe"] != probe["identity"]
            or clock["inference_seed"] != probe["inference_seed"]
            or clock["work"] != model["weights"]
            or clock["source"] != model["profile"]
            or clock["expected_portfolios"] != len(keys)
            or clock["missing_portfolios"] != 0
            or set(gen["records"]) != keys
            or len(clock["clocks"]) != len(keys)
            or {c["prompt"] for c in clock["clocks"]} != keys
            or set(item["case_us"]) != keys
            or not 0 < gen["elapsed_us"] <= item["elapsed_us"]
            or any(
                type(t) is not int or t < 0
                for t in (*item["components"].values(), *item["case_us"].values())
            )
        ):
            raise ValueError("registered model, opportunity, conditioning or search clock differs")
        for case in clock["clocks"]:
            p = gen["records"][case["prompt"]]["costs"]["correct"]
            if (
                case != clock_record(**{k: case[k] for k in CLOCK_FIELDS})
                or case["mode"] != "correct"
                or case["snapshot"] != model["weights"]
                or case["policy"] != policy
                or case["expected"] != look["attempts"]
                or case["started"] != len(p["details"])
            ):
                raise ValueError("issued attempts differ from the registered opportunity")
            archive = p.get("archive", {})
            count = sum(len(archive.get(k, [])) for k in ("retained", "omitted"))
            if count > 64:
                raise ValueError("bank exceeds original per-case limit")
            total_programs += count
        evidence[mode] = item
    return dict(modes=evidence, bank_programs=total_programs)


def check_paired_scores(record, out, request, data, generations):
    from oeis_learn.generalization.progressive.contracts import digest, read_bound_record, seal

    if (
        record["request"] != request["identity"]
        or record["phase"] != "score"
        or set(record["modes"]) != set(CONDITIONING_MODES)
    ):
        raise ValueError("paired scoring has a different request or mode roster")
    modes, outcomes = {}, {}
    for mode in CONDITIONING_MODES:
        probe, search = request["probes"][mode], request["search"][mode]
        item = record["modes"][mode]
        if (
            item["selected"]["path"] != mode + "/selected/scores.json"
            or item["bank"]["path"] != mode + "/output/bank/scores.json"
            or any(type(item[k]) is not int or item[k] < 0 for k in ("bank_us", "selected_us"))
        ):
            raise ValueError("unexpected paired scoring artifact or clock")
        selected = read_bound_record(out, item["selected"], version="discovery-scoring/v1")
        scores = read_bound_record(out, item["bank"], version=BANK_SCORES)
        bank, cells = paired_bank_projection(
            probe,
            data["truth"],
            generations[mode],
            selected,
            search["artifact"]["sha256"],
            search["case_us"],
        )
        bank_req = seal(dict(pair=request["identity"], mode=mode, data=digest(bank)))
        checked = bank_cost_cells(bank_req, bank, scores)
        outcomes[mode] = {k: v["equal_budget"]["status"] for k, v in selected["outcomes"].items()}
        if set(outcomes[mode].values()) - {"solve", "miss", "unresolved"}:
            raise ValueError("unknown selected disposition")
        modes[mode] = dict(
            selected_us=item["selected_us"],
            bank_us=item["bank_us"],
            selected_cells=cells,
            bank_cells=checked,
        )
    counts = dict(both=0, true_only=0, shuffled_only=0, neither=0, unresolved=0)
    for key, a in outcomes["true"].items():
        b = outcomes["shuffled"][key]
        category = (
            "unresolved"
            if "unresolved" in (a, b)
            else "both"
            if a == b == "solve"
            else "true_only"
            if a == "solve"
            else "shuffled_only"
            if b == "solve"
            else "neither"
        )
        counts[category] += 1
    return dict(modes=modes, paired_selected=counts)


def conditioning_probes(registration, rows, prompts, models, tensors):
    from oeis_learn.generalization.progressive.contracts import digest
    from oeis_learn.generalization.progressive.graph import velocity_look
    from oeis_learn.pipeline_validation.progressive_evaluation import model_receipt, velocity_probe

    pairs = {}
    for name, (method, profile) in CONDITIONING_GROUPS.items():
        original = models[method]
        look = velocity_look(
            registration=registration,
            method=method,
            seed=original["seed"],
            look=4_000_000,
            profile=profile,
            development=rows,
            retention=[],
        )
        model = model_receipt(
            look,
            profile=digest(original["recipe"]),
            weights=tensors[method],
            checkpoint=original["weight"]["sha256"],
            origin=original["origin"],
        )
        pairs[name] = {
            mode: velocity_probe(look, prompts, mode=mode, inference_seed=57, model=model)
            for mode in CONDITIONING_MODES
        }
    return pairs


def paired_model_inputs(root, cuts):
    """Project only the two weight files and their authenticated model recipes to the model reader."""
    from oeis_learn.generalization.progressive.contracts import digest, load
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    from oeis_learn.pipeline_validation.progressive_velocity_census import training_source
    from oeis_learn.pipeline_validation.progressive_velocity_donor import donated_ordinary

    freeze, training, _ = training_source(root)
    donated, mounts = donated_ordinary(root)
    weights = donated["parent"]["weights"]
    destination = "/work/inputs/previous/" + weights["path"]
    (ordinary_path,) = [path for path, dest in mounts if dest == destination]
    _, campaign, _ = frozen_control_study(root)
    (recipe_ref,) = [r for r in campaign["payload"]["references"] if r["name"] == "recipe"]
    if (recipe_ref["role"], recipe_ref["kind"]) != ("custodian", "recipe"):
        raise ValueError("the original D recipe authority differs")
    base_recipe = load(checked_file(root, recipe_ref["artifact"]))
    models, paths = {}, {}
    for method, original, path, recipe, look in (
        ("00", weights, ordinary_path, training["profile"], 4_000_000),
        ("D", freeze["reference"], checked_file(root, freeze["reference"]), base_recipe, None),
    ):
        (cut,) = [c for c in cuts if c["method"] == method and c["look"] == look]
        if (
            cut["inference"] is not True
            or original["sha256"] != cut["weights"]
            or method == "00"
            and (
                cut["lineage"] != training["stream"]["identity"]
                or cut["seeds"] != freeze["seeds"]
                or donated["parent"]["checkpoint"]["sha256"] != cut["full_state"]
            )
            or method == "D"
            and digest(recipe) != cut["recipe"]
        ):
            raise ValueError("model differs from the qualified original")
        models[method] = dict(
            method=method,
            seed=freeze["seeds"]["initialization"],
            recipe=recipe,
            weight=dict(original, path="inputs/" + method + ".pt"),
            origin=digest([cut["registration"], cut["lineage"], cut["weights"]]),
        )
        paths[method] = path
    return models, paths


def paired_bank_projection(probe, truths, generated, scores, generation_hash, case_us):
    from oeis_learn.pipeline_validation.pilot_clocks import CORRECT_FULL, DESCRIPTIVE
    from oeis_learn.pipeline_validation.progressive_timing_worker import observations

    rows = probe["look"]["members"]
    by_id = {r["id"]: r for r in truths}
    truth = [by_id[r["task"]] for r in rows]
    prompts = [dict(id=t["id"], indices=t["indices"][:20], values=t["values"][:20]) for t in truth]
    common = probe["look"]["profile"] == "common"
    cells = observations(
        generated,
        scores,
        {r["task"]: r["stratum"] for r in rows},
        case_us,
        generation_hash,
        DESCRIPTIVE if common else CORRECT_FULL,
    )
    if any(
        scores["outcomes"][p["id"]]["equal_budget"]["probe_ref"] != probe["identity"]
        for p in prompts
    ):
        raise ValueError("selected score has a different conditioning parent")
    return admitted_bank_rows(rows, prompts, truth, generated, scores), cells


def conditioning_role(work, request, data, client, *, device):
    import time
    from oeis_learn.generalization.progressive.baselines import load_baseline
    from oeis_learn.generalization.progressive.contracts import (
        atomic_write,
        digest,
        read_bound_record,
        reference,
        seal,
    )
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.pipeline_validation.progressive_evaluation import generate, score_velocity
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file

    out = work / "output"
    phase = request["phase"]
    record = dict(version=CONDITIONING_ROLE, request=request["identity"], phase=phase)
    if phase == "models" and set(data) == {"models"} and device == "cpu":
        tensors = {}
        for method, original in data["models"].items():
            recipe = original["recipe"]
            checked_file(work, original["weight"])
            encoder, decoder, _ = load_baseline(
                "retained_D" if method == "D" else "earlier_checkpoint",
                recipe,
                root=work,
                ref=original["weight"],
                reference_recipe=recipe if method == "00" else None,
                device=device,
            )
            tensors[method] = model_snapshot(encoder, decoder)
            del encoder, decoder
        record["tensors"] = tensors
    elif phase == "search" and set(data) == {"prompts", "model"}:
        original = data["model"]
        recipe = original["recipe"]
        record["modes"] = {}
        for mode in request["order"]:
            probe = request["probes"][mode]
            case_us, components = {}, {}
            began = time.monotonic_ns()
            ref = generate(
                work,
                mode,
                data["prompts"],
                recipe,
                checkpoint=original["weight"],
                model_control="retained_D" if original["method"] == "D" else "earlier_checkpoint",
                reference_recipe=recipe if original["method"] == "00" else None,
                device=device,
                output_root=out,
                client=client,
                seconds=21600,
                methods=("correct",),
                registered_probe=probe,
                case_elapsed_us=case_us,
                measurements=components,
            )
            record["modes"][mode] = dict(
                artifact=ref,
                case_us=case_us,
                components=components,
                elapsed_us=(time.monotonic_ns() - began) // 1000,
            )
    elif phase == "score" and set(data) == {"truth"} and device == "cpu":
        record["modes"] = {}
        for mode in request["order"]:
            began = time.monotonic_ns()
            probe, search = request["probes"][mode], request["search"][mode]
            ref = search["artifact"]
            if ref["path"] != "inputs/" + mode + ".json":
                raise ValueError("unexpected candidate mount")
            generated = read_bound_record(work, ref, version="discovery-candidates/v1")
            score_velocity(
                work,
                ref,
                data["truth"],
                client=client,
                output=f"output/{mode}/selected",
                probe=probe,
                seconds=21600,
            )
            selected = reference(out, out / mode / "selected/scores.json")
            selected_us = (time.monotonic_ns() - began) // 1000
            began = time.monotonic_ns()
            scores = read_bound_record(out, selected, version="discovery-scoring/v1")
            bank, _ = paired_bank_projection(
                probe, data["truth"], generated, scores, ref["sha256"], search["case_us"]
            )
            bank_request = seal(dict(pair=request["identity"], mode=mode, data=digest(bank)))
            private = out / mode
            result = score_banks(private, bank_request, bank, client)["artifact"]
            bank_ref = reference(out, private / "output" / result["path"])
            record["modes"][mode] = dict(
                selected=selected,
                bank=bank_ref,
                selected_us=selected_us,
                bank_us=(time.monotonic_ns() - began) // 1000,
            )
    else:
        raise ValueError("unexpected conditioning role inputs")
    path = out / "conditioning/result.json"
    atomic_write(path, seal(record))
    return dict(artifact=reference(out, path))


def conditioning_worker(work, request, data, client):
    if request["phase"] != "search":
        return conditioning_role(work, request, data, client, device="cpu")
    import time
    from oeis_learn.tracking.pipeline_profile import Recorder, device_view
    from oeis_learn.pipeline_validation.progressive_generator_probe import usage_summary
    from oeis_learn.pipeline_validation.progressive_qualification import require_hip

    torch = require_hip()
    nodes = sorted(Path("/dev/dri").glob("renderD*"))
    start = time.monotonic_ns()
    torch.cuda.reset_peak_memory_stats()
    with (
        Recorder(
            work / "output/telemetry",
            instrumented=True,
            render_node=nodes[0] if len(nodes) == 1 else None,
            host_counters=False,
        ) as recorder,
        recorder.phase("conditioning"),
    ):
        result = conditioning_role(work, request, data, client, device="cuda")
        torch.cuda.synchronize()
    elapsed = (time.monotonic_ns() - start) // 1000
    return dict(
        result,
        device="hip",
        device_view=device_view(),
        measured_us=elapsed,
        peak_device_bytes=torch.cuda.max_memory_allocated(),
        telemetry=usage_summary(work / "output/telemetry", elapsed),
    )


def original_bank_rows(
    rows, prompts, truths, gen, scores, parent, profile, case_us, released_cells
):
    """Freeze the original admitted identities and selections before opening a new scoring role."""
    from oeis_learn.pipeline_validation.pilot_clocks import CORRECT_FULL, DESCRIPTIVE
    from oeis_learn.pipeline_validation.progressive_timing_worker import observations

    members = {r["task"]: r["stratum"] for r in rows}
    policy = DESCRIPTIVE if profile == "one" else CORRECT_FULL
    if observations(gen, scores, members, case_us, parent, policy) != released_cells:
        raise ValueError("private choices and scorer clocks differ from the released timing cells")
    return admitted_bank_rows(rows, prompts, truths, gen, scores)


def admitted_bank_rows(rows, prompts, truths, gen, scores):
    from oeis_learn.generalization.progressive.contracts import digest
    from oeis_learn.generalization.progressive.program_search_bank import (
        checked_bank,
        selected_source,
    )

    if gen["prompts"] != digest(sorted(prompts, key=lambda p: p["id"])):
        raise ValueError("historical search used a different visible panel")
    clocks = {c["prompt"]: c for c in gen["clock"]["clocks"]}
    result = []
    for row, prompt, truth in zip(rows, prompts, truths, strict=True):
        key = row["task"]
        if (
            key != prompt["id"]
            or key != truth["id"]
            or any(
                truth[k][:20] != prompt[k] or len(truth[k]) != 100 for k in ("indices", "values")
            )
        ):
            raise ValueError("original scoring recipient differs")
        rec = gen["records"][key]
        score = scores["outcomes"][key]["equal_budget"]
        portfolio = rec["costs"]["correct"]
        envelope = rec["selections"]["equal_budget"]
        if (
            portfolio["equal_budget"] != envelope
            or score["upstream"] != rec["arm_evidence"]["equal_budget"]
        ):
            raise ValueError("original choice differs from its sealed score")
        selected = selected_source(envelope, score, prompt)
        candidates, missing, gap = checked_bank(portfolio, selected)
        details = portfolio["details"]
        if [d["position"] for d in details] != list(range(len(details))):
            raise ValueError("original attempt ordinals differ")
        origins = {}
        for detail in details:
            fitting = detail["fitting"]
            for admission in fitting.get("admission", []):
                if admission["outcome"] == "verified_solution":
                    sha = admission["source_sha256"]
                    completed = fitting.get("clock", {}).get("completed")
                    elapsed = (
                        None
                        if completed is None
                        else microseconds(completed - clocks[key]["start"])
                    )
                    origins.setdefault(sha, {})[detail["position"]] = elapsed
        programs = []
        for candidate in candidates:
            sha = candidate["source_sha256"]
            attempts = origins.get(sha, {})
            first = min(attempts) if attempts else None
            if first is None:
                if gap is None:
                    raise ValueError("eligible source has no original admission")
                continue
            programs.append(
                dict(
                    source=candidate["source"],
                    sha256=sha,
                    ordinals=sorted(attempts),
                    fit_completed_us=attempts.get(first),
                )
            )
        original = score["scoring"]["outcome"]
        verdict = {"success": "solve", "continuation_mismatch": "miss"}.get(original)
        result.append(
            dict(
                id=key,
                stratum=row["stratum"],
                truth=truth,
                programs=programs,
                gap=gap,
                absent_sources=len(missing),
                issued=len(details),
                expected=clocks[key]["expected"],
                search_complete=portfolio["complete"],
                selected=selected["source_sha256"] if selected else None,
                selected_verdict=verdict,
            )
        )
    return dict(rows=result)


def score_banks(work, request, data, client, *, reference=None):
    import time
    from oeis_learn.generalization.progressive.contracts import (
        atomic_write,
        digest,
        reference as artifact,
        seal,
    )
    from oeis_learn.synthesis.admission import source_identity
    from oeis_learn.synthesis.reference import evaluate

    evaluate_reference = reference or evaluate

    def execution(outcome, values):
        values = list(values)
        return dict(
            outcome=outcome, count=len(values), values=digest(values), prefix=digest(values[:20])
        )

    if request["data"] != digest(data):
        raise ValueError("sealed scoring input changed")
    cases = {}
    for row in data["rows"]:
        began = time.monotonic_ns()
        values = {}
        truth = row["truth"]
        if row["id"] in cases or truth["id"] != row["id"] or len(truth["indices"]) != 100:
            raise ValueError("duplicate or incomplete scoring recipient")
        for candidate in row["programs"]:
            sha, source = candidate["sha256"], candidate["source"]
            if sha in values or source_identity(source) != sha:
                raise ValueError("duplicate or substituted admitted source")
            native = client.execute(source, truth["indices"], deadline=time.monotonic() + 30)
            independent = evaluate_reference(
                source, truth["indices"], client.limits, deadline=time.monotonic() + 30
            )
            values[sha] = dict(
                native=execution(native["outcome"], native.get("values", [])),
                reference=execution(independent.outcome, independent.values),
            )
        cases[row["id"]] = dict(programs=values, elapsed_us=(time.monotonic_ns() - began) // 1000)
    out = work / "output/bank/scores.json"
    atomic_write(out, seal(dict(version=BANK_SCORES, request=request["identity"], cases=cases)))
    return dict(artifact=artifact(work / "output", out))


def bank_cost_cells(request, data, scores):
    """Independently reduce two executions on the original truth, with unknown bounds."""
    from oeis_learn.generalization.progressive.cohorts import STRATA
    from oeis_learn.generalization.progressive.contracts import digest, seal

    if (
        request["data"] != digest(data)
        or scores != seal(scores)
        or scores["version"] != BANK_SCORES
        or scores["request"] != request["identity"]
        or len(data["rows"]) != 8
        or set(scores["cases"]) != {r["id"] for r in data["rows"]}
    ):
        raise ValueError("bank scorer evidence differs from its frozen panel")
    fields = (
        "cases",
        "issued",
        "programs",
        "empty",
        "unresolved",
        "any_solve",
        "passed_over",
        "short_rosters",
        "incomplete_search",
        "missing_sources",
        "scoring_us",
        "max_us",
    )
    cells = {s: dict.fromkeys(fields, 0) for s in STRATA}
    for row in data["rows"]:
        case = scores["cases"][row["id"]]
        elapsed = case["elapsed_us"]
        programs = case["programs"]
        if (
            type(elapsed) is not int
            or elapsed < 0
            or programs.keys() != {p["sha256"] for p in row["programs"]}
            or len(programs) != len(row["programs"])
        ):
            raise ValueError("scored program roster or charge differs")
        truth = row["truth"]["values"]
        verdicts = {}
        for sha, result in programs.items():
            verdicts[sha] = None
            pair = [result[k] for k in ("native", "reference")]
            for evidence in pair:
                if (
                    type(evidence["count"]) is not int
                    or not 0 <= evidence["count"] <= 100
                    or any(
                        not re.fullmatch("sha256:[a-f0-9]{64}", evidence[k])
                        for k in ("values", "prefix")
                    )
                ):
                    raise ValueError("invalid execution evidence")
            if all(e["outcome"] == "ok" and e["count"] == 100 for e in pair):
                first, other = pair
                if first != other or first["prefix"] != digest(truth[:20]):
                    raise ValueError(
                        "independent executions or original visible admission disagree"
                    )
                verdicts[sha] = "solve" if first["values"] == digest(truth) else "miss"
        chosen = verdicts.get(row["selected"])
        if chosen and row["selected_verdict"] and chosen != row["selected_verdict"]:
            raise ValueError("original sealed selection outcome differs")
        found = "solve" in verdicts.values()
        c = cells[row["stratum"]]
        c["cases"] += 1
        c["issued"] += row["issued"]
        c["short_rosters"] += int(row["issued"] < row["expected"])
        c["incomplete_search"] += int(not row["search_complete"])
        c["programs"] += len(programs)
        c["empty"] += int(not programs and row["gap"] is None)
        c["unresolved"] += int(row["gap"] is not None or None in verdicts.values())
        c["missing_sources"] += row["absent_sources"]
        c["any_solve"] += int(found)
        c["passed_over"] += int(found and (chosen == "miss" or row["selected"] is None))
        c["scoring_us"] += elapsed
        c["max_us"] = max(c["max_us"], elapsed)
    if any(c["cases"] != 2 for c in cells.values()):
        raise ValueError("two exposed families per stratum required")
    return cells


def bank_sample_forecast(results, controls):
    from oeis_learn.generalization.progressive.program_search_timing import BLOCKS

    sample = dict(
        known_wave=controls,
        observed_sample_us=sum(r["elapsed_us"] for r in results.values()),
        nonempty_program_counts={
            k: sum(c["programs"] for c in r["cells"].values()) for k, r in results.items()
        },
        matched_sample_expected_us=None,
        matched_sample_tail_us=None,
        full_wave_us=None,
        basis="eight matched cases twice per learner, at one and eight attempts, and once for full D; "
        "ordinary prices 00 and first-divergence is the cost proxy for 04/05/06",
    )
    if set(results) != set(BLOCKS):
        return sample
    expected = tail = 0
    overheads = {}
    for name, r in results.items():
        cells = r["cells"].values()
        if any(c["unresolved"] for c in cells):
            return sample
        overheads[name] = r["elapsed_us"] - sum(c["scoring_us"] for c in cells)
        if overheads[name] < 0:
            raise ValueError("bank role shorter than its constituent clocks")
    if not any(c["programs"] for r in results.values() for c in r["cells"].values()):
        return sample
    for name, r in results.items():
        count = 3 if name.startswith("first_divergence-") else 1
        expected += count * r["elapsed_us"]
        tail += count * (
            max(overheads.values()) + sum(c["cases"] * c["max_us"] for c in r["cells"].values())
        )
    return dict(sample, matched_sample_expected_us=expected, matched_sample_tail_us=tail)


def control_role(work, request, data, client):
    """Execute one installed role with only its own input projection."""
    from oeis_learn.generalization.progressive.cohorts import validate_prompt
    from oeis_learn.generalization.progressive.contracts import reference
    from oeis_learn.pipeline_validation.progressive_evaluation import generate, score_velocity

    probe, phase, method = request["probe"], request["phase"], request["method"]
    if (
        method not in CONTROL_METHODS
        or probe["look"]["method"] != method
        or probe["mode"] != "true"
        or probe["look"]["profile"] != "full"
    ):
        raise ValueError("expected the registered conventional full endpoint")
    if phase == "search" and set(data) == {"rows", "recipe"}:
        for prompt in data["rows"]:
            validate_prompt(prompt)
        ref = generate(
            work,
            "generated",
            data["rows"],
            data["recipe"],
            client=client,
            device="cpu",
            seconds=540,
            methods=(method,),
            registered_probe=probe,
            output_root=work / "output",
        )
    elif (
        phase == "score"
        and set(data) == {"rows"}
        and request["candidates"]["path"] == "inputs/candidates.json"
    ):
        score_velocity(
            work,
            request["candidates"],
            data["rows"],
            client=client,
            output="output/scored",
            probe=probe,
            seconds=540,
        )
        ref = reference(work / "output", work / "output/scored/scores.json")
    else:
        raise ValueError("unexpected role input projection")
    return dict(artifact=ref)


def isolated_control_worker(wheel_name, expected_hash, expected_source):
    import time

    work = Path("/work")
    wheel = Path(wheel_name)
    if (
        "sha256:" + hashlib.sha256(wheel.read_bytes()).hexdigest() != expected_hash
        or wheel.parent != work / "inputs"
    ):
        raise ValueError("funded installed wheel differs")
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "pip",
                "--isolated",
                "--disable-pip-version-check",
                "install",
                "--no-index",
                "--no-deps",
                "--no-compile",
                "--no-cache-dir",
                "--target",
                tmp,
                str(wheel),
            ],
            check=True,
        )
        sys.path.insert(0, tmp)
        from oeis_learn.experiments import source_snapshot as provenance
        from oeis_learn.generalization.progressive.contracts import CAMPAIGN_DOCUMENT_BYTES, atomic_write, digest, load, seal
        from oeis_learn.synthesis.client import SynthesisClient

        if (
            not Path(provenance.__file__).resolve().is_relative_to(Path(tmp))
            or digest(provenance.source_snapshot()) != expected_source
        ):
            raise ValueError("installed application differs from its funded source")
        request = load(work / "inputs/request.json")
        data = load(work / "inputs/data.json", maximum=CAMPAIGN_DOCUMENT_BYTES)
        if (
            request != seal(request)
            or request["version"] not in (ISOLATED_CONTROLS, BANK_PRICES, CONDITIONING)
            or request["data"] != digest(data)
            or request["image"] != os.environ["OEIS_QUALIFICATION_IMAGE"]
            or request["source"] != expected_source
            or request["wheel"] != expected_hash
            or os.environ["OEIS_QUALIFICATION_ENGINE"] != "podman"
        ):
            raise ValueError("isolated control request differs from its binding")
        began = time.monotonic_ns()
        with SynthesisClient(digest([request["version"], request["identity"]])) as client:
            native = client.runtime_identity("bigint-body/v1")
            if request["version"] == CONDITIONING:
                result = conditioning_worker(work, request, data, client)
            elif request["version"] == BANK_PRICES and request["phase"] == "bank_score":
                result = score_banks(work, request, data, client)
            else:
                result = control_role(work, request, data, client)
        atomic_write(
            work / "output/producer.json",
            seal(
                dict(
                    request=request["identity"],
                    native=native,
                    worker_us=(time.monotonic_ns() - began) // 1000,
                    **result,
                )
            ),
        )


def control_case_costs(probe, prompts, gen, generation_hash, scores):
    from oeis_learn.generalization.progressive.cohorts import STRATA
    from oeis_learn.generalization.progressive.contracts import digest, seal
    from oeis_learn.generalization.progressive.program_search_bank import selected_source

    look = probe["look"]
    method = look["method"]
    clock = gen["clock"]
    if (
        gen != seal(gen)
        or scores != seal(scores)
        or clock != seal(clock)
        or gen["methods"] != [method]
        or gen["model_control"] != method
        or gen["prompts"] != digest(sorted(prompts, key=lambda p: p["id"]))
        or clock["velocity_probe"] != probe["identity"]
        or clock["policy"] != "round029-conventional-full/v1"
        or clock["inference_seed"] != probe["inference_seed"]
        or not clock["cost_complete"]
        or scores["candidates"] != generation_hash
        or scores["denominator"] != len(prompts)
        or gen["records"].keys() != scores["outcomes"].keys()
        or set(gen["records"]) != {p["id"] for p in prompts}
    ):
        raise ValueError("original conventional search, scorer or clock differs")
    cases = {s: [] for s in STRATA}
    for prompt, member, timer in zip(prompts, look["members"], clock["clocks"], strict=True):
        key = prompt["id"]
        search = timer["elapsed_us"]
        expected = look["attempts"] if method == "template" else 1
        if (
            key != member["task"]
            or timer["prompt"] != key
            or timer["allowed"] != look["attempts"]
            or type(search) is not int
            or search < 0
            or timer["cost_complete"] is not True
            or not 0 <= timer["started"] <= expected
            or timer["unissued"] != look["attempts"] - timer["started"]
            or timer["started"] < expected
            and search < look["opportunity"]["issue_us"]
        ):
            raise ValueError("incomplete conventional portfolio clock")
        record, score = gen["records"][key], scores["outcomes"][key]["equal_budget"]
        choice = selected_source(record["selections"]["equal_budget"], score, prompt)
        scoring = microseconds(score["scoring"].get("elapsed_seconds"))
        if (
            scoring is None
            or score["probe_ref"] != probe["identity"]
            or score["upstream"] != record["arm_evidence"]["equal_budget"]
        ):
            raise ValueError("selected scorer clock or probe parent differs")
        cases[member["stratum"]].append((search, scoring, choice is None))
    result = {}
    for s, rows in cases.items():
        if len(rows) != 2:
            raise ValueError("two exposed cases per stratum required")
        total = sum(a + b for a, b, _ in rows)
        result[s] = dict(
            cases=len(rows),
            empty=sum(e for _, _, e in rows),
            search_us=sum(a for a, _, _ in rows),
            scoring_us=sum(b for _, b, _ in rows),
            mean_us=(total + len(rows) - 1) // len(rows),
            max_us=max(a + b for a, b, _ in rows),
        )
    return result


def control_cost_forecast(results, wave):
    """Estimate one full endpoint for each comparator, keeping retention and tail proxies explicit."""
    from oeis_learn.generalization.progressive.cohorts import STRATA

    if set(results) != set(CONTROL_METHODS):
        return None
    ops = wave["operations"]
    if (
        ops["conventional_cpu_controls"] != 1
        or ops["D.full.search.startup"] != 1
        or "conventional_cpu_controls" not in wave["missing"]
    ):
        raise ValueError("expected one unpriced full endpoint per conventional comparator")
    expected = tail = 0
    cold_tail = sum(
        max(r["overhead_us"][phase] for r in results.values()) for phase in ("search", "score")
    )
    for row in results.values():
        cells = row["cells"]
        expected += sum(row["overhead_us"].values())
        tail += cold_tail
        for s in (*STRATA, "retention"):
            rates = cells.values() if s == "retention" else [cells[s]]
            n = ops.get("D.full.search." + s, 0)
            expected += n * max(c["mean_us"] for c in rates)
            tail += n * max(c["max_us"] for c in rates)
    missing = [k for k in wave["missing"] if k != "conventional_cpu_controls"]
    return dict(
        control_expected_us=expected,
        control_tail_us=tail,
        known_expected_us=wave["known_expected_us"] + expected,
        known_tail_us=wave["known_tail_us"] + tail,
        missing=missing,
        expected_us=None if missing else wave["known_expected_us"] + expected,
        tail_us=None if missing else wave["known_tail_us"] + tail,
        retention_basis="slowest exposed stratum for this method; retention remains sealed",
        tail_basis="slowest observed paired case per stratum and highest cold overhead per role",
    )


def exposed_control_inputs(root):
    from oeis_learn.generalization.progressive.contracts import digest, load, seal
    from oeis_learn.generalization.progressive.program_search_timing import verify_timing
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_round028_audit import SESSION as historical
    from oeis_learn.pipeline_validation.progressive_round028_timing import (
        AREA,
        SHARE,
        original_channels,
        timing_cases,
    )
    from oeis_learn.pipeline_validation.progressive_velocity_qualification import _original_profile

    _, campaign, roster = frozen_control_study(root)
    rows = timing_cases(
        dict(cases=[dict(task=k, stratum=s, group=g) for k, (s, g) in roster.items()])
    )
    archive = confined(root, Path(historical) / SHARE / "round028-timing.zip", exists=False)
    registration = confined(root, Path(historical) / AREA / "registration.json", exists=False)
    if archive.is_file() and registration.is_file():
        paid = verify_timing(archive)
        reg = load(registration)
        if (
            reg != seal(reg)
            or paid["registration"] != reg["identity"]
            or reg["scope"] != "previously_exposed_timing_only"
            or reg["roster"] != rows
        ):
            raise ValueError("the independently verified historical timing roster differs")
        exposure = reg["identity"]
    else:
        # The original closed capability partitions also prove exposure of this entire roster.
        history = read_history(root)
        if history["status"] != "COMPLETE":
            raise ValueError("exposure of the historical timing sample could not be verified")
        exposure = digest([ISOLATED_CONTROLS, campaign["identity"], rows, history["coverage"]])
    refs = {r["name"]: r for r in campaign["payload"]["references"]}
    bound = []
    for member in rows:
        channels = {}
        for channel, role in (("visible", "generator"), ("truth", "scorer")):
            ref = refs[f"private/channels/{channel}-{digest(member['task'])[7:]}"]
            if (ref["kind"], ref["role"]) != ("prepared", role):
                raise ValueError("original channel ownership differs")
            channels[channel] = ref["artifact"]
        bound.append(dict(member, channels=channels))
    prompts, truths = original_channels(root, bound)
    return rows, prompts, truths, _original_profile(root), exposure


def paid_control_runtime(root, source, folder, receipt):
    """Reuse the authenticated checkpoint wheel when its installed control workload matches."""
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.progressive.contracts import load, seal
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation import progressive_velocity_checkpoint_host as checkpoint
    from oeis_learn.pipeline_validation.progressive_velocity_broad_host import attempt
    from oeis_learn.pipeline_validation.progressive_velocity_runtime import (
        package_evidence,
        wheel_sources,
    )
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file

    marker = load(confined(folder, checkpoint.MARKER))
    if marker != seal(marker) or marker["receipt"] != receipt:
        raise ValueError("checkpoint producer differs from the reopened receipt")
    area = confined(folder, "qualification/" + marker["grant"][7:])
    grant, binding = [load(confined(area, n + ".json")) for n in ("grant", "binding")]
    out = attempt(area, checkpoint.coordinates(grant, "package"), grant)
    runtime = load(confined(out, "runtime.json"))
    package_evidence(root, area, runtime, binding)
    wheel = checked_file(area, runtime["wheel"])
    installed = wheel_sources(wheel, binding["source"])
    current = collect_snapshot(source)
    # These host modules only enumerate tests, inspect the donor or price the wave.
    host_only = {
        "src/oeis_learn/pipeline_validation/progressive_cli.py",
        "src/oeis_learn/pipeline_validation/progressive_round029.py",
        "src/oeis_learn/generalization/progressive/program_search_costing.py",
    }
    if any(
        installed.get(n) != current.get(n)
        for n in installed.keys() | current.keys()
        if n not in host_only and not n.startswith("tests/")
    ):
        raise ValueError("the paid wheel contains a different control workload")
    return dict(
        path=wheel,
        wheel=runtime["wheel"]["sha256"],
        source=binding["source"],
        image=runtime["parent"]["image"],
        physical=binding["physical"],
    )


def control_installation_peak(wheel, area):
    from oeis_learn.pipeline_validation.progressive_velocity_launch import wheel_peak

    return wheel_peak(wheel, area)


def control_capacity(root, area, revision, floor, installation, *, conditioning=False):
    from oeis_learn.generalization.progressive.contracts import CAMPAIGN_DOCUMENT_BYTES
    from oeis_learn.pipeline_validation.progressive_research_resources import velocity_capacity

    # One bounded candidate or scoring document is written at a time; prior results already occupy the mount.
    path = str(area.relative_to(root))
    owner = dict(
        owner="paired_conditioning" if conditioning else "isolated_controls",
        path=path,
        scope="paired_diagnostic_role" if conditioning else "cpu_control_role",
        retained=[],
        observed=dict(bytes=0, inodes=0),
        projected=dict(
            bytes=installation["bytes"] + (6 if conditioning else 1) * CAMPAIGN_DOCUMENT_BYTES,
            inodes=installation["inodes"] + (2048 if conditioning else 32),
        ),
        revision=revision,
        harm="incomplete exposed control record",
        exhaustion="preserve completed roles",
        rehearsal="T060",
    )
    mounts = velocity_capacity(
        root, owners=[owner], floors=[dict(path=path, bytes=floor, inodes=0)]
    )["mounts"]
    return [
        select_fields(
            row,
            "device",
            "available_bytes",
            "available_inodes",
            "required_bytes",
            "required_inodes",
            "reserved_bytes",
            "reserved_inodes",
            "deficit_bytes",
            "deficit_inodes",
        )
        for row in mounts
    ]


def conditioning_device_evidence(out, inspection, producer, dispatch_us, render_node):
    """Corroborate the original selected device, host lease and independent role telemetry."""
    from oeis_learn.container_runtime import normalized_image
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_generator_probe import usage_summary
    from oeis_learn.tracking.pipeline_profile import (
        audit_device_view,
        character_device,
        gpu_sensor_path,
    )

    expected = {"/dev/kfd", render_node}
    reported = inspection["HostConfig"].get("Devices") or []
    mapping = (
        isinstance(reported, list)
        and len(reported) == len(expected)
        and all(isinstance(e, dict) for e in reported)
        and {e.get("PathOnHost") for e in reported} == expected
    )
    bad_mapping = bool(reported) and (
        not mapping
        or any(e.get("PathInContainer") not in (None, "", e["PathOnHost"]) for e in reported)
    )
    explicit = mapping and not bad_mapping and all(e.get("PathInContainer") for e in reported)

    def issued_devices(argv):
        # The engine builder terminates its option list with --entrypoint ENTRY IMAGE.
        if not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
            return False
        try:
            end = argv.index("--entrypoint")
            if (
                Path(argv[0]).name != "podman"
                or argv[1] != "run"
                or normalized_image(argv[end + 2]) != normalized_image(inspection["Image"])
            ):
                return False
            devices = []
            for i, arg in enumerate(argv[:end]):
                if arg == "--device" or arg.startswith("--device="):
                    spec = argv[i + 1] if arg == "--device" else arg.split("=", 1)[1]
                    parts = spec.split(":")
                    if (
                        len(parts) > 3
                        or len(parts) > 1
                        and parts[1] != parts[0]
                        or len(parts) == 3
                        and parts[2] != "rwm"
                    ):
                        return False
                    devices.append(parts[0])
                elif arg.split("=", 1)[0] in {"--privileged", "--gpus", "--device-cgroup-rule"}:
                    return False
            return len(devices) == len(expected) and set(devices) == expected
        except (ValueError, IndexError):
            return False

    errors = []
    if bad_mapping:
        errors.append("device_mapping")
    command = confined(out, "command.json", exists=False)
    issued = command.is_file() and issued_devices(load(command).get("argv"))
    created = (inspection.get("Config") or {}).get("CreateCommand")
    if ((command.is_file() or not explicit) and not issued) or (
        created and not issued_devices(created)
    ):
        errors.append("device_argv")
    sensor_matches = None
    if "device_view" in producer:
        try:
            audit_device_view(
                reported, producer["device_view"], {p: character_device(p) for p in expected}
            )
        except (OSError, ValueError):
            errors.append("device_observation")
    elif not explicit:
        # On the original worker this sensor is set only if exactly one render
        # node was visible inside the role. Match its PCI device to the host.
        path = confined(out, "telemetry/capability.json", exists=False)
        sensor = gpu_sensor_path(render_node)
        sensor_matches = bool(
            sensor and path.is_file() and load(path).get("gpu_sensor") == str(sensor)
        )
        if not sensor_matches:
            errors.append("device_observation")
    if producer.get("device") != "hip":
        errors.append("hip_backend")
    lease = load(confined(out, "gpu-lease.json"))["elapsed_us"]
    wall, worker = producer["measured_us"], producer["worker_us"]
    if (
        not all(type(n) is int for n in (wall, worker, lease, dispatch_us))
        or not 0 < wall <= worker <= lease <= dispatch_us
    ):
        errors.append("lease_clock")
    telemetry = usage_summary(confined(out, "telemetry/samples.jsonl").parent, wall)
    if telemetry != producer["telemetry"]:
        errors.append("telemetry_replay")
    if type(wall) is not int or telemetry["covered_us"] > wall:
        errors.append("telemetry_coverage")
    peak = producer["peak_device_bytes"]
    if type(peak) is not int or peak <= 0:
        errors.append("device_memory")
    if errors:
        observations = dict(
            reported_devices=len(reported) if isinstance(reported, list) else None,
            explicit_mapping=bool(explicit),
            launcher_devices_match=bool(issued),
            engine_argv_present=bool(created),
            role_view_present="device_view" in producer,
            sole_render_sensor_matches=sensor_matches,
            samples=telemetry["samples"],
            covered_us=telemetry["covered_us"],
        )
        for name, value in (
            ("measured_us", wall),
            ("worker_us", worker),
            ("lease_us", lease),
            ("dispatch_us", dispatch_us),
            ("peak_device_bytes", peak),
        ):
            observations[name] = value if type(value) is int and value >= 0 else None
        raise ValueError(
            "conditioning evidence: "
            + ",".join(errors)
            + "\nconditioning observations: "
            + json.dumps(observations, sort_keys=True)
        )
    return dict(lease_us=lease, telemetry=telemetry, peak_device_bytes=peak)


def recover_conditioning_stage(area, journal, grant, coords, request, cost, verify, recovery):
    """Reopen an original paid attempt. Only the host reader can run in this path."""
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS, vector
    from oeis_learn.generalization.progressive.contracts import load, seal
    from oeis_learn.generalization.progressive.velocity_records import operation_key
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.preparation_recovery import retain
    from oeis_learn.pipeline_validation.progressive_velocity_operations import (
        check_reserved_work,
        input_record,
        operation_receipt,
        replay_operation,
        run_operation,
    )

    state = BudgetJournal.inspect_existing(
        area / "accounting",
        grant["identity"],
        grant["resources"],
        include_work=True,
        include_windows=True,
    )
    logical = operation_key(coords, 0)["logical"]
    block = confined(area, "operations/" + logical[7:])
    group = request["method"]
    if confined(block, "receipt.json", exists=False).is_file():
        proof = replay_operation(area, coords, request, grant, cost, state, verify)
        if request["phase"] == "search":
            recovery["groups"][group] = dict(
                search=proof["identity"], source="completed", attempt=proof["attempt"]
            )
        return proof, confined(block, f"attempt-{proof['attempt']:04d}")
    if request["phase"] != "search":
        raise ValueError("completed original model receipt required")
    old = load(confined(block, "first-error.json"))
    work = vector(dict(dict.fromkeys(KEYS, 0), **cost))
    number = old["attempt"]
    if (
        old != seal(old)
        or old["stage"] != coords["stage"]
        or old["error"] != "ValueError"
        or type(number) is not int
        or number not in (0, 1)
        or not (
            old["reason"]
            == "ValueError: original HIP devices, lease, samples, or measured clocks differ"
            or old["reason"].startswith("ValueError: conditioning evidence: ")
        )
        or load(confined(block, "input.json")) != input_record(coords, request, work, grant)
    ):
        raise ValueError("original registered verifier rejection required")
    key = f"{logical}/dispatch/{number}"
    check_reserved_work(state, logical + "/dispatch/", work)
    window = state["windows"].get(key, {})
    if (
        key not in state["committed"]
        or key not in state["reserved_work"]
        or window.get("event") != "COMMIT"
        or window["end_us"] is None
        or window["charged_us"] != window["end_us"] - window["start_us"]
        or confined(block, f"boundary-{number:04d}.json", exists=False).exists()
    ):
        raise ValueError("original settled failed dispatch required")
    original = confined(block, f"attempt-{number:04d}")
    check_request = seal(
        dict(
            reader=recovery["reader"],
            original=request["identity"],
            first_error=old["identity"],
            key=key,
            charged_us=window["charged_us"],
        )
    )
    check_coords = dict(coords, role="verify-paid-search")

    def host_reader(out, allowance):
        # No worker or container owns this host-only operation. run_operation
        # separately supervises and terminates its verification subprocess.
        retain(out / "reader.json", check_request)
        retain(out / "cleanup.json", dict(acknowledged=True, within_bound=True))

    def verify_original(out):
        if load(confined(out, "reader.json")) != check_request:
            raise ValueError("paid search reader differs")
        evidence = verify(original)
        if evidence["elapsed_us"] > window["charged_us"]:
            raise ValueError("original dispatch exceeds its committed charge")
        return seal(dict(original=operation_receipt(coords, request, grant, number, evidence)))

    receipt = run_operation(
        area=area,
        journal=journal,
        grant=grant,
        coordinates=check_coords,
        request=check_request,
        cost={},
        execute=host_reader,
        verify=verify_original,
        installed=lambda out: (out / "reader.json").is_file(),
        max_attempts=2,
    )
    proof = receipt["evidence"]["original"]
    recovery["groups"][group] = dict(
        search=proof["identity"],
        source="rechecked",
        attempt=number,
        reader_receipt=receipt["identity"],
        original_charge_us=window["charged_us"],
    )
    return proof, original


def run_control_stage(
    root, source, area, grant, journal, host, runtime, request, data, cost, extra, proof, stage,
    *, recovery=None
):
    import time
    from oeis_learn.generalization.progressive.contracts import (
        CAMPAIGN_DOCUMENT_BYTES,
        load,
        read_bound_record,
        seal,
    )
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.preparation_recovery import retain
    from oeis_learn.pipeline_validation.progressive_round028_receipts import verify_role_mounts
    from oeis_learn.pipeline_validation.progressive_velocity_broad_host import attempt
    from oeis_learn.pipeline_validation.progressive_velocity_operations import (
        replay_operation,
        run_operation,
    )
    from oeis_learn.generalization.progressive.budgets import BudgetJournal

    phase, method = request["phase"], request["method"]
    bank = request["version"] == BANK_PRICES
    paired = request["version"] == CONDITIONING
    gpu = paired and phase == "search"
    coords = dict(
        study=grant["identity"],
        stage="qualification",
        method=method,
        seed=29,
        look=0 if bank else 4_000_000,
        panel="exposed_development",
        profile=method.split("-")[-1] if bank else "full",
        input_mode="true",
        role=phase,
        ordinal=0,
    )
    if paired:
        coords = conditioning_coordinates(grant, method, phase, request["seed"])
    controls = area / "inputs" / method / phase
    rp, dp = controls / "request.json", controls / "data.json"
    reopen = recovery is not None and phase in {"models", "search"}
    if reopen:
        if (load(confined(area, rp)) != request or
                load(confined(area, dp), maximum=CAMPAIGN_DOCUMENT_BYTES) != data):
            raise ValueError("original conditioning input differs")
    else:
        retain(rp, request)
        retain(dp, data)
    worker = source / Path(LAUNCHER).with_name("qualify_continuation.py")
    wheel_dest = "/work/inputs/" + runtime["path"].name
    inputs = [
        (worker, "/work/inputs/control-worker.py"),
        (runtime["path"], wheel_dest),
        (rp, "/work/inputs/request.json"),
        (dp, "/work/inputs/data.json"),
        *extra,
    ]

    def guard():
        mounts = control_capacity(
            root,
            area,
            request["candidate"],
            host.policy["free_disk_floor_bytes"],
            runtime["installation"],
            conditioning=paired,
        )
        if any(m["deficit_bytes"] or m["deficit_inodes"] for m in mounts):
            raise ValueError("insufficient physical space for the registered role")
        if gpu:
            from oeis_learn.pipeline_validation.progressive_round028_rehearsal import physical_scope
            if physical_scope(host, "hip", machine=True) != runtime["physical"] or request["physical"] != runtime["physical"]:
                raise ValueError("current HIP device differs from the paid qualification")

    def execute(out, allowance):
        timeout = allowance // 1_000_000 - 120
        if not paired:
            timeout = min(660, timeout)
        if timeout < 1:
            raise ValueError("insufficient time for a supervised role and cleanup")
        began = time.monotonic_ns()
        stage(
            host,
            source,
            runtime["image"],
            out,
            [
                "python",
                "-I",
                "-B",
                "/work/inputs/control-worker.py",
                "--isolated-control-worker",
                wheel_dest,
                runtime["wheel"],
                runtime["source"],
            ],
            gpu=gpu,
            seconds=timeout,
            readonly_inputs=inputs,
            **({"record_lease": True} if gpu else {}),
        )
        retain(
            out / "dispatch.json",
            seal(
                dict(request=request["identity"], elapsed_us=(time.monotonic_ns() - began) // 1000)
            ),
        )

    def verify(out):
        inspection = load(confined(out, "container-inspect.json"))
        verify_role_mounts(
            inspection,
            runtime["image"],
            {**{k: (p, False) for p, k in inputs}, "/work/output": (out, True)},
        )
        limits = inspection[0]["HostConfig"]
        proc, clean, dispatch, producer = [
            load(confined(out, name + ".json"))
            for name in ("process", "cleanup", "dispatch", "producer")
        ]
        if (
            (limits.get("Devices") and not gpu)
            or limits.get("DeviceRequests")
            or proc["outcome"] != "COMPLETE"
            or proc["returncode"] != 0
            or proc["cleanup_failure"]
            or clean["acknowledged"] is not True
            or clean["within_bound"] is not True
            or dispatch != seal(dispatch)
            or producer != seal(producer)
            or dispatch["request"] != request["identity"]
            or producer["request"] != request["identity"]
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", producer["native"])
            or "expected_native" in request and producer["native"] != request["expected_native"]
            or not 0 < producer["worker_us"] <= dispatch["elapsed_us"]
        ):
            raise ValueError("original role confinement, producer or supervision clock differs")
        device = {}
        if gpu:
            device = conditioning_device_evidence(out, inspection[0], producer, dispatch["elapsed_us"],
                                                  host.policy["render_node"])
        if paired:
            name, version = "conditioning/result.json", CONDITIONING_ROLE
        elif bank:
            name, version = "bank/scores.json", BANK_SCORES
        elif phase == "search":
            name, version = "generated/candidates.json", "discovery-candidates/v1"
        else:
            name, version = "scored/scores.json", "discovery-scoring/v1"
        ref = producer["artifact"]
        if ref["path"] != name or ref["bytes"] > CAMPAIGN_DOCUMENT_BYTES:
            raise ValueError("unexpected isolated control artifact")
        value = read_bound_record(out, ref, version=version)
        return seal(
            dict(
                artifact=ref,
                elapsed_us=dispatch["elapsed_us"],
                worker_us=producer["worker_us"],
                native=producer["native"],
                **device,
                **(proof(value, out) if paired else proof(value)),
            )
        )

    if reopen:
        return recover_conditioning_stage(area, journal, grant, coords, request, cost, verify, recovery)

    result = run_operation(
        area=area,
        grant=grant,
        journal=journal,
        coordinates=coords,
        request=request,
        cost=cost,
        execute=execute,
        verify=verify,
        before_dispatch=guard,
        minimum_us=121_000_000,
        max_attempts=2,
        installed=lambda out: (out / "dispatch.json").is_file(),
    )
    snapshot = BudgetJournal.inspect_existing(
        area / "accounting", grant["identity"], grant["resources"], include_work=True
    )
    if replay_operation(area, coords, request, grant, cost, snapshot, verify) != result:
        raise ValueError("private replay of the published CPU control stage differs")
    return result, attempt(area, coords, grant)


def settled_control_balance(area, journal, grant):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal

    if journal.path.is_file():
        return BudgetJournal.inspect_existing(area / "accounting", grant["identity"], grant["resources"])
    if any(area.glob("operations/*/attempt-*")):
        raise ValueError("an issued preparation operation has no original budget journal")
    return journal.snapshot()


def isolated_control_matrix(
    root, source, folder, revision, minutes, runtime, panel, wave, host, stage
):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, role_work_requirements
    from oeis_learn.generalization.progressive.contracts import (
        digest,
        load,
        read_bound_record,
        seal,
    )
    from oeis_learn.generalization.progressive.graph import velocity_look
    from oeis_learn.generalization.progressive.velocity_records import (
        physical_root_id,
        validate_grant,
    )
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.preparation_recovery import retain
    from oeis_learn.pipeline_validation.progressive_evaluation import velocity_probe
    from oeis_learn.pipeline_validation.progressive_round028_rehearsal import physical_scope
    from oeis_learn.pipeline_validation.progressive_velocity_qualification import (
        authorize_qualification,
    )

    rows, prompts, truth, recipe, timing = panel
    machine = physical_scope(host, "cpu", machine=True)
    plan = seal(
        dict(
            version=ISOLATED_CONTROLS,
            candidate=revision,
            checkpoint=wave["checkpoint_receipt"],
            timing=timing,
            roster=digest(rows),
            visible=digest(prompts),
            truth=digest(truth),
            recipe=digest(recipe),
            machine=machine,
            wheel=runtime["wheel"],
            source=runtime["source"],
            image=runtime["image"],
            methods=list(CONTROL_METHODS),
            mode="exposed_costs_only",
        )
    )
    phases = {}
    for phase, op, kind in (
        ("search", "generate", "visible_prompt"),
        ("score", "score", "development_truth"),
    ):
        inputs = [dict(kind=kind) for _ in rows]
        if phase == "search":
            inputs.append(dict(kind="oeis_generation"))
        cost = role_work_requirements(op, inputs)
        phases[phase] = {k: v for k, v in cost.items() if k != "elapsed_us" and v}
    limits = {
        k: 2 * len(CONTROL_METHODS) * sum(c.get(k, 0) for c in phases.values())
        for k in set().union(*phases.values())
    }
    grant = load(authorize_qualification(root, int(minutes), plan["identity"], limits))
    approval = load(
        confined(
            root,
            "run-state/discovery-authorizations/" + grant["authorizing_record_ref"][7:] + ".json",
        )
    )
    validate_grant(
        grant, approval, root=physical_root_id(root), parent=plan["identity"], action="qualify"
    )
    area = confined(
        folder, "preparation/isolated-control-prices/" + grant["identity"][7:], exists=False
    )
    retain(area / "plan.json", plan)
    retain(area / "grant.json", grant)
    runtime = dict(runtime, installation=control_installation_peak(runtime["path"], area))
    journal = BudgetJournal(area / "accounting", grant["identity"], grant["resources"])
    files, results, errors = {}, {}, {}
    report = dict(
        candidate=revision,
        scope=ISOLATED_CONTROLS,
        grant=grant["identity"],
        grant_minutes=int(minutes),
        gpu_dispatches=0,
        research_run_requested=False,
        scientific_verdict=None,
        files=files,
        errors=errors,
    )
    files["capacity.json"] = dict(
        mounts=control_capacity(
            root, area, revision, host.policy["free_disk_floor_bytes"], runtime["installation"]
        ),
        projected_installation=runtime["installation"],
    )
    files["forecast.json"] = dict(supported_wave=wave, measured_control_projection=None)
    for method in CONTROL_METHODS:
        phase = "search"
        try:
            look = velocity_look(
                registration=plan["identity"],
                method=method,
                seed=29,
                look=4_000_000,
                profile="full",
                development=rows,
                retention=[],
            )
            probe = velocity_probe(look, prompts, mode="true", inference_seed=57)
            base = dict(
                version=ISOLATED_CONTROLS,
                candidate=revision,
                method=method,
                probe=probe,
                plan=plan["identity"],
                source=runtime["source"],
                image=runtime["image"],
                wheel=runtime["wheel"],
            )
            data = dict(rows=prompts, recipe=recipe)

            def search_proof(gen):
                if (
                    gen["methods"] != [method]
                    or gen["recipe"] != digest(recipe)
                    or gen["model_control"] != method
                    or gen["clock"]["velocity_probe"] != probe["identity"]
                    or gen["prompts"] != digest(sorted(prompts, key=lambda p: p["id"]))
                ):
                    raise ValueError("original search probe or recipe differs")
                return {}

            search, search_out = run_control_stage(
                root,
                source,
                area,
                grant,
                journal,
                host,
                runtime,
                seal(dict(base, phase=phase, data=digest(data))),
                data,
                phases[phase],
                [],
                search_proof,
                stage,
            )
            first = search["evidence"]
            files[method + "-search.json"] = dict(
                receipt=search["identity"],
                **select_fields(first, "elapsed_us", "worker_us", "native"),
            )
            phase = "score"
            gen = read_bound_record(
                search_out, first["artifact"], version="discovery-candidates/v1"
            )
            ref = dict(first["artifact"], path="inputs/candidates.json")
            data = dict(rows=truth)

            def score_proof(scores):
                return dict(cells=control_case_costs(probe, prompts, gen, ref["sha256"], scores))

            scoring, _ = run_control_stage(
                root,
                source,
                area,
                grant,
                journal,
                host,
                runtime,
                seal(
                    dict(
                        base,
                        phase=phase,
                        data=digest(data),
                        candidates=ref,
                        search=search["identity"],
                    )
                ),
                data,
                phases[phase],
                [(confined(search_out, first["artifact"]["path"]), "/work/inputs/candidates.json")],
                score_proof,
                stage,
            )
            last = scoring["evidence"]
            cells = last["cells"]
            overhead = {
                k: elapsed - sum(c[name] for c in cells.values())
                for k, elapsed, name in (
                    ("search", first["elapsed_us"], "search_us"),
                    ("score", last["elapsed_us"], "scoring_us"),
                )
            }
            if any(v < 0 for v in overhead.values()) or first["native"] != last["native"]:
                raise ValueError("supervised role clocks or native implementation differ")
            row = seal(
                dict(
                    method=method,
                    search=search["identity"],
                    score=scoring["identity"],
                    cells=cells,
                    overhead_us=overhead,
                    search_elapsed_us=first["elapsed_us"],
                    score_elapsed_us=last["elapsed_us"],
                )
            )
            retain(area / (method + ".json"), row)
            results[method] = files["control-" + method + ".json"] = row
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
            KeyboardInterrupt,
        ) as exc:
            errors[method] = dict(phase=phase, error=type(exc).__name__)
            print(f"{method} {phase}: {exc}", file=sys.stderr)
            if isinstance(exc, KeyboardInterrupt):
                break
    balance = settled_control_balance(area, journal, grant)
    if (
        balance["pending"]
        or balance["consumed"]["elapsed_us"] + balance["remaining"]["elapsed_us"]
        != grant["total_us"]
    ):
        raise ValueError("isolated control debits differ from the original grant")
    report.update(
        status="PARTIAL" if errors else "COMPLETE",
        elapsed_us=balance["consumed"]["elapsed_us"],
        remaining_us=balance["remaining"]["elapsed_us"],
        charged_work=balance["consumed"],
    )
    files["verification.json"] = dict(
        scope="private_exposed_cpu_control_costs",
        research_dispatches=0,
        verified={m: dict(search=row["search"], score=row["score"]) for m, row in results.items()},
        settled_elapsed_us=balance["consumed"]["elapsed_us"],
        remaining_us=balance["remaining"]["elapsed_us"],
    )
    files["forecast.json"]["measured_control_projection"] = control_cost_forecast(results, wave)
    if not errors:
        retain(
            area / "receipt.json",
            seal(
                dict(
                    plan=plan["identity"],
                    grant=grant["identity"],
                    controls={m: r["identity"] for m, r in results.items()},
                    charged=balance["consumed"],
                )
            ),
        )
        report["receipt"] = load(area / "receipt.json")["identity"]
    return report


def paid_bank_boundary(root, rows):
    """Anchor the private file index to the independently published round028 workstation."""
    import io
    from oeis_learn.generalization.progressive.campaign import velocity_policy
    from oeis_learn.generalization.progressive.contracts import load, read_bound_record, seal
    from oeis_learn.generalization.progressive.program_search_timing import BLOCKS, PANEL
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_round028_audit import SESSION as historical
    from oeis_learn.pipeline_validation.progressive_round028_pilot import PARENT_WORKSTATION
    from oeis_learn.pipeline_validation.progressive_round028_timing import AREA
    from oeis_learn.pipeline_validation.progressive_velocity_donor import _archives

    paths, _ = _archives(root, velocity_policy())
    with zipfile.ZipFile(paths["pilot"]) as pilot:
        with zipfile.ZipFile(io.BytesIO(pilot.read("round028-parent.zip"))) as parent:
            workstation = json.loads(parent.read("workstation.json"))
    folder = confined(root, historical)
    area = confined(folder, AREA)
    closure = load(confined(folder, "private/terminal/closure.json"))
    replay = closure["replay"]
    cut = load(confined(area, "boundary.json"))
    if (
        workstation != seal(workstation)
        or workstation["identity"] != PARENT_WORKSTATION
        or closure != seal(closure)
        or replay != seal(replay)
        or workstation["private_replay"] != replay["identity"]
        or replay["timing_boundary"] != cut["identity"]
        or cut != seal(cut)
        or cut["panel"] != PANEL
        or replay["portfolios"] != 40
    ):
        raise ValueError(
            "historical bank file index differs from the independently verified terminal receipt"
        )
    reg = read_bound_record(area, cut["registration"], identity=replay["registration"])
    if (
        reg["roster"] != rows
        or reg["blocks"] != list(BLOCKS)
        or reg["panel"] != PANEL
        or reg["scope"] != "previously_exposed_timing_only"
        or reg["attempts"] != {"one": 1, "full": 8}
    ):
        raise ValueError("historical timing roster or attempt policy differs")
    index = {f["path"]: f for f in cut["files"]}
    if len(index) != len(cut["files"]):
        raise ValueError("repeated path in the historical timing closure")
    public = read_bound_record(area, cut["public"], identity=replay["timing"])
    if public["registration"] != reg["identity"]:
        raise ValueError("released clocks belong to another timing registration")
    return area, reg["identity"], cut["identity"], index, public["blocks"]


def paid_bank_inputs(area, registration, index, name, rows, prompts, truths, released):
    """All opened files must be exact members of the previously committed role chain."""
    from oeis_learn.generalization.progressive.contracts import read_bound_record
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_round028_receipts import sealed

    results, refs = {}, {}
    case_us = None
    for role, label, expected, version, channel in (
        ("search", "candidates", "generation/candidates.json", "discovery-candidates/v1", prompts),
        ("score", "scores", "scored/scores.json", "discovery-scoring/v1", truths),
    ):
        # The assignment is only a locator. Every projected byte is authenticated by the original cut.
        assignment = sealed(confined(area, "assignments/" + name + "-" + role + ".json"))
        folder = Path("roles", assignment["request"][7:])

        def at(path, **kwargs):
            return read_bound_record(area, index[str(folder / path)], **kwargs)

        request = at("request.json", identity=assignment["request"])
        data = at("data.json", identity=request["data"])
        boundary = at("boundary.json")
        if (
            request["registration"] != registration
            or request["role"] != role
            or data["block"] != name
            or data["rows"] != channel
            or boundary["request"] != request["identity"]
        ):
            raise ValueError("timing block, assignment, or frozen recipient differs")
        producer_path = Path(boundary["producer"]["path"])
        producer = at(producer_path, version="round028-timing-output/v1")
        ref = producer["result"][label]
        rel = folder / producer_path.parent / expected
        if (
            producer["request"] != request["identity"]
            or ref["path"] != expected
            or dict(ref, path=str(rel)) != index[str(rel)]
        ):
            raise ValueError("saved worker output differs from its private file reference")
        if role == "score" and data["candidates"]["sha256"] != refs["search"]:
            raise ValueError("original scorer did not mount this exact choice")
        results[role] = read_bound_record(area, index[str(rel)], version=version)
        refs[role] = ref["sha256"]
        if role == "search":
            case_us = producer["result"]["case_us"]
    return original_bank_rows(
        rows,
        prompts,
        truths,
        results["search"],
        results["score"],
        refs["search"],
        name.split("-")[-1],
        case_us,
        released["cells"],
    ), refs


def reopen_control_costs(root, folder, receipt_id, wave, panel, runtime, machine):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.contracts import digest, load, seal
    from oeis_learn.generalization.progressive.velocity_records import (
        operation_key,
        physical_root_id,
        validate_grant,
    )
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_velocity_operations import operation_receipt

    def saved(base, rel):
        obj = load(confined(base, rel))
        if obj != seal(obj):
            raise ValueError("original CPU control receipt is not sealed")
        return obj

    if not re.fullmatch("sha256:[a-f0-9]{64}", receipt_id):
        raise ValueError("full original isolated control receipt required")
    parent = confined(folder, "preparation/isolated-control-prices")
    matches = []
    for path in parent.glob("*/receipt.json"):
        rec = saved(parent, path)
        if rec["identity"] == receipt_id:
            matches.append((path.parent, rec))
    if len(matches) != 1:
        raise ValueError("original isolated control receipt missing or duplicated")
    area, receipt = matches[0]
    plan, grant = (saved(area, n + ".json") for n in ("plan", "grant"))
    auth = saved(
        root, "run-state/discovery-authorizations/" + grant["authorizing_record_ref"][7:] + ".json"
    )
    validate_grant(
        grant, auth, root=physical_root_id(root), parent=plan["identity"], action="qualify"
    )
    rows, visible, truth, recipe, timing = panel
    binding = dict(
        version=ISOLATED_CONTROLS,
        checkpoint=wave["checkpoint_receipt"],
        roster=digest(rows),
        visible=digest(visible),
        truth=digest(truth),
        recipe=digest(recipe),
        timing=timing,
        machine=machine,
        methods=list(CONTROL_METHODS),
        mode="exposed_costs_only",
        **select_fields(runtime, "wheel", "source", "image"),
    )
    snapshot = BudgetJournal.inspect_existing(
        area / "accounting", grant["identity"], grant["resources"]
    )
    if (
        any(plan[k] != v for k, v in binding.items())
        or receipt["plan"] != plan["identity"]
        or receipt["grant"] != grant["identity"]
        or area.name != grant["identity"][7:]
        or snapshot["pending"]
        or receipt["charged"] != snapshot["consumed"]
        or set(receipt["controls"]) != set(CONTROL_METHODS)
    ):
        raise ValueError("closed original control grant or current pricing operands differ")
    results = {}
    for method in CONTROL_METHODS:
        row = saved(area, method + ".json")
        if row["identity"] != receipt["controls"][method] or row["method"] != method:
            raise ValueError("different original control costs")
        for phase in ("search", "score"):
            coords = dict(
                study=grant["identity"],
                stage="qualification",
                method=method,
                seed=29,
                look=4_000_000,
                panel="exposed_development",
                profile="full",
                input_mode="true",
                role=phase,
                ordinal=0,
            )
            logical = operation_key(coords, 0)["logical"]
            op = saved(area, "operations/" + logical[7:] + "/receipt.json")
            req = saved(area, f"inputs/{method}/{phase}/request.json")
            if (
                op["identity"] != row[phase]
                or req["plan"] != plan["identity"]
                or op != operation_receipt(coords, req, grant, op["attempt"], op["evidence"])
                or f"{logical}/dispatch/{op['attempt']}" not in snapshot["committed"]
                or op["evidence"]["elapsed_us"] != row[phase + "_elapsed_us"]
            ):
                raise ValueError("previously paid role receipt or debit differs")
        results[method] = row
    return control_cost_forecast(results, wave)


def bank_price_matrix(
    root, source, folder, revision, minutes, runtime, panel, wave, host, stage, previous
):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.contracts import digest, load, seal
    from oeis_learn.generalization.progressive.program_search_timing import BLOCKS
    from oeis_learn.generalization.progressive.velocity_records import (
        physical_root_id,
        validate_grant,
    )
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.preparation_recovery import retain
    from oeis_learn.pipeline_validation.progressive_round028_rehearsal import physical_scope
    from oeis_learn.pipeline_validation.progressive_velocity_qualification import (
        authorize_qualification,
    )

    rows, prompts, truths, _, _ = panel
    machine = physical_scope(host, "cpu", machine=True)
    results, errors, batches, files = {}, {}, {}, {}
    report = dict(
        candidate=revision,
        scope=BANK_PRICES,
        grant_minutes=int(minutes),
        grant=None,
        elapsed_us=0,
        files=files,
        errors=errors,
        gpu_dispatches=0,
        research_run_requested=False,
        scientific_verdict=None,
    )
    step = "paid-controls"
    try:
        controls = reopen_control_costs(root, folder, previous, wave, panel, runtime, machine)
        files["paid-controls.json"] = dict(receipt=previous, projection=controls)
        files["forecast.json"] = bank_sample_forecast(results, controls)
        step = "original-bank-index"
        original, registration, cut, index, released = paid_bank_boundary(root, rows)
    except (KeyError, ValueError, TypeError, OSError) as exc:
        print(f"{step}: {exc}", file=sys.stderr)
        return dict(report, status="BLOCKED", stage=step, error=type(exc).__name__)
    plan = seal(
        dict(
            version=BANK_PRICES,
            candidate=revision,
            original=cut,
            timing=registration,
            blocks=list(BLOCKS),
            controls=previous,
            checkpoint=wave["checkpoint_receipt"],
            visible=digest(prompts),
            truth=digest(truths),
            machine=machine,
            **select_fields(runtime, "wheel", "source", "image"),
        )
    )
    for name in BLOCKS:
        try:
            batches[name] = paid_bank_inputs(
                original, registration, index, name, rows, prompts, truths, released[name]
            )
        except (KeyError, ValueError, TypeError, OSError) as exc:
            errors[name] = dict(phase="original-bank", error=type(exc).__name__)
            print(f"{name} original bank: {exc}", file=sys.stderr)
    if not batches:
        files["forecast.json"] = bank_sample_forecast(results, controls)
        return dict(report, status="BLOCKED", stage="original-banks")
    # The historical archive bound is 64 distinct source identities per case. Debit only the
    # actual frozen number of native requests for an issued block. One retry is bounded separately.
    grant = load(
        authorize_qualification(
            root, int(minutes), plan["identity"], dict(native_work=2 * len(BLOCKS) * len(rows) * 64)
        )
    )
    approval = load(
        confined(
            root,
            "run-state/discovery-authorizations/" + grant["authorizing_record_ref"][7:] + ".json",
        )
    )
    validate_grant(
        grant, approval, root=physical_root_id(root), parent=plan["identity"], action="qualify"
    )
    area = confined(
        folder, "preparation/matched-bank-prices/" + grant["identity"][7:], exists=False
    )
    retain(area / "plan.json", plan)
    retain(area / "grant.json", grant)
    journal = BudgetJournal(area / "accounting", grant["identity"], grant["resources"])
    runtime = dict(runtime, installation=control_installation_peak(runtime["path"], area))
    files["capacity.json"] = dict(
        mounts=control_capacity(
            root, area, revision, host.policy["free_disk_floor_bytes"], runtime["installation"]
        )
    )
    for name, (data, refs) in batches.items():
        request = seal(
            dict(
                version=BANK_PRICES,
                candidate=revision,
                method=name,
                phase="bank_score",
                plan=plan["identity"],
                parents=refs,
                data=digest(data),
                **select_fields(runtime, "wheel", "source", "image"),
            )
        )
        cost = dict(native_work=sum(len(r["programs"]) for r in data["rows"]))
        try:
            paid, _ = run_control_stage(
                root,
                source,
                area,
                grant,
                journal,
                host,
                runtime,
                request,
                data,
                cost,
                [],
                lambda value: dict(cells=bank_cost_cells(request, data, value)),
                stage,
            )
            evidence = paid["evidence"]
            result = seal(
                dict(
                    block=name,
                    receipt=paid["identity"],
                    **select_fields(evidence, "elapsed_us", "worker_us", "native", "cells"),
                )
            )
            retain(area / (name + ".json"), result)
            results[name] = files[name + ".json"] = result
        except (
            OSError,
            ValueError,
            RuntimeError,
            KeyError,
            TypeError,
            subprocess.SubprocessError,
            KeyboardInterrupt,
        ) as exc:
            errors[name] = dict(phase="bank-score", error=type(exc).__name__)
            print(f"{name} bank scorer: {exc}", file=sys.stderr)
            if isinstance(exc, KeyboardInterrupt):
                break
    balance = settled_control_balance(area, journal, grant)
    if (
        balance["pending"]
        or balance["consumed"]["elapsed_us"] + balance["remaining"]["elapsed_us"]
        != grant["total_us"]
    ):
        raise ValueError("bank cost or remaining authority differs from the original grant")
    files["forecast.json"] = bank_sample_forecast(results, controls)
    files["verification.json"] = dict(
        scope="original_admissions_on_exposed_timing_panel",
        original=cut,
        research_dispatches=0,
        verified={k: r["receipt"] for k, r in results.items()},
    )
    report.update(
        status="COMPLETE" if len(results) == len(BLOCKS) else "PARTIAL",
        grant=grant["identity"],
        elapsed_us=balance["consumed"]["elapsed_us"],
        remaining_us=balance["remaining"]["elapsed_us"],
        charged_work=balance["consumed"],
    )
    if report["status"] == "COMPLETE":
        receipt = seal(
            dict(
                plan=plan["identity"],
                grant=grant["identity"],
                charged=balance["consumed"],
                blocks={k: v["identity"] for k, v in results.items()},
            )
        )
        retain(area / "receipt.json", receipt)
        report["receipt"] = receipt["identity"]
    return report


def previous_bank_prices(root, folder, previous, wave, panel, runtime, host):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.contracts import digest, load, seal
    from oeis_learn.generalization.progressive.program_search_timing import BLOCKS
    from oeis_learn.generalization.progressive.velocity_records import (
        operation_key,
        physical_root_id,
        validate_grant,
    )
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_round028_rehearsal import physical_scope

    parent = confined(folder, "preparation/matched-bank-prices")
    records = []
    for candidate in parent.glob("*/receipt.json"):
        rec = load(confined(parent, candidate))
        if rec == seal(rec) and rec["identity"] == previous:
            records.append((candidate.parent, rec))
    if len(records) != 1:
        raise ValueError("exact completed bank receipt required on the donor")
    area, receipt = records[0]
    grant, plan = [load(confined(area, n + ".json")) for n in ("grant", "plan")]
    auth = load(
        confined(
            root,
            "run-state/discovery-authorizations/" + grant["authorizing_record_ref"][7:] + ".json",
        )
    )
    validate_grant(
        grant, auth, root=physical_root_id(root), parent=plan["identity"], action="qualify"
    )
    machine = physical_scope(host, "cpu", machine=True)
    if (
        grant != seal(grant)
        or plan != seal(plan)
        or plan["version"] != BANK_PRICES
        or receipt["grant"] != grant["identity"]
        or area.name != grant["identity"][7:]
        or receipt["plan"] != plan["identity"]
        or plan["machine"] != machine
        or set(receipt["blocks"]) != set(BLOCKS)
        or plan["blocks"] != list(BLOCKS)
        or plan["visible"] != digest(panel[1])
        or plan["truth"] != digest(panel[2])
        or plan["checkpoint"] != wave["checkpoint_receipt"]
        or any(plan[k] != runtime[k] for k in ("wheel", "source", "image"))
    ):
        raise ValueError("bank receipt, registered runtime or historical panel differs")
    state = BudgetJournal.inspect_existing(
        area / "accounting", grant["identity"], grant["resources"]
    )
    if state["pending"] or state["consumed"] != receipt["charged"]:
        raise ValueError("original bank charges differ")
    blocks = {}
    for group in BLOCKS:
        row = load(confined(area, group + ".json"))
        coords = dict(
            study=grant["identity"],
            stage="qualification",
            method=group,
            seed=29,
            look=0,
            panel="exposed_development",
            profile=group.split("-")[-1],
            input_mode="true",
            role="bank_score",
            ordinal=0,
        )
        logical = operation_key(coords, 0)["logical"]
        op = load(confined(area, "operations/" + logical[7:] + "/receipt.json"))
        if (
            row != seal(row)
            or row["identity"] != receipt["blocks"][group]
            or row["block"] != group
            or op != seal(op)
            or op["identity"] != row["receipt"]
            or op["coordinates"] != coords
            or op["evidence"]["cells"] != row["cells"]
            or op["evidence"]["elapsed_us"] != row["elapsed_us"]
            or f"{logical}/dispatch/{op['attempt']}" not in state["committed"]
        ):
            raise ValueError("paid bank group or original debit differs")
        blocks[group] = row
    controls = reopen_control_costs(root, folder, plan["controls"], wave, panel, runtime, machine)
    return dict(
        receipt=previous,
        controls=plan["controls"],
        projection=bank_sample_forecast(blocks, controls),
    )


def conditioning_panel(root, panel):
    """Keep the exposed timing families whenever a within-stratum differing donor is available."""
    from oeis_learn.generalization.progressive.cohorts import STRATA, validate_prompt
    from oeis_learn.generalization.progressive.contracts import digest, read_bound_record
    from oeis_learn.pipeline_validation.progressive_round028_timing import original_channels

    rows, visible, truth = panel[:3]

    def differs(a, b):
        return a["group"] != b["group"] and by_id[a["task"]]["values"] != by_id[b["task"]]["values"]

    by_id = {p["id"]: p for p in visible}
    pairs = {s: [r for r in rows if r["stratum"] == s] for s in STRATA}
    if all(len(rr) == 2 and differs(*rr) for rr in pairs.values()):
        return rows, visible, truth
    # The capability closure proves every possible replacement was previously exposed.
    if read_history(root)["status"] != "COMPLETE":
        raise ValueError("complete original exposure required for a substitute conditioning donor")
    _, campaign, roster = frozen_control_study(root)
    refs = {r["name"]: r for r in campaign["payload"]["references"]}

    def channel(row, kind):
        ref = refs[f"private/channels/{kind}-{digest(row['task'])[7:]}"]
        role = "generator" if kind == "visible" else "scorer"
        if (ref["kind"], ref["role"]) != ("prepared", role):
            raise ValueError("historical channel owner differs")
        return ref["artifact"]

    used = {r["group"] for r in rows}
    for s, pair in pairs.items():
        if len(pair) != 2:
            raise ValueError("two original exposed families per stratum required")
        if differs(*pair):
            continue
        candidates = [
            dict(task=k, stratum=s, group=g)
            for k, (stratum, g) in roster.items()
            if stratum == s and g not in used
        ]
        for alternate in sorted(
            candidates, key=lambda r: digest([CONDITIONING, s, r["group"], r["task"]])
        ):
            p = read_bound_record(
                root, channel(alternate, "visible"), version="discovery-channel/v1"
            )
            if (
                p["role"] != "generator"
                or p["kind"] != "visible_prompt"
                or p["data"]["id"] != alternate["task"]
            ):
                raise ValueError("original substitute visible prompt differs")
            by_id[alternate["task"]] = validate_prompt(p["data"])
            if differs(pair[0], alternate):
                used.add(alternate["group"])
                pair[1] = alternate
                break
        else:
            raise ValueError("no eligible differing donor in exposed stratum " + s)
    selected = [r for s in STRATA for r in pairs[s]]
    bound = [dict(r, channels={k: channel(r, k) for k in ("visible", "truth")}) for r in selected]
    prompts, targets = original_channels(root, bound)
    return selected, prompts, targets


def conditioning_matrix(
    root, source, folder, revision, minutes, runtime, panel, wave, cuts, host, stage, previous,
    *, recovery=None
):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
    from oeis_learn.generalization.progressive.contracts import (
        digest,
        load,
        read_bound_record,
        seal,
    )
    from oeis_learn.generalization.progressive.velocity_records import (
        operation_key,
        physical_root_id,
        validate_grant,
    )
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.preparation_recovery import retain
    from oeis_learn.pipeline_validation.progressive_round028_rehearsal import physical_scope
    from oeis_learn.pipeline_validation.progressive_velocity_qualification import (
        authorize_qualification,
    )

    files, errors, searched, finished = {}, {}, {}, {}
    report = dict(
        candidate=revision,
        scope=CONDITIONING,
        grant_minutes=int(minutes),
        grant=None,
        elapsed_us=0,
        gpu_dispatches=0,
        research_run_requested=False,
        scientific_verdict=None,
        files=files,
        errors=errors,
    )
    step = "prior-bank-receipt"
    try:
        files["previous-prices.json"] = previous_bank_prices(
            root, folder, previous, wave, panel, runtime, host
        )
        step = "differing-visible-donors"
        rows, prompts, truth = conditioning_panel(root, panel)
        step = "donated-models"
        models, paths = paired_model_inputs(root, cuts)
        step = "physical-device"
        if physical_scope(host, "hip", machine=True) != runtime["physical"]:
            raise ValueError("qualified HIP device differs")
    except (KeyError, TypeError, OSError, ValueError, RuntimeError) as exc:
        print(f"{step}: {exc}", file=sys.stderr)
        return dict(report, status="BLOCKED", stage=step, error=type(exc).__name__)
    seed = models["00"]["seed"]
    order = {
        group: (["shuffled", "true"] if group == "ordinary-full" else ["true", "shuffled"])
        for group in CONDITIONING_GROUPS
    }
    plan = seal(
        dict(
            version=CONDITIONING,
            candidate=revision,
            previous=previous,
            checkpoint=wave["checkpoint_receipt"],
            rows=digest(rows),
            visible=digest(prompts),
            truth=digest(truth),
            models=digest(models),
            order=order,
            inference_seed=57,
            seed=seed,
            **select_fields(runtime, "source", "image", "wheel", "physical"),
        )
    )
    work = {}
    for group in CONDITIONING_GROUPS:
        for phase in ("search", "score"):
            bound = conditioning_work(group, phase, len(rows), 2 * len(rows) * 64)
            for k, v in bound.items():
                work[k] = work.get(k, 0) + 2 * v
    if recovery is None:
        grant = load(authorize_qualification(root, int(minutes), plan["identity"], work))
    else:
        try:
            ref = recovery["grant"]
            if not re.fullmatch(r"sha256:[0-9a-f]{64}", ref):
                raise ValueError("original conditioning grant required")
            area = confined(folder, "preparation/paired-conditioning/" + ref[7:])
            grant = load(confined(area, "grant.json"))
            if (
                grant["identity"] != ref
                or grant != seal(grant)
                or load(confined(area, "plan.json")) != plan
                or load(
                    confined(root, "run-state/discovery-authorizations/grant-" + ref[7:] + ".json")
                )
                != grant
                or grant["scope"] != "qualification"
                or grant["total_us"] != int(minutes) * 60_000_000
                or grant["resources"]
                != dict(dict.fromkeys(KEYS, 0), elapsed_us=grant["total_us"], **work)
            ):
                raise ValueError("original conditioning grant, work or plan differs")
            initial = BudgetJournal.inspect_existing(area / "accounting", ref, grant["resources"])
            hip_prefixes = tuple(
                operation_key(conditioning_coordinates(grant, g, "search", seed), 0)["logical"]
                + "/dispatch/"
                for g in CONDITIONING_GROUPS
            )
            if any(k.startswith(hip_prefixes) for k in initial["pending"]):
                raise ValueError("original HIP dispatch has not settled")
            recovery = files["recovery.json"] = dict(
                grant=ref,
                reader=recovery["reader"],
                worker_candidate=revision,
                original_hip_attempts=sum(k.startswith(hip_prefixes) for k in initial["committed"]),
                additional_hip_attempts=0,
                groups={},
                failure_checks={},
                observations={},
            )
        except (ValueError, OSError, KeyError, TypeError) as exc:
            print(f"original grant: {exc}", file=sys.stderr)
            return dict(report, status="BLOCKED", stage="original-grant", error=type(exc).__name__)
    auth = load(
        confined(
            root,
            "run-state/discovery-authorizations/" + grant["authorizing_record_ref"][7:] + ".json",
        )
    )
    validate_grant(
        grant, auth, root=physical_root_id(root), parent=plan["identity"], action="qualify"
    )
    area = confined(
        folder, "preparation/paired-conditioning/" + grant["identity"][7:], exists=False
    )
    retain(area / "plan.json", plan)
    retain(area / "grant.json", grant)
    journal = BudgetJournal(area / "accounting", grant["identity"], grant["resources"])
    runtime = dict(runtime, installation=control_installation_peak(runtime["path"], area))
    files["capacity.json"] = dict(
        mounts=control_capacity(
            root,
            area,
            revision,
            host.policy["free_disk_floor_bytes"],
            runtime["installation"],
            conditioning=True,
        )
    )
    files["panel.json"] = dict(
        cases=len(rows),
        substituted=sum(r not in panel[0] for r in rows),
        strata=4,
        attempts={"common": 1, "full": 8},
        order=order,
    )

    def operate(group, phase, data, extra, proof, additional, bank_count=0):
        req = seal(
            dict(
                version=CONDITIONING,
                candidate=revision,
                method=group,
                phase=phase,
                seed=seed,
                plan=plan["identity"],
                data=digest(data),
                **select_fields(runtime, "wheel", "source", "image", "physical"),
                **additional,
            )
        )

        def check(value, out):
            if value["request"] != req["identity"] or value["phase"] != phase:
                raise ValueError("conditioning artifact belongs to a different role")
            return proof(value, out, req)

        return run_control_stage(
            root,
            source,
            area,
            grant,
            journal,
            host,
            runtime,
            req,
            data,
            conditioning_work(group, phase, len(rows), bank_count),
            extra,
            check,
            stage,
            recovery=recovery,
        )

    def capture(key, call):
        try:
            return call()
        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as exc:
            errors[key] = type(exc).__name__
            if recovery is not None:
                match = re.search(
                    r"conditioning evidence: ([a-z_,]+)\nconditioning observations: (\{[^\n]*\})$",
                    str(exc),
                )
                if match:
                    recovery["failure_checks"][key] = match[1].split(",")
                    values = json.loads(match[2])
                    fields = (
                        "reported_devices",
                        "explicit_mapping",
                        "launcher_devices_match",
                        "engine_argv_present",
                        "role_view_present",
                        "sole_render_sensor_matches",
                        "samples",
                        "covered_us",
                        "measured_us",
                        "worker_us",
                        "lease_us",
                        "dispatch_us",
                        "peak_device_bytes",
                    )
                    recovery["observations"][key] = {
                        k: values[k]
                        for k in fields
                        if k in values and (values[k] is None or type(values[k]) in (int, bool))
                    }
                else:
                    # Exceptions from private files may contain case names or paths.
                    known = (
                        "original registered verifier rejection required",
                        "original settled failed dispatch required",
                        "original conditioning input differs",
                        "original dispatch exceeds its committed charge",
                        "original role confinement, producer or supervision clock differs",
                        "registered model, opportunity, conditioning or search clock differs",
                        "issued attempts differ from the registered opportunity",
                    )
                    reason = next((s for s in known if str(exc).endswith(s)), type(exc).__name__)
                    recovery["failure_checks"][key] = [reason]
            print(f"{key}: {exc}", file=sys.stderr)
            return None

    def identify(value, out, req):
        tensors = value["tensors"]
        if set(tensors) != set(models) or any(
            not re.fullmatch("sha256:[a-f0-9]{64}", h) for h in tensors.values()
        ):
            raise ValueError("both model tensor identities are required")
        return dict(tensors=tensors)

    pairs = {}
    try:
        prepared = capture(
            "models",
            lambda: operate(
                "paired-models",
                "models",
                dict(models=models),
                [(paths[k], "/work/" + m["weight"]["path"]) for k, m in models.items()],
                identify,
                {},
            ),
        )
        if prepared:
            model_op = prepared[0]
            files["models.json"] = dict(
                receipt=model_op["identity"], elapsed_us=model_op["evidence"]["elapsed_us"]
            )
            pairs = conditioning_probes(
                plan["identity"], rows, prompts, models, model_op["evidence"]["tensors"]
            )
            retain(
                area / "probes.json",
                seal(dict(plan=plan["identity"], model_receipt=model_op["identity"], pairs=pairs)),
            )
        for group, probes in pairs.items():
            method, _ = CONDITIONING_GROUPS[group]
            original = models[method]
            data = dict(prompts=prompts, model=original)

            def check(value, out, req):
                return check_paired_search(value, out, req, data)

            result = capture(
                "search-" + group,
                lambda: operate(
                    group,
                    "search",
                    data,
                    [(paths[method], "/work/" + original["weight"]["path"])],
                    check,
                    dict(order=order[group], probes=probes, model_receipt=model_op["identity"],
                         expected_native=model_op["evidence"]["native"]),
                ),
            )
            if result:
                operation, output = result
                evidence = operation["evidence"]
                searched[group] = result
                files["search-" + group + ".json"] = dict(
                    receipt=operation["identity"],
                    **select_fields(
                        evidence,
                        "elapsed_us",
                        "worker_us",
                        "lease_us",
                        "peak_device_bytes",
                        "telemetry",
                        "native",
                        "bank_programs",
                    ),
                    modes={
                        m: select_fields(v, "elapsed_us", "components")
                        for m, v in evidence["modes"].items()
                    },
                )
        if searched.keys() == CONDITIONING_GROUPS.keys():
            barrier = seal(
                dict(
                    plan=plan["identity"],
                    searches={g: r[0]["identity"] for g, r in searched.items()},
                )
            )
            retain(area / "sealed-searches.json", barrier)
            for group, (search_op, output) in searched.items():

                def score_group():
                    children, extra, generations = {}, {}, {}
                    for m, v in search_op["evidence"]["modes"].items():
                        ref = v["artifact"]
                        generations[m] = read_bound_record(
                            output, ref, version="discovery-candidates/v1"
                        )
                        dest = "inputs/" + m + ".json"
                        extra["/work/" + dest] = confined(output, ref["path"])
                        children[m] = dict(artifact=dict(ref, path=dest), case_us=v["case_us"])
                    data = dict(truth=truth)

                    def check(value, out, req):
                        return check_paired_scores(value, out, req, data, generations)

                    return operate(
                        group,
                        "score",
                        data,
                        [(p, k) for k, p in extra.items()],
                        check,
                        dict(
                            search=children,
                            barrier=barrier["identity"],
                            expected_native=search_op["evidence"]["native"],
                            order=order[group],
                            probes=pairs[group],
                        ),
                        search_op["evidence"]["bank_programs"],
                    )

                result = capture("score-" + group, score_group)
                if result:
                    op = result[0]
                    row = seal(
                        dict(
                            group=group,
                            search=search_op["identity"],
                            score=op["identity"],
                            barrier=barrier["identity"],
                            **select_fields(
                                op["evidence"],
                                "elapsed_us",
                                "worker_us",
                                "native",
                                "modes",
                                "paired_selected",
                            ),
                        )
                    )
                    retain(area / (group + ".json"), row)
                    finished[group] = files["paired-" + group + ".json"] = row
    except (KeyError, TypeError, ValueError, OSError, RuntimeError, KeyboardInterrupt) as exc:
        errors["matrix"] = type(exc).__name__
        print(f"conditioning matrix: {exc}", file=sys.stderr)
    balance = settled_control_balance(area, journal, grant)
    if (
        balance["pending"]
        or balance["consumed"]["elapsed_us"] + balance["remaining"]["elapsed_us"]
        != grant["total_us"]
    ):
        raise ValueError("conditioning charges do not reconcile to the original grant")
    issued, leases, lease_us = conditioning_leases(area, grant, seed, balance)
    report.update(
        status="COMPLETE"
        if not errors and len(finished) == len(CONDITIONING_GROUPS)
        else "PARTIAL",
        grant=grant["identity"],
        elapsed_us=balance["consumed"]["elapsed_us"],
        remaining_us=balance["remaining"]["elapsed_us"],
        charged_work=balance["consumed"],
        gpu_dispatches=leases if issued == leases else None,
        hip_attempts=issued,
        verified_leases=leases,
        recorded_lease_us=lease_us,
    )
    if recovery is not None:
        recovery["additional_hip_attempts"] = issued - recovery["original_hip_attempts"]
        if recovery["additional_hip_attempts"] != 0:
            raise ValueError("conditioning recovery issued another HIP attempt")
    files["verification.json"] = dict(
        scope="paired_conditioning_on_exposed_families",
        research_dispatches=0,
        plan=plan["identity"],
        sealed_searches=barrier["identity"] if finished else None,
        verified={g: {"search": v["search"], "score": v["score"]} for g, v in finished.items()},
    )
    files["forecast.json"] = conditioning_prices(files)
    if report["status"] == "COMPLETE":
        receipt = seal(
            dict(
                plan=plan["identity"],
                grant=grant["identity"],
                groups={g: v["identity"] for g, v in finished.items()},
                charged=balance["consumed"],
                leases=leases,
                lease_us=lease_us,
            )
        )
        retain(area / "receipt.json", receipt)
        report["receipt"] = receipt["identity"]
        if recovery is not None:
            reviewed = seal(dict(result=receipt["identity"], recovery=digest(recovery)))
            retain(area / "recovery-receipts" / (reviewed["identity"][7:] + ".json"), reviewed)
            report["recovery_receipt"] = reviewed["identity"]
    return report


def donor_isolated_controls(action, root, source, revision, minutes, previous=None, *, bank_receipt=None, recovery=None):
    import runpy
    from oeis_learn.generalization.progressive.lifecycle import RunStore
    from oeis_learn.generalization.progressive.velocity_records import SESSION
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_handoff import stage

    files = {}
    step = "preflight"
    try:
        inventory = action("preflight")["inventory"]
        files["preflight.json"] = dict(
            gpu_dispatches=inventory["gpu_dispatches"],
            mount=select_fields(
                inventory.get("mount", {}),
                "available_bytes",
                "available_inodes",
                "required_bytes",
                "required_inodes",
                "deficit_bytes",
                "deficit_inodes",
            ),
            full_wave_capacity_registration=(inventory.get("capacity") or {}).get("registration"),
            **select_fields(
                inventory["prices"],
                "supported_wave",
                "common_probe",
                "followup",
                "followup_reserve_us",
            ),
        )
        wave = inventory["prices"].get("supported_wave")
        cycles = inventory["prices"].get("checkpoint_cycles") or {}
        if (
            inventory["gpu_dispatches"] != 0
            or not wave
            or cycles.get("receipt") != wave["checkpoint_receipt"]
        ):
            raise ValueError(
                "verified four-treatment preflight required before issuing a preparation grant"
            )
        step = "checkpoint-verification"
        checked = action("verify", at=cycles["candidate"])
        if (
            checked["mechanical_status"] != "VERIFIED"
            or checked["verification_scope"] != "private_checkpoint_matrix"
            or checked["scientific_verdict"] is not None
            or checked["report_ref"] != cycles["receipt"]
            or checked["last_committed_ref"] != cycles["receipt"]
        ):
            raise ValueError("original checkpoint verification differs")
        files["checkpoint-verification.json"] = select_fields(
            checked,
            "mechanical_status",
            "verification_scope",
            "last_committed_ref",
            "report_ref",
            "scientific_verdict",
        )
        step = "exposed-inputs"
        folder = confined(root, "run-state/discovery/" + SESSION)
        with RunStore.owner(folder):
            panel = exposed_control_inputs(root)
            step = "paid-runtime"
            runtime = paid_control_runtime(root, source, folder, cycles["receipt"])
            host_api = runpy.run_path(str(source / "scripts/foundation_local.py"))
            args = host_api["parser"]().parse_args(
                [
                    "--engine",
                    "podman",
                    "discovery",
                    "round029",
                    "--workspace",
                    str(root),
                    "--action",
                    "preflight",
                ]
            )
            host = host_api["Host"](args, source_root=source, workspace=root, revision=revision)
            if host.engine.image(runtime["image"]) != runtime["image"]:
                raise ValueError("qualified runtime image differs")
            step = "cpu-roles"
            if bank_receipt:
                step = "conditioning-roles"
                report = conditioning_matrix(root, source, folder, revision, minutes, runtime, panel, wave,
                                             inventory["cuts"], host, stage, bank_receipt, recovery=recovery)
            elif previous:
                report = bank_price_matrix(root, source, folder, revision, minutes, runtime, panel,
                                           wave, host, stage, previous)
            else:
                report = isolated_control_matrix(
                    root, source, folder, revision, minutes, runtime, panel, wave, host, stage
                )
        report["files"].update(files)
        return report
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        print(f"{step}: {exc}", file=sys.stderr)
        result = dict(
            candidate=revision,
            scope=CONDITIONING if bank_receipt else BANK_PRICES if previous else ISOLATED_CONTROLS,
            status="UNVERIFIED" if step in {"cpu-roles", "conditioning-roles"} else "BLOCKED",
            stage=step,
            error=type(exc).__name__,
            grant_minutes=int(minutes),
            files=files,
            research_run_requested=False,
            scientific_verdict=None,
            gpu_dispatches=None if step == "conditioning-roles" else 0,
        )
        if step not in {"cpu-roles", "conditioning-roles"}:
            result.update(grant=None, elapsed_us=0)
        return result


def donor_control_history(action, root, revision):
    files, checks, errors = {}, {}, {}
    report = dict(
        candidate=revision,
        scope="round029-control-history/v1",
        grant_minutes=0,
        gpu_dispatches=0,
        research_run_requested=False,
        scientific_verdict=None,
        files=files,
        checks=checks,
        errors=errors,
    )

    def failed(name, exc):
        checks[name] = "FAILED"
        errors[name] = type(exc).__name__
        print(f"{name}: {exc}", file=sys.stderr)

    try:
        history = read_history(root)
        for method, cohorts in history["controls"].items():
            files["control-" + method + ".json"] = dict(
                method=method,
                models=cohorts,
                basis=history["basis"],
                price_scope=history["price_scope"],
            )
        files["coverage.json"] = select_fields(
            history,
            "coverage",
            "basis",
            "status",
            "price_scope",
            "current_isolated_control_price_us",
        )
        checks["historical-controls"] = history["status"]
    except (OSError, KeyError, ValueError, TypeError) as exc:
        failed("historical-controls", exc)

    prices = {}
    try:
        inventory = action("preflight")["inventory"]
        if inventory["gpu_dispatches"] != 0:
            raise ValueError("control history preflight issued device work")
        amounts = (
            "device",
            "available_bytes",
            "available_inodes",
            "required_bytes",
            "required_inodes",
            "reserved_bytes",
            "reserved_inodes",
            "deficit_bytes",
            "deficit_inodes",
        )
        mounts = inventory["capacity"]["mounts"]
        files["capacity.json"] = dict(
            registration=inventory["capacity"]["registration"],
            mount=select_fields(inventory["mount"], *amounts),
            mounts=None if mounts is None else [select_fields(m, *amounts) for m in mounts],
        )
        checks["capacity"] = "COMPLETE"
        prices = inventory["prices"]
    except (Exception, SystemExit) as exc:
        failed("capacity", exc)

    try:
        producer = (prices.get("checkpoint_cycles") or {}).get("candidate")
        if not isinstance(producer, str) or not re.fullmatch(r"[a-f0-9]{40}", producer):
            raise ValueError("missing original checkpoint candidate")
        # The published verifier compares its own source snapshot to the funded binding.
        checked = action("verify", at=producer)
        if (
            checked["mechanical_status"] != "VERIFIED"
            or checked["verification_scope"] != "private_checkpoint_matrix"
            or checked["scientific_verdict"] is not None
            or checked["report_ref"] != checked["last_committed_ref"]
        ):
            raise ValueError("expected the completed private checkpoint matrix")
        files["verification.json"] = select_fields(
            checked,
            "verification_scope",
            "mechanical_status",
            "report_ref",
            "last_committed_ref",
            "scientific_verdict",
        )
        checks["private-verification"] = "COMPLETE"
    except (Exception, SystemExit) as exc:
        failed("private-verification", exc)

    if prices and "verification.json" in files:
        try:
            cycles = prices["checkpoint_cycles"]
            wave = prices["supported_wave"]
            if (
                cycles["receipt"] != checked["report_ref"]
                or wave["checkpoint_receipt"] != cycles["receipt"]
            ):
                raise ValueError("supported forecast belongs to a different checkpoint measurement")
            files["capacity.json"]["measured_checkpoint_trees"] = {
                m: select_fields(row, "checkpoint_bytes", "checkpoint_inodes")
                for m, row in cycles["methods"].items()
            }
            files["forecast.json"] = select_fields(
                prices,
                "supported_wave",
                "wave",
                "common_probe",
                "followup",
                "followup_reserve_us",
                "wave_with_reserve_us",
            )
            checks["supported-forecast"] = "COMPLETE"
        except (KeyError, ValueError, TypeError) as exc:
            failed("supported-forecast", exc)
    else:
        checks["supported-forecast"] = "UNAVAILABLE"
    report["status"] = "COMPLETE" if all(v == "COMPLETE" for v in checks.values()) else "PARTIAL"
    return report


def prior_launch_projection(root, parent, plan, wave):
    from oeis_learn.generalization.progressive.contracts import load, seal
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_velocity_launch import settled_area

    bank, bplan, _, _ = settled_area(
        root, parent / "matched-bank-prices", "receipt", plan["previous"]
    )
    controls, cplan, _, _ = settled_area(
        root, parent / "isolated-control-prices", "receipt", bplan["controls"]
    )
    if bplan["checkpoint"] != plan["checkpoint"] or cplan["checkpoint"] != plan["checkpoint"]:
        raise ValueError("different bank or control price source")

    def records(directory, field):
        refs = load(confined(directory, "receipt.json"))[field]
        rows = {k: load(confined(directory, k + ".json")) for k in refs}
        if any(r != seal(r) or r["identity"] != refs[k] for k, r in rows.items()):
            raise ValueError("original price row differs")
        return rows

    projection = bank_sample_forecast(
        records(bank, "blocks"), control_cost_forecast(records(controls, "controls"), wave)
    )
    return dict(controls=bplan["controls"], receipt=plan["previous"], projection=projection)


def paid_conditioning_operation(area, state, grant, coordinates, *, expected):
    from oeis_learn.generalization.progressive.contracts import load, seal
    from oeis_learn.generalization.progressive.velocity_records import operation_key
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_velocity_operations import (
        check_reserved_work,
        input_record,
        operation_receipt,
    )

    logical = operation_key(coordinates, 0)["logical"]
    block = confined(area, "operations/" + logical[7:])
    row = load(confined(block, "receipt.json"))
    req = load(confined(block, "input.json"))
    attempt = row["attempt"]
    key = f"{logical}/dispatch/{attempt}"
    boundary = load(confined(block, f"boundary-{attempt:04d}.json"))
    work = state["reserved_work"][key]
    if (
        expected != seal(expected)
        or req != input_record(coordinates, expected, dict(work, elapsed_us=0), grant)
        or row != operation_receipt(coordinates, expected, grant, attempt, row["evidence"])
        or key not in state["committed"]
        or boundary != seal(boundary)
        or boundary["result"] != row["identity"]
    ):
        raise ValueError("original paid operation request or verified boundary differs")
    check_reserved_work(state, logical + "/dispatch/", work)
    return row


def conditioning_role_request(area, plan, coordinates):
    from oeis_learn.generalization.progressive.contracts import load, seal
    from oeis_learn.pipeline_validation.identity import confined

    phase, group = coordinates["role"], coordinates["method"]
    req = load(confined(area, f"inputs/{group}/{phase}/request.json"))
    if (
        req != seal(req)
        or req["plan"] != plan["identity"]
        or req["phase"] != phase
        or req["method"] != group
        or any(
            req[k] != plan[k]
            for k in ("version", "candidate", "seed", "wheel", "image", "physical", "source")
        )
    ):
        raise ValueError("conditioning role differs from the registered plan")
    return req


def paid_search_projection(area, state, grant, plan, coords):
    from oeis_learn.generalization.progressive.contracts import load, seal
    from oeis_learn.generalization.progressive.velocity_records import operation_key
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_velocity_operations import (
        input_record,
        operation_receipt,
    )

    request = conditioning_role_request(area, plan, coords)
    logical = operation_key(coords, 0)["logical"]
    block = confined(area, "operations/" + logical[7:])
    if confined(block, "receipt.json", exists=False).exists():
        original = paid_conditioning_operation(area, state, grant, coords, expected=request)
    else:
        old = load(confined(block, "first-error.json"))
        number = old["attempt"]
        reader_coords = dict(coords, role="verify-paid-search")
        reader_id = operation_key(reader_coords, 0)["logical"][7:]
        registered = load(confined(area, f"operations/{reader_id}/input.json"))["request"]
        key = f"{logical}/dispatch/{number}"
        window = state["windows"][key]
        expected = seal(
            dict(
                reader=registered["reader"],
                original=request["identity"],
                first_error=old["identity"],
                key=key,
                charged_us=window["charged_us"],
            )
        )
        if (
            old != seal(old)
            or old["error"] != "ValueError"
            or old["stage"] != coords["stage"]
            or type(number) is not int
            or number not in (0, 1)
            or not re.fullmatch("[0-9a-f]{40}", registered["reader"])
            or window["event"] != "COMMIT"
            or window["end_us"] is None
            or window["charged_us"] != window["end_us"] - window["start_us"]
            or key not in state["committed"]
            or load(confined(block, "input.json"))
            != input_record(coords, request, dict(state["reserved_work"][key], elapsed_us=0), grant)
            or confined(block, f"boundary-{number:04d}.json", exists=False).exists()
        ):
            raise ValueError("original failed conditioning dispatch differs")
        receipt = paid_conditioning_operation(area, state, grant, reader_coords, expected=expected)
        original = receipt["evidence"]["original"]
        if (
            original != operation_receipt(coords, request, grant, number, original["evidence"])
            or original["evidence"]["elapsed_us"] > window["charged_us"]
        ):
            raise ValueError("paid search evidence differs")
    evidence = original["evidence"]
    return dict(
        receipt=original["identity"],
        **select_fields(
            evidence,
            "elapsed_us",
            "worker_us",
            "lease_us",
            "peak_device_bytes",
            "telemetry",
            "native",
            "bank_programs",
        ),
        modes={
            k: select_fields(v, "elapsed_us", "components") for k, v in evidence["modes"].items()
        },
    )


def paid_conditioning_parts(root, area, plan, grant, state, wave, *, groups, searches, models):
    from oeis_learn.generalization.progressive.contracts import load, seal
    from oeis_learn.pipeline_validation.identity import confined

    if plan["version"] != CONDITIONING or plan["checkpoint"] != wave["checkpoint_receipt"]:
        raise ValueError("original paired plan and checkpoint required")
    parts = {"previous-prices.json": prior_launch_projection(root, area.parent.parent, plan, wave)}

    def coords(group, role):
        return conditioning_coordinates(grant, group, role, plan["seed"])

    if models:
        coordinate = coords("paired-models", "models")
        model = paid_conditioning_operation(
            area,
            state,
            grant,
            coordinate,
            expected=conditioning_role_request(area, plan, coordinate),
        )
        parts["models.json"] = dict(
            receipt=model["identity"], elapsed_us=model["evidence"]["elapsed_us"]
        )
    for g in searches:
        parts["search-" + g + ".json"] = paid_search_projection(
            area, state, grant, plan, coords(g, "search")
        )
    if groups:
        if set(searches) != set(CONDITIONING_GROUPS):
            raise ValueError("all searches must seal before scoring")
        barrier = load(confined(area, "sealed-searches.json"))
        if barrier != seal(
            dict(
                plan=plan["identity"],
                searches={g: parts["search-" + g + ".json"]["receipt"] for g in searches},
            )
        ):
            raise ValueError("original paid search barrier differs")
        for g in groups:
            coordinate = coords(g, "score")
            request = conditioning_role_request(area, plan, coordinate)
            if request["barrier"] != barrier["identity"]:
                raise ValueError("scorer used a different sealed search barrier")
            score = paid_conditioning_operation(area, state, grant, coordinate, expected=request)
            row = load(confined(area, g + ".json"))
            expected = seal(
                dict(
                    group=g,
                    search=barrier["searches"][g],
                    score=score["identity"],
                    barrier=barrier["identity"],
                    **select_fields(
                        score["evidence"],
                        "elapsed_us",
                        "worker_us",
                        "native",
                        "modes",
                        "paired_selected",
                    ),
                )
            )
            if row != expected:
                raise ValueError("scored conditioning group differs from its original boundary")
            parts["paired-" + g + ".json"] = row
    return parts


def bound_conditioning(root, summary, parts, wave):
    from oeis_learn.generalization.progressive.contracts import load, seal
    from oeis_learn.generalization.progressive.velocity_records import SESSION
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_velocity_launch import settled_area

    if (
        summary["status"] not in {"COMPLETE", "PARTIAL"}
        or parts["preflight.json"]["supported_wave"] != wave
    ):
        raise ValueError("same paid supported wave and issued conditioning grant required")
    parent = confined(root, "run-state/discovery/" + SESSION + "/preparation/paired-conditioning")
    area, plan, grant, state = settled_area(
        root, parent, "plan", parts["verification.json"]["plan"], grant_ref=summary["grant"]
    )
    if (
        grant["identity"] != summary["grant"]
        or summary["candidate"] != plan["candidate"]
        or state["consumed"] != summary["charged_work"]
    ):
        raise ValueError("conditioning diagnostic differs from the settled original")
    records = paid_conditioning_parts(
        root,
        area,
        plan,
        grant,
        state,
        wave,
        groups=[g for g in CONDITIONING_GROUPS if "paired-" + g + ".json" in parts],
        searches=[g for g in CONDITIONING_GROUPS if "search-" + g + ".json" in parts],
        models="models.json" in parts,
    )
    if any(parts[k] != v for k, v in records.items()):
        raise ValueError("returned conditioning inputs differ from original paid parts")
    if summary["status"] == "COMPLETE":
        if (
            any("paired-" + g + ".json" not in records for g in CONDITIONING_GROUPS)
            or "models.json" not in records
        ):
            raise ValueError("completed conditioning requires every paid role")
        receipt = load(confined(area, "receipt.json"))
        if (
            receipt != seal(receipt)
            or receipt["identity"] != summary["receipt"]
            or receipt["plan"] != plan["identity"]
            or receipt["grant"] != grant["identity"]
            or receipt["charged"] != state["consumed"]
        ):
            raise ValueError("original paired conditioning receipt differs")


def settled_launch_estimate(root, wave, preparation, reserve):
    """Read the original settled preparation directly on an already qualified donor."""
    from oeis_learn.generalization.progressive.contracts import load, seal
    from oeis_learn.generalization.progressive.velocity_records import SESSION
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_velocity_launch import settled_area

    parent = confined(root, "run-state/discovery/" + SESSION + "/preparation/paired-conditioning")
    selected = (preparation.get("wave_space") or {}).get("sample_grant")
    area, plan, grant, state = settled_area(
        root, parent, "plan", wave["checkpoint_receipt"], field="checkpoint", grant_ref=selected
    )
    receipt = load(confined(area, "receipt.json"))
    parts = paid_conditioning_parts(
        root,
        area,
        plan,
        grant,
        state,
        wave,
        groups=CONDITIONING_GROUPS,
        searches=CONDITIONING_GROUPS,
        models=True,
    )
    issued, leases, lease_us = conditioning_leases(area, grant, plan["seed"], state)
    if (
        receipt != seal(receipt)
        or receipt["plan"] != plan["identity"]
        or receipt["grant"] != grant["identity"]
        or receipt["charged"] != state["consumed"]
        or issued != leases
        or leases < len(CONDITIONING_GROUPS)
        or (leases, lease_us) != (receipt["leases"], receipt["lease_us"])
        or not 0 < lease_us <= state["consumed"]["elapsed_us"]
        or receipt["groups"]
        != {g: parts["paired-" + g + ".json"]["identity"] for g in CONDITIONING_GROUPS}
    ):
        raise ValueError("settled paired preparation receipt differs")
    parts["preflight.json"] = dict(supported_wave=wave, followup_reserve_us=reserve)
    parts["checkpoint-verification.json"] = dict(
        report_ref=preparation["checkpoint_receipt"], mechanical_status="VERIFIED"
    )
    estimate = estimate_with_preparation(parts, preparation)
    paired = conditioning_prices(parts)
    # Repeat the exposed ordinary proxy for each funded learner and the D measurement once.
    learner_count = len(wave["methods"])
    components = {
        k: sum((1 if name == "D-full" else learner_count) * sum(mode[k] for mode in group["modes"].values())
               for name, group in paired["groups"].items())
        for k in ("search_us", "selected_us", "bank_us")
    }
    expected = paired_expected(parts, paired, learner_count)
    estimate["conditioning_components"] = dict(components, overhead_us=expected - sum(components.values()))
    estimate["conditioning_extra_role_us"] = {
        name: group["shared_overhead_us"] for name, group in paired["groups"].items()
    }
    estimate["sources"] = dict(
        paired=receipt["identity"],
        bank=plan["previous"],
        controls=parts["previous-prices.json"]["controls"],
        checkpoint=wave["checkpoint_receipt"],
        fresh=wave["fresh_receipt"],
        accounting=preparation.get("accounting_receipt"),
    )
    return estimate


def conditioning_copy(member):
    raw = base64.b64decode(member["base64"], validate=True)
    if len(raw) > 1 << 20 or hashlib.sha256(raw).hexdigest() != member["sha256"]:
        raise ValueError("conditioning ZIP copy differs")
    verify_conditioning_zip(io.BytesIO(raw))
    return diagnostic_zip_members(io.BytesIO(raw), CONDITIONING)


def estimate_with_preparation(parts, preparation):
    refreshed = dict(parts["preflight.json"], launch_preparation=preparation)
    return launch_estimate({**parts, "preflight.json": refreshed})


def donor_launch_prices(action, root, revision, minutes, supplied):
    from oeis_learn.generalization.progressive.lifecycle import RunStore
    from oeis_learn.generalization.progressive.velocity_records import SESSION
    from oeis_learn.pipeline_validation import progressive_velocity_ledger_prices as accounting
    from oeis_learn.pipeline_validation.identity import confined

    files, errors = {}, {}
    report = dict(
        scope=LAUNCH_PRICES,
        candidate=revision,
        grant_minutes=int(minutes),
        grant=None,
        research_run_requested=False,
        scientific_verdict=None,
        gpu_dispatches=0,
        files=files,
        errors=errors,
    )
    bound = False
    try:
        with Path(supplied).open("rb") as f:
            raw = f.read((1 << 20) + 1)
        member = dict(
            sha256=hashlib.sha256(raw).hexdigest(), base64=base64.b64encode(raw).decode("ascii")
        )
        summary, parts = conditioning_copy(member)
        files["conditioning.json"] = member
        report["conditioning_status"] = summary["status"]
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
        errors["conditioning"] = type(exc).__name__
    step = "paid-checkpoint-inventory"
    try:
        initial = action("preflight")["inventory"]
        prices = initial["prices"]
        fresh, cycles = prices["fresh_start"], prices["checkpoint_cycles"]
        wave = prices["supported_wave"]
        if initial["gpu_dispatches"] != 0 or wave["checkpoint_receipt"] != cycles["receipt"]:
            raise ValueError("verified current four-treatment checkpoint required")
        files["preparation.json"] = prices["launch_preparation"]
        if "conditioning.json" in files:
            try:
                bound_conditioning(root, summary, parts, wave)
                if parts["preflight.json"]["followup_reserve_us"] != prices["followup_reserve_us"]:
                    raise ValueError("original follow-up reserve differs")
                bound = True
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors["conditioning"] = type(exc).__name__
        step = "update-accounting"
        folder = confined(root, "run-state/discovery/" + SESSION)
        with RunStore.owner(folder):
            plan, order, lengths = accounting.source_inputs(root, fresh, cycles, revision)
            result = accounting.run(
                root,
                folder,
                plan,
                order,
                lengths,
                minutes=int(minutes),
                floor=5 << 30,
                registered=initial["capacity"]["mounts"],
            )
            files["accounting.json"] = result
            report["grant"] = result["grant"]
            if result["errors"]:
                errors[step] = "incomplete CPU batches"
            else:
                accounting.record(root, folder, plan, order, lengths, result)
        step = "launch-preflight"
        # Reopen through the normal operator entry point after releasing the session lock.
        current = action("preflight")["inventory"]
        prep = current["prices"]["launch_preparation"]
        if (
            current["gpu_dispatches"] != 0
            or current["prices"]["supported_wave"] != wave
            or (not result["errors"] and prep.get("accounting_receipt") != result["identity"])
        ):
            raise ValueError("normal preflight did not reproduce the paid update accounting")
        files["preparation.json"] = prep
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, SystemExit) as exc:
        print(f"{step}: {exc}", file=sys.stderr)
        errors[step] = type(exc).__name__
    if bound:
        try:
            files["estimate.json"] = estimate_with_preparation(parts, files["preparation.json"])
        except (ValueError, KeyError, TypeError) as exc:
            errors["estimate"] = type(exc).__name__
    complete = bound and summary["status"] == "COMPLETE" and not errors
    return dict(report, status="COMPLETE" if complete else "PARTIAL")


def review_mounts(capacity):
    """Recalculate public owner guards, mount demand, reserves and the reported fit."""
    def count(value):
        if type(value) is not int or value < 0:
            raise ValueError("nonnegative physical measurement required")
        return value

    def review(rows):
        if not rows:
            raise ValueError("at least one destination mount required")
        devices, owners, margins = set(), set(), []
        for mount in rows:
            device = count(mount["device"])
            if device in devices:
                raise ValueError("duplicate destination mount")
            devices.add(device)
            total = dict(bytes=0, inodes=0)
            for owner in mount["owners"]:
                key = owner["owner"]
                if key in owners or not key:
                    raise ValueError("duplicate storage owner")
                owners.add(key)
                guards = {g["units"]:g for g in owner["guards"]}
                if guards.keys() != total.keys() or len(owner["guards"]) != len(guards):
                    raise ValueError("storage owner requires byte and inode guards")
                for unit, guard in guards.items():
                    peak = max(count(guard["observed"]),count(guard["projected"]))
                    if (not peak or guard["owner"] != key or guard["basis"] != "provisional"
                            or count(guard["value"]) != 2*peak or count(guard["headroom"]) != peak):
                        raise ValueError("storage guard differs from its scoped measurement")
                    need = max(0, guard["value"] - count(owner["retained"][unit]))
                    if count(owner["incremental"][unit]) != need:
                        raise ValueError("incremental storage differs from the retained credit")
                    total[unit] += need
            result = dict(device=device)
            for unit, need in total.items():
                for label in ("available","reserved","required","deficit"):
                    k = label + "_" + unit
                    result[k] = count(mount[k])
                short = max(0, need + result["reserved_"+unit] - result["available_"+unit])
                if result["required_"+unit] != need or result["deficit_"+unit] != short:
                    raise ValueError("mount requirement or physical deficit differs")
            margins.append(result)
        return margins

    missing = capacity["missing"]
    if not isinstance(missing, list) or not all(isinstance(k,str) and k for k in missing):
        raise ValueError("list of missing storage prices required")
    wave = review(capacity["wave"])
    fits = not any(m["deficit_bytes"] or m["deficit_inodes"] for m in wave)
    expected = False if not fits else None if missing else True
    if capacity["fits_with_reserve"] is not expected:
        raise ValueError("reported wave fit differs from its destination mounts")
    result = dict(wave=wave, first_ordinary_cut=review(capacity["first_ordinary_cut"]),
                  fits_with_reserve=expected, missing=missing)
    if capacity.get("ordinary_initialization"):
        result["ordinary_initialization"] = review(capacity["ordinary_initialization"])
    return result


def verify_launch_zip(path):
    summary, files = diagnostic_zip_members(path, LAUNCH_PRICES)
    if summary["status"] not in {"PARTIAL", "COMPLETE"}:
        return dict(status=summary["status"], verified_members=len(files))
    if summary["gpu_dispatches"] != 0:
        raise ValueError("CPU launch preparation issued a device dispatch")
    result = files.get("accounting.json")
    if result is not None:
        original = {k: v for k, v in result.items() if k != "identity"}
        canonical = json.dumps(
            original, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        work = result["charged_work"]
        if (
            result["identity"] != "sha256:" + hashlib.sha256(canonical).hexdigest()
            or result["grant"] != summary["grant"]
            or work["tokens"] != 0
            or work["elapsed_us"] + result["remaining_us"] != summary["grant_minutes"] * 60_000_000
            or set(result["batches"]) | set(result["errors"]) != {"16", "32"}
        ):
            raise ValueError("original CPU accounting scope or charge differs")
    conditioning = None
    if "conditioning.json" in files:
        conditioning, parts = conditioning_copy(files["conditioning.json"])
        if summary["conditioning_status"] != conditioning["status"]:
            raise ValueError("conditioning status differs from original ZIP")
        if "estimate.json" in files:
            if estimate_with_preparation(parts, files["preparation.json"]) != files["estimate.json"]:
                raise ValueError("returned launch estimate differs")
    if "estimate.json" in files and (not conditioning or "conditioning" in summary["errors"]):
        raise ValueError("a launch estimate requires privately linked conditioning evidence")
    independently_reopened = not (
        {"launch-preflight", "update-accounting"} & summary["errors"].keys()
    )
    if result and not result["errors"] and independently_reopened:
        if files["preparation.json"].get("accounting_receipt") != result["identity"]:
            raise ValueError("published trajectory has different CPU prices")
    if summary["status"] == "COMPLETE" and (
        not result
        or result["errors"]
        or summary["errors"]
        or not conditioning
        or conditioning["status"] != "COMPLETE"
        or "estimate.json" not in files
        or files["preparation.json"].get("accounting_receipt") != result["identity"]
    ):
        raise ValueError(
            "complete preparation requires the original recovered matrix and CPU prices"
        )
    preparation = files.get("preparation.json", {})
    capacity = preparation.get("wave_space")
    margins = review_mounts(capacity) if capacity else None
    return dict(
        status=summary["status"],
        verified_members=len(files),
        estimate=files.get("estimate.json"),
        capacity=margins,
        preparation_errors=summary["errors"],
    )


def donor_initialization(action, root, revision, minutes, *, recovery=None):
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.pipeline_validation import progressive_velocity_prescore as prescore
    from oeis_learn.pipeline_validation.progressive_velocity_donation import retained_qualification

    files = {}
    report = dict(
        scope=INITIALIZATION,
        candidate=revision,
        grant_minutes=int(minutes),
        files=files,
        research_run_requested=False,
        scientific_verdict=None,
        status="PARTIAL",
        errors={},
    )
    stage = "preflight"
    area = grant = None
    try:
        if recovery:
            from oeis_learn.generalization.progressive.velocity_records import SESSION
            from oeis_learn.pipeline_validation.identity import confined
            from oeis_learn.pipeline_validation.progressive_velocity_initialization_recovery import (
                history,
                original_grant,
            )

            stage = "original-grant"
            authority = original_grant(root, revision, recovery["grant"], minutes)
            grant = load(authority)
            report["grant"] = grant["identity"]
            area = confined(
                root, f"run-state/discovery/{SESSION}/qualification/" + grant["identity"][7:]
            )
            stage = "original-attempt"
            files["recovery.json"] = dict(
                reader=recovery["reader"], before=history(area, grant, private_log=True)
            )
            stage = "preflight"
        flight = action("preflight")["inventory"]
        if flight["gpu_dispatches"] or flight["assessment"] != "sealed":
            raise ValueError("invalid read-only launch inventory")
        options = flight["prices"]["launch_options"]
        files["launch.json"] = options
        if options["scopes"]["ordinary_initialization"]["tail_with_reserve_us"] is None:
            raise ValueError("paid initialization and follow-up prices required")
        stage = "qualification"
        if not recovery:
            authority = prescore.authorize(root, revision, minutes)
            grant = load(authority)
            report["grant"] = grant["identity"]
        paid = action("qualify", authority)
        rec = paid["receipt"]
        if (
            paid["scope"] != "ordinary_initialization"
            or paid["research_dispatches"]
            or paid["charged_tokens"]
        ):
            raise ValueError("ordinary zero-token qualification required")
        stage = "private-replay"
        status, checked = action("status"), action("verify")
        if not (
            status["last_committed_ref"] == checked["report_ref"] == rec["identity"]
            and checked["verification_scope"] == "private_ordinary_initialization"
        ):
            raise ValueError("private closure differs from issued initialization")
        from oeis_learn.generalization.progressive.velocity_donation_report import REPORT
        from oeis_learn.pipeline_validation.identity import confined
        from oeis_learn.pipeline_validation.progressive_velocity_verify import archive_result

        public_report = confined(root, paid["receipt_path"]).parent / "share" / REPORT
        archive_check, code = archive_result(public_report, retained_qualification(root, grant))
        public_cut = prescore.read_member(public_report, prescore.REPORT_MEMBER)
        if (
            code
            or archive_check["report_ref"] != public_cut["identity"]
            or archive_check["last_committed_ref"] != rec["identity"]
        ):
            raise ValueError("standalone public archive verification differs from the private cut")
        files["cut.json"] = public_cut
        files["verification.json"] = select_fields(
            checked, "report_ref", "verification_scope", "mechanical_status"
        )
        files["verification.json"]["public_report"] = archive_check["report_ref"]
        report.update(
            status="COMPLETE",
            elapsed_us=paid["elapsed_us"],
            remaining_us=paid["remaining_authority"],
        )
    except (OSError, RuntimeError, ValueError, KeyError, TypeError, SystemExit) as exc:
        print(f"{stage}: {exc}", file=sys.stderr)
        report["errors"][stage] = type(exc).__name__
        if isinstance(exc, SystemExit):
            try:
                failed = json.loads(str(exc))
                files["failure.json"] = select_fields(
                    failed,
                    *[
                        k
                        for k in (
                            "mechanical_status",
                            "stage",
                            "elapsed_us",
                            "remaining_authority",
                            "deficit_bytes",
                            "deficit_inodes",
                        )
                        if k in failed
                    ],
                )
                diagnostic = failed.get("diagnostics") or failed.get("first_error")
                if diagnostic:
                    from oeis_learn.pipeline_validation.progressive_velocity_initialization_recovery import public_error

                    files["failure.json"]["cause"] = public_error(diagnostic)
            except (ValueError, TypeError):
                pass
    if "recovery.json" in files:
        try:
            files["recovery.json"]["after"] = history(
                area, grant, private_log=report["status"] != "COMPLETE"
            )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"final grant inspection: {exc}", file=sys.stderr)
            report["errors"]["final-grant"] = type(exc).__name__
            report["status"] = "PARTIAL"
    if grant and (
        report["status"] == "COMPLETE"
        or files.get("recovery.json", {})
        .get("after", {})
        .get("operations", {})
        .get("publish_prescore", {})
        .get("receipt")
    ):
        try:
            from oeis_learn.generalization.progressive.velocity_donation_report import QUALIFICATION
            from oeis_learn.generalization.progressive.velocity_records import SESSION
            from oeis_learn.pipeline_validation.identity import confined

            retained = retained_qualification(root, grant)
            with retained.open("rb") as stream:
                raw = stream.read((1 << 20) + 1)
            if len(raw) > 1 << 20:
                raise ValueError("bounded pre-score archive required")
            share = confined(
                root,
                f"run-state/discovery/{SESSION}/qualification/{grant['identity'][7:]}/share/{QUALIFICATION}",
            )
            with share.open("rb") as stream:
                if stream.read(len(raw) + 1) != raw:
                    raise ValueError(
                        "installed and independently retained pre-score archives differ"
                    )
            member = dict(
                sha256=hashlib.sha256(raw).hexdigest(), base64=base64.b64encode(raw).decode("ascii")
            )
            initialization_prescore(member, None, report)
            files["prescore.json"] = member
        except (OSError, RuntimeError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            print(f"pre-score archive: {exc}", file=sys.stderr)
            report["errors"]["prescore-copy"] = type(exc).__name__
            report["status"] = "PARTIAL"
    return report


def measurement(value):
    if type(value) is not int or value < 0:
        raise ValueError("nonnegative integer measurement required")
    return value


def mount_margins(mounts):
    rows = []
    devices = set()
    for mount in mounts:
        device = measurement(mount["device"])
        if device in devices:
            raise ValueError("duplicate destination mount")
        devices.add(device)
        row = dict(device=device)
        for unit in ("bytes", "inodes"):
            free, need, floor, deficit = (
                measurement(mount[k + "_" + unit])
                for k in ("available", "required", "reserved", "deficit")
            )
            margin = free - need - floor
            if deficit != max(0, -margin):
                raise ValueError("launch capacity deficit differs from mount totals")
            row["margin_" + unit] = margin
        rows.append(row)
    return rows


def initialization_margins(options):
    names = ("wave", "first_ordinary_cut", "ordinary_initialization")
    scopes = options["scopes"]
    if set(scopes) != set(names) or options["research_authority"] is not None:
        raise ValueError("expected physical launch scenarios without research authority")
    margins = {}
    eligible = []
    reserve = options["followup_reserve_us"]
    if reserve is not None:
        measurement(reserve)
    for name in names:
        scope = scopes[name]
        rows = mount_margins(scope["mounts"])
        margins[name] = rows
        shortage = any(min(row["margin_bytes"], row["margin_inodes"]) < 0 for row in rows)
        fits = scope["fits"]
        if (
            fits is not None
            and type(fits) is not bool
            or (fits is False) != shortage
            or fits is True
            and not rows
        ):
            raise ValueError("reported launch fit differs from physical demand")
        expected, tail = scope["expected_us"], scope["tail_us"]
        for price in (expected, tail, scope["tail_with_reserve_us"]):
            if price is not None:
                measurement(price)
        if expected is not None and tail is not None and expected > tail:
            raise ValueError("expected price exceeds tail")
        funded_time = tail + reserve if tail is not None and reserve is not None else None
        if scope["tail_with_reserve_us"] != funded_time:
            raise ValueError("launch tail and follow-up reserve differ")
        if fits is True and funded_time is not None:
            eligible.append(name)
    if options["next_scope"] != next(iter(eligible), None):
        raise ValueError("next launch scope differs from priced capacity")
    return dict(next_scope=options["next_scope"], capacity=margins)


def check_initialization_history(summary, member):
    phases = ("freeze_trials", "package", "verify_paid_package", "publish_prescore", "measure")
    build_phases = {"build_runtime", "package_runtime"}

    def ref(value):
        if not isinstance(value, str) or not re.fullmatch("sha256:[0-9a-f]{64}", value):
            raise ValueError("original initialization evidence digest required")

    def error(value):
        if value is not None:
            if set(value) != {"identity", "stage", "error", "diagnosis"}:
                raise ValueError("invalid public first error")
            ref(value["identity"])
            if not all(
                re.fullmatch("[A-Za-z_]{1,40}", value[k]) for k in ("stage", "error", "diagnosis")
            ):
                raise ValueError("invalid public error classification")

    if set(member) not in ({"reader", "before"}, {"reader", "before", "after"}) or not re.fullmatch(
        "[0-9a-f]{40}", member["reader"]
    ):
        raise ValueError("initialization recovery requires the original journal and pinned reader")
    total = measurement(summary["grant_minutes"]) * 60_000_000
    ref(summary["grant"])
    for snapshot in (member[k] for k in ("before", "after") if k in member):
        stages = snapshot["operations"]
        if (
            set(snapshot) != {"grant", "first_error", "operations", "elapsed_us", "remaining_us"}
            or snapshot["grant"] != summary["grant"]
            or set(stages) not in (set(phases), set(phases) | build_phases)
        ):
            raise ValueError("original initialization journal scope differs")
        error(snapshot["first_error"])
        elapsed = 0
        for name, state in stages.items():
            fields = {"attempts", "receipt", "first_error"}
            if name == "build_runtime" and state["receipt"]:
                fields.add("image_build")
                image = state["image_build"]
                ref(image["image"])
                if (
                    set(image) != {"image", "samples", "lowest"}
                    or measurement(image["samples"]) < 2
                    or not 1 <= len(image["lowest"]) <= 2
                ):
                    raise ValueError("invalid image build observations")
                devices = set()
                fields_per_mount = {"device"} | {
                    k + "_" + unit
                    for k in ("available", "required", "reserved", "deficit")
                    for unit in ("bytes", "inodes")
                }
                for row in image["lowest"]:
                    device = measurement(row["device"])
                    if device in devices or set(row) != fields_per_mount:
                        raise ValueError("invalid build storage device or measurements")
                    devices.add(device)
                    for unit in ("bytes", "inodes"):
                        free, need, floor, deficit = (
                            measurement(row[k + "_" + unit])
                            for k in ("available", "required", "reserved", "deficit")
                        )
                        if free < need + floor or deficit:
                            raise ValueError("image build exceeded its storage reserve")
            if set(state) != fields or len(state["attempts"]) > 2:
                raise ValueError("invalid initialization operation history")
            error(state["first_error"])
            if state["receipt"] is not None:
                ref(state["receipt"])
                if not any(a["event"] == "COMMIT" for a in state["attempts"]):
                    raise ValueError("original committed attempt required")
            for i, a in enumerate(state["attempts"]):
                if (
                    set(a)
                    != {
                        "ordinal",
                        "event",
                        "charged_us",
                        "process_exit",
                        "diagnosis",
                        "stderr_digest",
                    }
                    or measurement(a["ordinal"]) != i
                    or a["event"] not in {"RESERVE", "COMMIT"}
                    or a["process_exit"] is not None
                    and type(a["process_exit"]) is not int
                    or not re.fullmatch("[a-z_]{1,40}", a["diagnosis"])
                ):
                    raise ValueError("invalid initialization attempt or supervisor result")
                if a["stderr_digest"] is not None:
                    ref(a["stderr_digest"])
                elapsed += measurement(a["charged_us"])
        if elapsed != measurement(snapshot["elapsed_us"]) or total != elapsed + measurement(
            snapshot["remaining_us"]
        ):
            raise ValueError("original operation debits differ from grant balance")
    before, after = member["before"], member.get("after")
    if after is None:
        if summary["status"] == "COMPLETE" or "final-grant" not in summary["errors"]:
            raise ValueError("final grant inspection missing")
        return
    if before["first_error"] and before["first_error"] != after["first_error"]:
        raise ValueError("first initialization failure changed")
    if before["operations"].keys() != after["operations"].keys():
        raise ValueError("recovery observations cover different operations")
    for phase in before["operations"]:
        old, new = (row["operations"][phase] for row in (before, after))
        if len(old["attempts"]) > len(new["attempts"]) or any(
            old.get(k) and new.get(k) != old[k] for k in ("receipt", "first_error", "image_build")
        ):
            raise ValueError("lost original initialization operation")
        for a, b in zip(old["attempts"], new["attempts"]):
            if a["event"] == "COMMIT" and a != b:
                raise ValueError("original settled initialization debit changed")
    stages = after["operations"]
    rebuilt = any(stages.get(k, {}).get("attempts") for k in build_phases)
    if rebuilt:
        lookup = stages["package"]
        if (
            lookup["receipt"]
            or not lookup["attempts"]
            or not lookup["first_error"]
            or lookup["first_error"]["diagnosis"] != "runtime_unavailable"
            or stages["verify_paid_package"]["attempts"]
            or stages["package_runtime"]["attempts"]
            and not stages["build_runtime"]["receipt"]
        ):
            raise ValueError("rebuilt runtime requires the original settled cache-only failure")
    if summary["status"] == "COMPLETE":
        required = {"freeze_trials", "publish_prescore", "measure"}
        required.update(build_phases if rebuilt else {"package"})
        if stages["verify_paid_package"]["attempts"]:
            required.add("verify_paid_package")
        if (
            after["elapsed_us"] != summary["elapsed_us"]
            or after["remaining_us"] != summary["remaining_us"]
            or any(not stages[k]["receipt"] for k in required)
            or any(a["event"] != "COMMIT" for row in stages.values() for a in row["attempts"])
        ):
            raise ValueError(
                "completed recovery differs from its original grant and required receipts"
            )


def initialization_prescore(copy, qualification, summary, recovery=None):
    raw = base64.b64decode(copy["base64"], validate=True)
    member = "ordinary-initialization-qualification.json"
    if len(raw) > 1 << 20 or hashlib.sha256(raw).hexdigest() != copy["sha256"]:
        raise ValueError("external pre-score copy differs")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if (
            archive.namelist() != [member]
            or archive.getinfo(member).compress_type != zipfile.ZIP_STORED
        ):
            raise ValueError("unexpected pre-score archive membership")
        q = json.loads(archive.read(member))

    if qualification is not None:
        with Path(qualification).open("rb") as original:
            if original.read(len(raw) + 1) != raw:
                raise ValueError("independently retained pre-score archive differs")
    canonical = io.BytesIO()
    with zipfile.ZipFile(canonical, "w") as archive:
        entry = zipfile.ZipInfo(member)
        entry.create_system = 3
        entry.external_attr = 0o600 << 16
        archive.writestr(
            entry, json.dumps(q, sort_keys=True, allow_nan=False, separators=(",", ":")).encode()
        )
    if raw != canonical.getvalue():
        raise ValueError("alternate pre-score archive encoding")

    encoded = json.dumps(
        {k: v for k, v in q.items() if k != "identity"},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    if (
        q["identity"] != "sha256:" + hashlib.sha256(encoded).hexdigest()
        or q["version"] != INITIALIZATION
        or q["scope"] != "ordinary_initialization"
        or q["candidate"] != summary["candidate"]
        or q["grant"] != summary["grant"]
        or measurement(q["target"]) != 0
        or q["score_authority"] is not None
        or measurement(q["total_us"]) != measurement(summary["grant_minutes"]) * 60_000_000
    ):
        raise ValueError("pre-score archive differs from the original initialization grant")
    image = (recovery or {}).get("after", {}).get("operations", {}).get("build_runtime", {}).get("image_build")
    if image is not None and image["image"] != q["image"]:
        raise ValueError("pre-score image differs from the paid build")
    return q


def verify_initialization_zip(path, qualification=None):
    summary, files = diagnostic_zip_members(path, INITIALIZATION)
    if (
        summary.get("research_run_requested") is not False
        or summary.get("scientific_verdict") is not None
    ):
        raise ValueError("expected an initialization-only diagnostic")
    launch = initialization_margins(files["launch.json"]) if "launch.json" in files else {}
    recovery = files.get("recovery.json")
    if recovery is not None:
        check_initialization_history(summary, recovery)
    if summary["status"] != "COMPLETE":
        if summary["status"] not in {"PARTIAL", "UNVERIFIED"} or not summary.get(
            "errors", summary.get("checks")
        ):
            raise ValueError("missing initialization failure disposition")
        failure = files.get("failure.json", {})
        if "elapsed_us" in failure or "remaining_authority" in failure:
            if (
                not re.fullmatch("sha256:[a-f0-9]{64}", summary.get("grant", ""))
                or measurement(failure.get("elapsed_us"))
                + measurement(failure.get("remaining_authority"))
                != measurement(summary["grant_minutes"]) * 60_000_000
            ):
                raise ValueError("failed operation differs from its original grant balance")
            if (
                recovery
                and "after" in recovery
                and recovery["after"]["elapsed_us"] != failure["elapsed_us"]
            ):
                raise ValueError("failed action disagrees with the original journal")
        prescore = None
        if "prescore.json" in files:
            if recovery and not recovery.get("after", {}).get("operations", {}).get(
                "publish_prescore", {}
            ).get("receipt"):
                raise ValueError(
                    "the returned pre-score archive requires the original publication receipt"
                )
            prescore = initialization_prescore(files["prescore.json"], qualification, summary, recovery)[
                "identity"
            ]
        elif qualification is not None:
            raise ValueError(
                "an independently retained pre-score archive requires a returned commitment"
            )
        return dict(
            status=summary["status"],
            errors=summary.get("errors", summary.get("checks")),
            grant=summary.get("grant"),
            failure=files.get("failure.json"),
            **({"prescore": prescore} if prescore else {}),
            **({"recovery": recovery} if recovery else {}),
            **launch,
        )
    if (
        set(files) - {"recovery.json"}
        != {"prescore.json", "cut.json", "verification.json", "launch.json"}
        or summary["errors"]
    ):
        raise ValueError("incomplete initialization archive")
    q = initialization_prescore(files["prescore.json"], qualification, summary, recovery)

    def identity(value):
        return (
            "sha256:"
            + hashlib.sha256(
                json.dumps(
                    value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
                ).encode()
            ).hexdigest()
        )

    def sealed(row):
        return row["identity"] == identity({k: v for k, v in row.items() if k != "identity"})

    cut, checked, options = (files[k + ".json"] for k in ("cut", "verification", "launch"))
    quote = q["quote"]
    if (
        not sealed(q)
        or not sealed(quote)
        or q["version"] != INITIALIZATION
        or q["scope"] != "ordinary_initialization"
        or q["candidate"] != summary["candidate"]
        or q["grant"] != summary["grant"]
        or q["target"] != 0
        or q["score_authority"] is not None
        or q["graph"] != identity(q["looks"])
        or q["supported"] != sorted(set(q["trials"]) - set(q["omitted"]))
        or "00" not in q["supported"]
        or set(q["trials"]) != {f"{n:02}" for n in range(7)}
        or not set(q["omitted"]) <= set(q["trials"])
        or cut["weights"] != q["initial_weights"]
        or any(
            not re.fullmatch("sha256:[0-9a-f]{64}", v)
            for v in (
                cut["checkpoint"],
                cut["receipt"],
                cut["state"],
                q["wheel"],
                q["source"],
                q["image"],
                q["physical"],
            )
        )
        or any(
            type(v) is not int or v <= 0
            for v in [
                cut["allocation"]["bytes"],
                cut["allocation"]["inodes"],
                cut["peak_device_bytes"],
                *cut["clocks"]["save"],
                *cut["clocks"]["load"],
            ]
        )
        or sum(cut["clocks"]["save"] + cut["clocks"]["load"]) > cut["lease_us"]
        or not 0 < quote["expected_us"] <= quote["tail_us"]
        or min(summary["remaining_us"], summary["elapsed_us"]) < 0
        or not sealed(cut)
        or cut["identity"] != checked["public_report"]
        or cut["version"] != "round029-ordinary-initialization-report/v1"
        or (cut["elapsed_us"], cut["remaining_us"])
        != (summary["elapsed_us"], summary["remaining_us"])
        or cut["plan"] != q["plan"]
        or checked["report_ref"] != cut["receipt"]
        or checked["mechanical_status"] != "VERIFIED"
        or checked["verification_scope"] != "private_ordinary_initialization"
        or cut["prescore"] != q["identity"]
        or (cut["tokens"], cut["updates"]) != (0, 0)
        or cut["restores"] != [cut["state"]] * 2
        or len(cut["clocks"]["save"]) != 1
        or len(cut["clocks"]["load"]) != 2
        or not 0 < cut["lease_us"] <= summary["elapsed_us"]
        or not q["total_us"]
        == summary["grant_minutes"] * 60_000_000
        == summary["elapsed_us"] + summary["remaining_us"]
        or quote["followup_reserve_us"] != options["followup_reserve_us"]
        or options["research_authority"] is not None
        or not cut["capacity"]
        or any(
            m["deficit_" + u] or m["available_" + u] < m["required_" + u] + m["reserved_" + u]
            for m in cut["capacity"]
            for u in ("bytes", "inodes")
        )
    ):
        raise ValueError("pre-score inputs, paid zero cut or independent verification differ")
    return dict(
        status="COMPLETE",
        prescore=q["identity"],
        registration=q["registration"],
        receipt=cut["receipt"],
        allocation=cut["allocation"],
        restores=len(cut["restores"]),
        lease_us=cut["lease_us"],
        research_dispatches=0,
        **launch,
    )


def pinned_source(root, commit):
    helper = subprocess.check_output(["git", "show", commit + ":" + CACHE_HELPER], cwd=root)
    return Path(subprocess.check_output(
        [sys.executable, "-I", "-S", "-B", "-", str(root), commit], input=helper, cwd=root
    ).decode().strip())


def main(revision, minutes, mode="donation", supplied=None, recover_grant=None):
    if mode not in MODES:
        raise SystemExit("Unknown continuation qualification mode.")
    if mode == "conditioning" and not re.fullmatch("sha256:[a-f0-9]{64}", supplied or ""):
        raise SystemExit("Conditioning requires the original bank receipt.")
    if mode == "bank-prices" and not re.fullmatch("sha256:[a-f0-9]{64}", supplied or ""):
        raise SystemExit("Bank pricing requires the original isolated control receipt.")
    root = Path.cwd().resolve()
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise SystemExit("Supply a full candidate commit.")
    if mode == "control-history":
        if minutes != "0":
            raise SystemExit("Control history is read-only and takes exactly zero grant minutes.")
    elif not re.fullmatch(r"[1-9][0-9]*", minutes):
        raise SystemExit("Supply a positive maximum number of minutes.")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root).decode().strip()

    recovery = None
    if recover_grant is not None:
        original, grant_ref = recover_grant
        if (
            mode not in {"conditioning", "initialization"}
            or not re.fullmatch("[a-f0-9]{40}", original)
            or not re.fullmatch("sha256:[a-f0-9]{64}", grant_ref)
            or git("rev-parse", "--verify", revision + "^{commit}") != revision
            or subprocess.run(
                ["git", "merge-base", "--is-ancestor", original, revision], cwd=root
            ).returncode
        ):
            raise SystemExit(
                "Recovery requires the original worker commit, grant and its pinned descendant reader."
            )
        recovery = dict(reader=revision, grant=grant_ref)
        # The worker, source mounts, grants and study bindings keep the original revision.
        revision = original
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

    source = pinned_source(root, revision)
    resume_init = recovery is not None and mode == "initialization"
    reader = pinned_source(root, recovery["reader"]) if resume_init else source
    sys.path.insert(0, str(reader / "src"))
    if mode == "review-children":
        return saved_children(root, source, revision, int(minutes))

    def action(name, *args, at=revision):
        if resume_init and name in {"qualify", "verify"}:
            return checked_json(
                [
                    "bash",
                    str(reader / LAUNCHER),
                    recovery["reader"],
                    name,
                    *map(str, args),
                    revision,
                ],
                root,
                name,
            )
        return checked_json(["bash", str(source / LAUNCHER), at, name, *map(str, args)], root, name)

    if mode == "initialization":
        return donor_initialization(action, root, revision, minutes, recovery=recovery)
    if mode == "launch-prices":
        return donor_launch_prices(action, root, revision, minutes, supplied)
    if mode == "isolated-controls":
        return donor_isolated_controls(action, root, source, revision, minutes)
    if mode == "conditioning":
        report = donor_isolated_controls(action, root, source, revision, minutes,
                                         bank_receipt=supplied, recovery=recovery)
        if recovery is not None:
            # Keep the historical runtime, verifier and plan above. This independent
            # read opens the newer cost reader even if an original search stayed partial.
            try:
                info = action("preflight", at=recovery["reader"])["inventory"]
                prep = info["prices"]["launch_preparation"]
                original = report["files"]["preflight.json"]
                if (info["gpu_dispatches"] != 0 or prep["checkpoint_receipt"]
                        != original["supported_wave"]["checkpoint_receipt"]):
                    raise ValueError("launch preflight requires the original paid checkpoint")
                original["launch_preparation"] = prep
            except (KeyError, OSError, ValueError, SystemExit) as exc:
                report["launch_preflight_error"] = type(exc).__name__
                print(f"launch preflight: {exc}", file=sys.stderr)
        return report
    if mode == "bank-prices":
        return donor_isolated_controls(action, root, source, revision, minutes, previous=supplied)
    if mode == "control-history":
        return donor_control_history(action, root, revision)
    if mode == "readiness":
        return donor_readiness(action, root, source, revision, minutes, supplied)
    if mode == "checkpoints":
        return donor_checkpoints(action, root, source, revision, minutes)
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


def conditioning_prices(files):
    result = dict(
        successful_role_us=files.get("models.json", {}).get("elapsed_us", 0),
        groups={},
        proxy_four_treatment_paired_sample_us=None,
        full_wave_us=None,
        basis="Eight exposed families per pair. The historical ordinary 4M common and full costs are proxies for 04/05/06. D is priced once.",
    )
    for group in CONDITIONING_GROUPS:
        search = files.get("search-" + group + ".json")
        scores = files.get("paired-" + group + ".json")
        if search:
            result["successful_role_us"] += search["elapsed_us"]
        if not scores:
            continue
        result["successful_role_us"] += scores["elapsed_us"]
        modes = {
            m: dict(
                search_us=search["modes"][m]["elapsed_us"],
                selected_us=scores["modes"][m]["selected_us"],
                bank_us=scores["modes"][m]["bank_us"],
            )
            for m in CONDITIONING_MODES
        }
        total = scores["elapsed_us"] + search["elapsed_us"]
        overhead = total - sum(sum(v.values()) for v in modes.values())
        if overhead < 0:
            raise ValueError("paired measurements exceed the nonoverlapping supervised roles")
        result["groups"][group] = dict(
            modes=modes, shared_overhead_us=overhead, supervised_us=total
        )
    result["proxy_four_treatment_paired_sample_us"] = paired_expected(files, result, 4)
    return result


def paired_expected(files, paired, learner_count):
    if set(paired["groups"]) == set(CONDITIONING_GROUPS):
        return files["models.json"]["elapsed_us"] + sum(
            (1 if group == "D-full" else learner_count) * r["supervised_us"]
            for group, r in paired["groups"].items()
        )


def launch_estimate(files):
    """Combine independently reopened prices for the funded learner roster."""
    preflight = files["preflight.json"]
    wave = preflight["supported_wave"]
    checkpoint = files["checkpoint-verification.json"]
    bank = files["previous-prices.json"]["projection"]
    controls = bank["known_wave"]
    methods = wave["methods"]
    if (methods != sorted(set(methods)) or not {"00", "04", "05", "06"} <= set(methods) <= {f"{n:02}" for n in range(7)}
            or checkpoint["mechanical_status"] != "VERIFIED"
            or checkpoint["report_ref"] != wave["checkpoint_receipt"]):
        raise ValueError("verified checkpoint prices required for the supported recipes")
    if (set(wave["missing"]) != {"conventional_cpu_controls", *controls["missing"]}
            or any(controls["known_" + k] != wave["known_" + k] + controls["control_" + k]
                   for k in ("expected_us", "tail_us"))):
        raise ValueError("paid control subtotal differs from its supported wave")
    missing = set(controls["missing"])
    items = {"screen_and_conventional_controls": {
        k: controls["known_" + k] for k in ("expected_us", "tail_us")}}

    def add(name, expected, tail, covers):
        if not set(covers) <= missing:
            raise ValueError("launch cost already included: " + name)
        if type(expected) is not int or type(tail) is not int or not 0 <= expected <= tail:
            raise ValueError("invalid launch cost for " + name)
        items[name] = dict(expected_us=expected, tail_us=tail)
        missing.difference_update(covers)

    paired = conditioning_prices(files)
    expected = paired_expected(files, paired, len(methods))
    if expected is not None:
        # Both modes already include their full-bank pass. Use a dedicated paired
        # block per profile, with no credit for unproven reuse of a main-panel search.
        tail = expected
        for g in CONDITIONING_GROUPS:
            repetitions = 1 if g == "D-full" else len(wave["methods"])
            modes = files["paired-" + g + ".json"]["modes"]
            for mode, row in modes.items():
                selected, scored = row["selected_cells"], row["bank_cells"]
                strata = {"parameter", "composition", "unseen_structure", "oeis"}
                if selected.keys() != strata or scored.keys() != strata:
                    raise ValueError("incomplete paired price strata")
                for cells, total, peak, wall in (
                    (selected, "search_us", "max_search_us",
                     files["search-" + g + ".json"]["modes"][mode]["elapsed_us"]),
                    (selected, "score_us", "max_score_us", row["selected_us"]),
                    (scored, "scoring_us", "max_us", row["bank_us"]),
                ):
                    if (sum(c[total] for c in cells.values()) > wall
                            or any(c["cases"] != 2 or type(c[peak]) is not int or
                                   type(c[total]) is not int or not 0 <= c[peak] <= c[total] <= 2*c[peak]
                                   for c in cells.values())):
                        raise ValueError("paired case and enclosing role clocks disagree")
                    tail += repetitions * sum(2*c[peak] - c[total] for c in cells.values())
        add("paired_search_selected_and_banks", expected, tail,
            ["matched_full_bank_scoring", "paired_conditioning_sample"])
    elif bank["matched_sample_expected_us"] is not None and bank["matched_sample_tail_us"] is not None:
        add("standalone_matched_banks", bank["matched_sample_expected_us"],
            bank["matched_sample_tail_us"], ["matched_full_bank_scoring"])

    preparation = preflight.get("launch_preparation")
    if preparation:
        if (preparation["checkpoint_receipt"] != wave["checkpoint_receipt"]
                or preparation["fresh_receipt"] != wave["fresh_receipt"]):
            raise ValueError("launch preparation and paid wave have different parents")
        trajectory = preparation["trajectory"]
        if trajectory:
            add("extra_trajectory_overhead", trajectory["known_expected_us"], trajectory["known_tail_us"],
                ["trajectory_startup_extra_checkpoints_and_drain"])
            missing.update(trajectory["missing"])
        publication = preparation["publication_proxy"]
        verified = preparation.get("verification_and_publication")
        if verified:
            add("projected_private_replay_and_publication", verified["expected_us"], verified["tail_us"],
                ["verification_and_publication"])
        elif publication:
            add("historical_scale_publication_and_read", publication["expected_us"], publication["tail_us"],
                ["verification_and_publication"])
            missing.add("round029_private_verification_and_report_scale")

    expected = sum(p["expected_us"] for p in items.values())
    tail = sum(p["tail_us"] for p in items.values())
    reserve = preflight["followup_reserve_us"]
    return dict(methods=wave["methods"], items=items, known_expected_us=expected,
                known_tail_us=tail, missing=sorted(missing), followup_reserve_us=reserve,
                expected_us=None if missing else expected, tail_us=None if missing else tail,
                tail_with_reserve_us=None if missing or reserve is None else tail + reserve,
                basis="Cached workstation. Each dedicated conditioning block includes selected and full-bank scoring. "
                      "The observed tail repeats the slower case in each exposed pair at its measured role overhead.",
                provisional_publication=(preparation or {}).get("verification_and_publication"),
                research_authority=None)


def verify_conditioning_zip(path):
    def identity(record):
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        return "sha256:" + hashlib.sha256(raw).hexdigest()

    summary, files = diagnostic_zip_members(path, CONDITIONING)
    if summary["status"] not in {"COMPLETE", "PARTIAL"}:
        return dict(status=summary["status"], verified_members=len(files))
    receipt_scope = set(CONDITIONING_GROUPS)
    proofs = files["verification.json"]
    searches = {g: files["search-" + g + ".json"] for g in receipt_scope
                if "search-" + g + ".json" in files}
    scores = {g: files["paired-" + g + ".json"] for g in receipt_scope
              if "paired-" + g + ".json" in files}
    complete = summary["status"] == "COMPLETE"
    work = summary["charged_work"]
    clocks = ("grant_minutes", "elapsed_us", "remaining_us", "verified_leases", "hip_attempts",
              "recorded_lease_us")
    if (
        any(type(summary[k]) is not int or summary[k] < 0 for k in clocks)
        or set(work) != {"tokens", "draws", "rejections", "attempts", "native_work",
                         "elapsed_us", "disk_bytes", "archive_bytes"}
        or any(type(n) is not int or n < 0 for n in work.values())
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", summary["grant"])
        or summary["grant_minutes"] < 1
        or work["tokens"] != 0
        or work["elapsed_us"] != summary["elapsed_us"]
        or summary["elapsed_us"] + summary["remaining_us"] != summary["grant_minutes"] * 60_000_000
        or not len(searches) <= summary["verified_leases"] <= summary["hip_attempts"]
        or bool(summary["verified_leases"]) != bool(summary["recorded_lease_us"])
        or summary["recorded_lease_us"] > summary["elapsed_us"]
        or summary["gpu_dispatches"] != (summary["verified_leases"]
                if summary["verified_leases"] == summary["hip_attempts"] else None)
    ):
        raise ValueError("conditioning leases or cumulative grant balance differ")
    search_refs = {g: row["receipt"] for g, row in searches.items()}
    barrier = identity(dict(plan=proofs["plan"], searches=search_refs)) if scores else None
    if (
        proofs["research_dispatches"] != 0
        or proofs["verified"].keys() != scores.keys()
        or (scores and searches.keys() != receipt_scope)
        or proofs["sealed_searches"] != barrier
        or complete != (scores.keys() == receipt_scope and not summary["errors"])
        or (not complete and (summary.get("receipt") or summary.get("recovery_receipt")))
    ):
        raise ValueError("conditioning status, paired scoring barrier or coverage differs")
    if (any(type(s["lease_us"]) is not int or not 0 < s["lease_us"] <= s["elapsed_us"]
            for s in searches.values())
            or sum(s["lease_us"] for s in searches.values()) > summary["recorded_lease_us"]):
        raise ValueError("verified search clocks exceed the recorded leases")
    prices = conditioning_prices(files)
    if prices != files["forecast.json"] or prices["successful_role_us"] > summary["elapsed_us"]:
        raise ValueError("paired cost projection or supervision differs")
    groups = {}
    refs = set(search_refs.values())
    for group, score in scores.items():
        search = searches[group]
        if (
            score["group"] != group
            or search["receipt"] != score["search"]
            or search["native"] != score["native"]
            or score["barrier"] != barrier
            or proofs["verified"][group] != {"search": score["search"], "score": score["score"]}
            or sum(score["paired_selected"].values()) != files["panel.json"]["cases"]
        ):
            raise ValueError("returned conditioning coverage or original role links differ")
        refs.add(score["score"])
        groups[group] = identity({k: v for k, v in score.items() if k != "identity"})
        if groups[group] != score["identity"]:
            raise ValueError("paired group seal differs")
        for mode in CONDITIONING_MODES:
            cells = score["modes"][mode]["bank_cells"]
            if len(cells) != 4 or any(v["cases"] != 2 for v in cells.values()):
                raise ValueError("eight matched cases required in both input modes")
    if len(refs) != len(searches) + len(scores):
        raise ValueError("distinct original role receipts required")
    if complete:
        receipt = identity(
            dict(
                plan=proofs["plan"],
                grant=summary["grant"],
                groups=groups,
                charged=summary["charged_work"],
                leases=summary["verified_leases"],
                lease_us=summary["recorded_lease_us"],
            )
        )
        if receipt != summary["receipt"]:
            raise ValueError("paired receipt differs from the original grant and group records")
        if "recovery.json" in files or "recovery_receipt" in summary:
            repair = files["recovery.json"]
            if (
                repair["grant"] != summary["grant"]
                or repair["worker_candidate"] != summary["candidate"]
                or not re.fullmatch("[0-9a-f]{40}", repair["reader"])
                or repair["additional_hip_attempts"] != 0
                or repair["original_hip_attempts"] != summary["hip_attempts"]
                or repair["failure_checks"]
                or repair["observations"]
                or {g: v["search"] for g, v in repair["groups"].items()} != search_refs
                or summary["recovery_receipt"]
                != identity(dict(result=receipt, recovery=identity(repair)))
            ):
                raise ValueError(
                    "conditioning recovery grant, search count or reader receipt differs"
                )
            reviewed = []
            for group, row in repair["groups"].items():
                if row["source"] == "rechecked":
                    reviewed.append(row["reader_receipt"])
                    if (
                        not re.fullmatch("sha256:[0-9a-f]{64}", row["reader_receipt"])
                        or row["original_charge_us"]
                        < files["search-" + group + ".json"]["elapsed_us"]
                    ):
                        raise ValueError("recovery reader or original HIP debit differs")
                elif row["source"] != "completed":
                    raise ValueError("unknown recovered search evidence")
            if len(set(reviewed)) != len(reviewed):
                raise ValueError("distinct original search readers required")
    return dict(status=summary["status"], verified_members=len(files))


def diagnostic_zip_members(path, scope):
    """Read the diagnostic ZIP without the donor, installed package, or permission to dispatch."""
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (
            len(names) != len(set(names))
            or len(names) > 16
            or any(
                not re.fullmatch(r"[A-Za-z0-9_-]+\.json", n)
                or archive.getinfo(n).file_size > 2 << 20
                for n in names
            )
        ):
            raise ValueError("unexpected bank ZIP members")
        summary = json.loads(archive.read("summary.json"))
        if summary["scope"] != scope or set(names) != {"summary.json", *summary["sha256"]}:
            raise ValueError("bank ZIP scope or manifest differs")
        parts = {}
        for name, checksum in summary["sha256"].items():
            raw = archive.read(name)
            if hashlib.sha256(raw).hexdigest() != checksum:
                raise ValueError("bank ZIP member digest differs")
            parts[name] = json.loads(raw)
    if (
        summary["status"] not in {"COMPLETE", "PARTIAL", "BLOCKED", "UNVERIFIED"}
        or summary["research_run_requested"] is not False
        or summary["scientific_verdict"] is not None
    ):
        raise ValueError("unexpected scientific status in a diagnostic ZIP")
    return summary, parts


def verify_bank_zip(path):
    summary, parts = diagnostic_zip_members(path, BANK_PRICES)
    if summary["status"] == "COMPLETE":
        expected = {
            arm + "-" + look for arm in ("ordinary", "first_divergence") for look in ("one", "full")
        }
        expected.add("incumbent-full")
        verified = parts["verification.json"]["verified"]
        forecast = parts["forecast.json"]
        supervised = 0
        for name in expected:
            block = parts[name + ".json"]
            if (
                block["block"] != name
                or block["receipt"] != verified[name]
                or len(block["cells"]) != 4
                or any(c["cases"] != 2 for c in block["cells"].values())
            ):
                raise ValueError("returned bank coverage differs from the original role receipts")
            supervised += block["elapsed_us"]
        if (
            set(verified) != expected
            or len(set(verified.values())) != len(expected)
            or summary["errors"]
            or parts["verification.json"]["research_dispatches"] != 0
            or summary["gpu_dispatches"] != 0
            or summary["charged_work"]["tokens"] != 0
            or not 0 < supervised <= summary["elapsed_us"]
            or forecast["observed_sample_us"] != supervised
            or parts["paid-controls.json"]["projection"] != forecast["known_wave"]
            or summary["elapsed_us"] != summary["charged_work"]["elapsed_us"]
            or summary["elapsed_us"] + summary["remaining_us"]
            != summary["grant_minutes"] * 60_000_000
        ):
            raise ValueError("bank publication, verified costs or grant balance disagree")
    return dict(status=summary["status"], verified_members=len(parts))


def write_readiness_zip(out, report):
    def encoded(value):
        return (json.dumps(value, indent=2, allow_nan=False) + "\n").encode("utf-8")

    report = dict(report)
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
        choices=MODES,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="save the result to a new file (ZIP for readiness, checkpoints and control reports; JSON for other modes)",
    )
    parser.add_argument("--from-result", type=Path, help="fresh-start JSON for readiness or the original conditioning ZIP for launch-prices")
    parser.add_argument("--bank-receipt", help="completed matched bank qualification receipt for conditioning")
    parser.add_argument("--recover-grant", nargs=2, metavar=("ORIGINAL_COMMIT","GRANT"),
                        help="resume conditioning or ordinary initialization on its original grant and worker")
    parser.add_argument("--controls-receipt", help="original isolated control receipt identity for bank pricing")
    args = parser.parse_args(argv)
    from_file = args.mode in {"readiness", "launch-prices"}
    zipped = args.mode in ZIP_SCOPES
    if zipped and (not args.output or args.output.suffix.lower() != ".zip"):
        parser.error(f"{args.mode} requires --output NEW.zip")
    if from_file and args.from_result is None:
        parser.error(f"{args.mode} requires --from-result INPUT")
    if args.from_result is not None and not from_file:
        parser.error("--from-result is only valid with readiness or launch-prices")
    if args.mode == "conditioning":
        if not re.fullmatch("sha256:[a-f0-9]{64}", args.bank_receipt or ""):
            parser.error("conditioning requires --bank-receipt sha256:ORIGINAL_RECEIPT")
    elif args.bank_receipt is not None:
        parser.error("--bank-receipt is only valid with conditioning")
    if args.recover_grant is not None and (args.mode not in {"conditioning", "initialization"}
            or not re.fullmatch("[0-9a-f]{40}", args.recover_grant[0])
            or not re.fullmatch("sha256:[0-9a-f]{64}", args.recover_grant[1])):
        parser.error("--recover-grant requires conditioning or initialization, a full original commit and a grant identity")
    if args.mode == "bank-prices":
        if not re.fullmatch("sha256:[a-f0-9]{64}", args.controls_receipt or ""):
            parser.error("bank-prices requires --controls-receipt sha256:ORIGINAL_RECEIPT")
    elif args.controls_receipt is not None:
        parser.error("--controls-receipt is only valid with bank-prices")
    if args.mode in {"supply", "timings", "broader", "fresh-start"} and args.output is None:
        parser.error("Donor preparation requires an output file (--output PATH)")
    request = (args.revision, args.minutes, args.mode)
    if from_file:
        request += (args.from_result,)
    elif args.mode == "conditioning":
        request += (args.bank_receipt,)
    elif args.mode == "bank-prices":
        request += (args.controls_receipt,)
    if args.recover_grant is not None:
        if args.mode == "initialization":
            request += (None,)
        request += (args.recover_grant,)
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
            if zipped:
                print(f"{args.mode}: {exc}", file=sys.stderr)
                report.update(scope=ZIP_SCOPES[args.mode], research_run_requested=False,
                              scientific_verdict=None, checks={"bootstrap": "FAILED"})
            else:
                report["reason"] = str(exc)
            code = 1
        if zipped:
            write_readiness_zip(out, report)
        else:
            out.write((json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8"))
        out.flush()
        os.fsync(out.fileno())
        if args.mode == "conditioning":
            try:
                verify_conditioning_zip(out.name)
            except (KeyError, TypeError, ValueError, zipfile.BadZipFile) as exc:
                print(f"conditioning ZIP verification: {exc}", file=sys.stderr)
                report.update(status="UNVERIFIED", receipt=None,
                              errors=dict(report.get("errors", {}), publication=type(exc).__name__))
                out.seek(0)
                out.truncate()
                write_readiness_zip(out, report)
                out.flush()
                os.fsync(out.fileno())
                verify_conditioning_zip(out.name)
                code = 1
        elif args.mode == "bank-prices":
            verify_bank_zip(out.name)
        elif args.mode == "initialization":
            try:
                verify_initialization_zip(out.name)
            except (KeyError, TypeError, ValueError, OSError, zipfile.BadZipFile) as exc:
                print(f"initialization ZIP verification: {exc}", file=sys.stderr)
                # Preserve each independently checkable diagnostic after a publication error.
                available = report.get("files", {})
                report.update(status="UNVERIFIED", files={}, errors=dict(
                    report.get("errors", {}), publication=type(exc).__name__))
                for member in ("launch.json", "recovery.json", "failure.json", "prescore.json"):
                    if member not in available:
                        continue
                    report["files"][member] = available[member]
                    probe = io.BytesIO()
                    write_readiness_zip(probe, report)
                    try:
                        verify_initialization_zip(probe)
                    except (KeyError, TypeError, ValueError, OSError, zipfile.BadZipFile):
                        del report["files"][member]
                out.seek(0)
                out.truncate()
                write_readiness_zip(out, report)
                out.flush()
                os.fsync(out.fileno())
                verify_initialization_zip(out.name)
                code = 1
        elif args.mode == "launch-prices":
            verify_launch_zip(out.name)
        os.link(out.name, args.output)
    print(f"Saved {args.output.resolve()}", file=sys.stderr)
    return code


def require_donor_root(root):
    if (
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        != str(root)
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != DONOR
        or subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"], stdout=subprocess.DEVNULL
        ).returncode
        != 1
    ):
        raise SystemExit("Run at the physical root of the detached round027 donor.")


def resume_initialization(revision, *, directory=None):
    """One donor handoff for the returned 30-minute grant, including a private terminal log."""
    if not re.fullmatch("[a-f0-9]{40}", revision):
        raise SystemExit("Supply the full recovery reader commit.")
    root = Path.cwd().resolve()
    require_donor_root(root)
    operator = "specs/016-measure-learning-velocity/validation/qualify_continuation.py"
    script = subprocess.check_output(["git", "show", revision + ":" + operator], cwd=root)
    os.umask(0o077)
    combined = directory is not None
    if directory is None:
        directory = Path(tempfile.mkdtemp(prefix="round029-init-recovery-", dir=Path.home()))
    reader = directory / "reader.py"
    reader.write_bytes(script)
    result, log = directory / "initialization.zip", directory / "operator.log"
    print(f"Private terminal output: {log}", file=sys.stderr, flush=True)
    with log.open("xb") as stream:
        proc = subprocess.run(
            [
                sys.executable,
                "-I",
                "-S",
                "-B",
                str(reader),
                revision,
                "30",
                "initialization",
                "--recover-grant",
                INITIAL_WORKER,
                INITIAL_GRANT,
                "--output",
                str(result),
            ],
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    if not result.is_file():
        raise SystemExit(
            f"Recovery exited {proc.returncode} before publishing the ZIP. Inspect {log}"
        )
    checked = verify_initialization_zip(result)
    if not combined:
        print(f"{checked['status']}. Return {result}", file=sys.stderr, flush=True)
    return proc.returncode


def copied_zip(path):
    with Path(path).open("rb") as stream:
        raw = stream.read((1 << 20) + 1)
    if len(raw) > 1 << 20:
        raise ValueError("trial preparation public archive exceeds its transport limit")
    return dict(sha256=hashlib.sha256(raw).hexdigest(), base64=base64.b64encode(raw).decode())


TRIAL_DIAGNOSES = {
    "unknown object member": "schema_unknown_member",
    "registered trial inputs or checkpoint targets differ": "trial_inputs_changed",
    "original launch quote no longer reconciles": "quote_changed",
    "No such file or directory": "missing_file",
    "updating the original workspace": "owner_active",
    "original dispatch debit missing": "missing_debit",
}


def verify_trial_preparation_zip(path):
    return checked_trial_preparation(*diagnostic_zip_members(path, TRIAL_PREPARATION))


def checked_trial_preparation(summary, files):
    provenance = {}
    if "readback" in summary:
        readback = summary["readback"]
        if (
            not isinstance(readback, dict)
            or set(readback) - {"reader", "input_sha256", "prior_failure"}
            or not re.fullmatch("[0-9a-f]{40}", readback.get("reader", ""))
            or not re.fullmatch("[0-9a-f]{64}", readback.get("input_sha256", ""))
        ):
            raise ValueError("invalid trial readback provenance")
        prior = readback.get("prior_failure")
        if prior is not None and (
            not isinstance(prior, dict)
            or set(prior) != {"category", "reason_sha256"}
            or prior["category"] not in {*TRIAL_DIAGNOSES.values(), "unclassified"}
            or not re.fullmatch("[0-9a-f]{64}", prior["reason_sha256"])
        ):
            raise ValueError("invalid historical trial diagnosis")
        provenance["readback"] = readback

    def unpack(name):
        item = files[name]
        raw = base64.b64decode(item["base64"], validate=True)
        if (
            set(item) != {"base64", "sha256"}
            or len(raw) > 1 << 20
            or hashlib.sha256(raw).hexdigest() != item["sha256"]
        ):
            raise ValueError("trial preparation archive copy differs")
        return io.BytesIO(raw)

    def sealed_member(name, member):
        stream = unpack(name)
        with zipfile.ZipFile(stream) as archive:
            if (
                archive.namelist() != [member]
                or archive.getinfo(member).file_size > 1 << 20
                or archive.getinfo(member).compress_type != zipfile.ZIP_STORED
            ):
                raise ValueError("unexpected trial preparation member")
            data = json.loads(archive.read(member))

        def encode(value):
            return json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()

        if (
            data["identity"]
            != "sha256:"
            + hashlib.sha256(encode({k: v for k, v in data.items() if k != "identity"})).hexdigest()
        ):
            raise ValueError("trial preparation record identity differs")
        canonical = io.BytesIO()
        with zipfile.ZipFile(canonical, "w") as archive:
            entry = zipfile.ZipInfo(member)
            entry.create_system = 3
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, encode(data))
        if canonical.getvalue() != stream.getvalue():
            raise ValueError("noncanonical trial preparation archive")
        return data

    start = unpack("initialization.json")
    checked = verify_initialization_zip(start)
    partial = summary["status"] == "PARTIAL"
    pair = {"qualification.json", "report.json"}
    if partial:
        if (
            not summary["errors"]
            or set(files) - {"initialization.json", "failure.json"} - pair
            or (set(files) & pair) not in (set(), pair)
        ):
            raise ValueError("incomplete trial preparation requires its stopping point")
        progress = dict(
            status="PARTIAL",
            initialization=checked,
            errors=summary["errors"],
            failure=files.get("failure.json"),
            **provenance,
        )
        if not pair <= set(files):
            return progress
    if checked["status"] != "COMPLETE" or (
        not partial
        and (
            summary["status"] != "COMPLETE"
            or summary["errors"]
            or set(files)
            != {"initialization.json", "qualification.json", "report.json", "verification.json"}
        )
    ):
        raise ValueError("missing original zero cut, trial commitment, report or verification")
    init_summary, init_files = diagnostic_zip_members(start, INITIALIZATION)
    original = initialization_prescore(init_files["prescore.json"], None, init_summary)
    q = sealed_member("qualification.json", "supported-wave-qualification.json")
    r = sealed_member("report.json", "supported-wave-preparation-report.json")
    if set(q) != set(
        (
            "identity version scope candidate reader owner preparation_grant initialization initialization_prescore "
            "checkpoint state source image wheel physical role_policy registration trials graph supported omitted seed stream_seed "
            "learners training inputs schedule policy sources expected_us tail_us followup_us proposed_total_us capacity "
            "research_authority assessment"
        ).split()
    ) or set(r) != set(
        (
            "identity version accepted receipt grant total_us "
            "elapsed_us remaining_us charged_tokens research_dispatches scientific_verdict"
        ).split()
    ):
        raise ValueError("unexpected trial preparation fields")
    methods = original["supported"]

    def identity(value):
        return (
            "sha256:"
            + hashlib.sha256(
                json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
        )

    targets = []
    for method in methods:
        worker = q["learners"][method]
        if (
            set(worker) != {"package", "batch", "first", "second"}
            or not re.fullmatch("sha256:[0-9a-f]{64}", worker["package"])
            or measurement(worker["batch"]) != (16 if method in {"04", "06"} else 32)
            or not 0 < measurement(worker["first"]) < measurement(worker["second"])
        ):
            raise ValueError("unqualified trial learner")
        previous = None
        for target in (0, 1_000_000, 4_000_000):
            row = dict(
                registration=original["registration"],
                method=method,
                target=target,
                learner=worker["package"],
                previous=previous,
            )
            previous = identity(row)
            targets.append(dict(row, identity=previous))
    schedule = [
        dict(method=m, profile=p, look=row["look"], roster=row["receipt"], members=row["members"])
        for m in [*methods, *[k for k in original["looks"] if k not in original["trials"]]]
        for p, rows in original["looks"][m].items()
        for row in rows
    ]
    order = lambda row: (row["method"], row["profile"], row["look"])
    if (
        q["version"] != TRIAL_PREPARATION
        or r["version"] != q["version"]
        or q["scope"] != "exploration"
        or q["reader"] != summary["reader"]
        or q["preparation_grant"] != summary["grant"]
        or r["grant"] != summary["grant"]
        or q["research_authority"] is not None
        or q["assessment"] != "sealed"
        or r["scientific_verdict"] is not None
        or measurement(r["charged_tokens"])
        or measurement(r["research_dispatches"])
        or q["initialization"] != checked["receipt"]
        or q["initialization_prescore"] != checked["prescore"]
        or any(
            q[k] != original[k]
            for k in (
                "candidate",
                "source",
                "image",
                "wheel",
                "physical",
                "role_policy",
                "graph",
                "registration",
                "trials",
                "supported",
                "seed",
                "stream_seed",
            )
        )
        or (q["checkpoint"], q["state"])
        != (init_files["cut.json"]["checkpoint"], init_files["cut.json"]["state"])
        or q["training"] != targets
        or q["learners"]["00"]["package"] != original["learner"]
        or any(
            not re.fullmatch("sha256:[0-9a-f]{64}", v)
            for v in (q["inputs"], q["policy"], q["owner"], r["receipt"])
        )
        or set(q["learners"]) != set(methods)
        or q["omitted"] != {m: identity(value) for m, value in original["omitted"].items()}
        or sorted(q["schedule"], key=order) != sorted(schedule, key=order)
        or q["sources"] != original["quote"]["sources"]
        or q["followup_us"] != original["quote"]["followup_reserve_us"]
        or any(q[k] != original["quote"][k] for k in ("expected_us", "tail_us"))
        or q["proposed_total_us"] != q["tail_us"] + q["followup_us"]
        or r["accepted"] != q["identity"]
        or r["total_us"] != measurement(summary["prep_minutes"]) * 60_000_000
        or measurement(r["elapsed_us"]) + measurement(r["remaining_us"]) != r["total_us"]
        or (
            not partial
            and files["verification.json"]
            != dict(private_receipt=r["receipt"], public_report=r["identity"])
        )
    ):
        raise ValueError(
            "supported wave differs from its independent initialization, prices or grants"
        )
    mount_fields = {"device"} | {
        p + "_" + u
        for p in ("available", "required", "reserved", "deficit")
        for u in ("bytes", "inodes")
    }
    mounts = q["capacity"]
    if not mounts or any(set(m) != mount_fields for m in mounts):
        raise ValueError("full wave requires its recorded mount observations")
    margins = mount_margins(mounts)
    if any(m["margin_bytes"] < 0 or m["margin_inodes"] < 0 for m in margins):
        raise ValueError("full wave exceeded its observed capacity")
    publication = dict(
        supported=methods,
        omitted=sorted(q["omitted"]),
        training_targets=len(targets),
        looks=len(schedule),
        initialization=checked["receipt"],
        accepted=q["identity"],
        elapsed_us=r["elapsed_us"],
        remaining_us=r["remaining_us"],
        expected_us=q["expected_us"],
        tail_us=q["tail_us"],
        followup_reserve_us=q["followup_us"],
        proposed_total_us=q["proposed_total_us"],
        capacity=[dict(mount, **margin) for mount, margin in zip(mounts, margins, strict=True)],
        research_authority=None,
    )
    if partial:
        return dict(progress, published=publication)
    return dict(status="COMPLETE", **publication, **provenance)


TRIAL_ERRORS = (
    OSError,
    ValueError,
    KeyError,
    RuntimeError,
    TypeError,
    SystemExit,
    subprocess.SubprocessError,
    zipfile.BadZipFile,
)


def trial_diagnosis(reason):
    # Only opaque fingerprints and fixed categories travel with the public archives.
    category = next((v for k, v in TRIAL_DIAGNOSES.items() if k in reason), "unclassified")
    return dict(category=category, reason_sha256=hashlib.sha256(reason.encode()).hexdigest())


def trial_failure(result, step, exc, log):
    result["errors"][step] = type(exc).__name__
    with log.open("a") as stream:
        print(f"{step}: {exc}", file=stream)
    failure = dict(stage=step, mechanical_status="FAILED")
    reason = str(exc) or type(exc).__name__
    if isinstance(exc, SystemExit):
        try:
            value = json.loads(reason)
        except ValueError:
            value = None
        if isinstance(value, dict):
            reason = str(value.get("reason", reason))
            failure.update(
                {
                    k: value[k]
                    for k in (
                        "stage",
                        "mechanical_status",
                        "elapsed_us",
                        "remaining_authority",
                        "deficit_bytes",
                        "deficit_inodes",
                    )
                    if k in value
                }
            )
    result["files"]["failure.json"] = dict(failure, **trial_diagnosis(reason))


def collect_trial_publication(root, result, verify, *, expected=None):
    from oeis_learn.generalization.progressive import velocity_donation_report as public
    from oeis_learn.generalization.progressive.contracts import load
    from oeis_learn.generalization.progressive.velocity_records import SESSION
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_velocity_donation import retained_qualification
    from oeis_learn.pipeline_validation.progressive_velocity_verify import archive_result

    public.hashes(result["grant"])
    area = confined(
        root, "run-state/discovery/" + SESSION + "/qualification/" + result["grant"][7:]
    )
    grant = load(confined(area, "grant.json"))
    if grant["identity"] != result["grant"]:
        raise ValueError("original supported-wave grant required")
    retained = retained_qualification(root, grant)
    report = confined(area, "share/" + public.REPORT)
    exported, code = archive_result(report, retained)
    receipt = exported["last_committed_ref"]
    if (
        code
        or exported["verification_scope"] != "paired_archives_trial_preparation"
        or (expected is not None and receipt != expected)
    ):
        raise ValueError("independently retained trial commitment differs")
    pair = {"qualification.json": copied_zip(retained), "report.json": copied_zip(report)}
    # Bind the actual bytes to the returned zero cut and paid schedule before exporting them,
    # including when the subsequent private replay stops.
    checked_trial_preparation(
        dict(result, status="PARTIAL", errors={"verification": "pending"}),
        {**result["files"], **pair},
    )
    result["files"].update(pair)
    private = verify()
    if (
        private["verification_scope"] != "private_trial_preparation"
        or private["report_ref"] != receipt
    ):
        raise ValueError("private trial replay differs")
    result["files"]["verification.json"] = dict(
        private_receipt=receipt, public_report=exported["report_ref"]
    )
    result["status"] = "COMPLETE"


def finish_trial_preparation(out, result):
    dest = out / "trial-preparation.zip"
    write_readiness_zip(dest, result)
    checked = verify_trial_preparation_zip(dest)
    if checked["status"] == "PARTIAL" and checked.get("failure"):
        failure = checked["failure"]
        print(
            f"{failure.get('stage', 'preparation')}: {failure.get('category', 'unclassified')}",
            file=sys.stderr,
        )
    print(f"{checked['status']}. Return {dest}", file=sys.stderr, flush=True)
    return 0 if checked["status"] == "COMPLETE" else 1


def prepare_trials(revision, minutes):
    if not re.fullmatch("[a-f0-9]{40}", revision) or not re.fullmatch("[1-9][0-9]*", minutes):
        raise SystemExit(
            "Supply the full reader commit and the cumulative host preparation minutes."
        )
    os.umask(0o077)
    root = Path.cwd().resolve()
    out = Path(tempfile.mkdtemp(prefix="round029-trial-prep-", dir=Path.home()))
    code = resume_initialization(revision, directory=out)
    init = out / "initialization.zip"
    files = {"initialization.json": copied_zip(init)}
    result = dict(
        scope=TRIAL_PREPARATION,
        reader=revision,
        prep_minutes=int(minutes),
        files=files,
        status="PARTIAL",
        errors={},
        research_run_requested=False,
        scientific_verdict=None,
    )
    step = "initialization"
    try:
        if code or verify_initialization_zip(init)["status"] != "COMPLETE":
            raise ValueError("original initialization is partial")
        step = "trial_qualification"
        source = pinned_source(root, revision)
        sys.path.insert(0, str(source / "src"))
        from oeis_learn.generalization.progressive.contracts import load
        from oeis_learn.pipeline_validation import progressive_velocity_trial_preparation as trials

        grant_path = trials.authorize(root, revision, int(minutes))
        grant = load(grant_path)
        result["grant"] = grant["identity"]

        def action(name, *args):
            with (out / "operator.log").open("ab") as log:
                return checked_json(
                    ["bash", str(source / LAUNCHER), revision, name, *map(str, args)],
                    root,
                    name,
                    log=log,
                )

        paid = action("qualify", grant_path)
        if (
            paid["scope"] != "trial_preparation"
            or paid["charged_tokens"]
            or paid["research_dispatches"]
        ):
            raise ValueError("expected funded host-only trial preparation")
        step = "verification"
        collect_trial_publication(
            root, result, lambda: action("verify"), expected=paid["last_committed_ref"]
        )
    except TRIAL_ERRORS as exc:
        trial_failure(result, step, exc, out / "operator.log")
    return finish_trial_preparation(out, result)


def prior_trial_failure(log):
    """Consult the last private terminal result only for a redacted historical diagnosis."""
    try:
        with log.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - (64 << 10)))
            lines = stream.read().splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            value = json.loads(line.removeprefix(b"verification: "))
        except (ValueError, UnicodeDecodeError):
            continue
        if (
            isinstance(value, dict)
            and value.get("action") == "verify"
            and value.get("mechanical_status") == "FAILED"
            and isinstance(value.get("reason"), str)
        ):
            return trial_diagnosis(value["reason"])
    return None


def recover_trial_preparation(revision, checksum, previous=None):
    """Reopen one already published host grant using the exact returned combined ZIP."""
    if not re.fullmatch("[a-f0-9]{40}", revision) or not re.fullmatch("[a-f0-9]{64}", checksum):
        raise SystemExit(
            "Supply the full reader commit and SHA-256 of the returned trial-preparation ZIP."
        )
    root = Path.cwd().resolve()
    require_donor_root(root)
    candidates = (
        [Path(previous)]
        if previous is not None
        else sorted(Path.home().glob("round029-trial-prep-*/trial-preparation.zip"))
    )
    for path in candidates:
        if path.is_file():
            with path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() == checksum:
                    stream.seek(0)
                    summary, files = diagnostic_zip_members(stream, TRIAL_PREPARATION)
                    returned = checked_trial_preparation(summary, files)
                    break
    else:
        raise SystemExit(
            "The original combined ZIP was not found. Supply its path after its SHA-256."
        )
    if (
        returned["status"] == "PARTIAL"
        and returned["initialization"]["status"] != "COMPLETE"
        or not re.fullmatch("sha256:[0-9a-f]{64}", summary.get("grant", ""))
        or not re.fullmatch("[0-9a-f]{40}", summary["reader"])
        or measurement(summary["prep_minutes"]) == 0
    ):
        raise SystemExit(
            "The returned ZIP must identify a completed initialization and its funded trial preparation."
        )
    os.umask(0o077)
    out = Path(tempfile.mkdtemp(prefix="round029-trial-readback-", dir=Path.home()))
    log = out / "operator.log"
    log.touch(mode=0o600)
    print(f"Private terminal output: {log}", file=sys.stderr, flush=True)
    readback = dict(reader=revision, input_sha256=checksum)
    if "verification" in summary["errors"]:
        readback["prior_failure"] = prior_trial_failure(path.with_name("operator.log"))
    result = dict(
        scope=TRIAL_PREPARATION,
        reader=summary["reader"],
        prep_minutes=summary["prep_minutes"],
        grant=summary["grant"],
        readback=readback,
        status="PARTIAL",
        errors={},
        files={"initialization.json": files["initialization.json"]},
        research_run_requested=False,
        scientific_verdict=None,
    )
    try:
        source = pinned_source(root, revision)
        sys.path.insert(0, str(source / "src"))
        from oeis_learn.generalization.progressive.velocity_records import SESSION
        from oeis_learn.pipeline_validation.identity import confined
        from oeis_learn.pipeline_validation.progressive_velocity_verify import private_result

        def verify():
            value, code = private_result(
                root,
                confined(root, "run-state/discovery/" + SESSION),
                source=source,
                trial_grant=result["grant"],
            )
            if code:
                raise SystemExit(json.dumps(value))
            return value

        collect_trial_publication(root, result, verify)
    except TRIAL_ERRORS as exc:
        trial_failure(result, "verification", exc, log)
    return finish_trial_preparation(out, result)


def review_launch(revision, minutes):
    """Resume the original paid recovery and collect the independent CPU prices in one ZIP."""
    if not re.fullmatch("[0-9a-f]{40}", revision) or not re.fullmatch("[1-9][0-9]*", minutes):
        raise SystemExit(
            "Supply the full reader commit and a positive maximum CPU grant in minutes."
        )
    root = Path.cwd().resolve()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root)

    if (
        Path(git("rev-parse", "--show-toplevel").decode().strip()) != root
        or git("rev-parse", "HEAD").decode().strip() != DONOR
        or subprocess.run(
            ["git", "symbolic-ref", "-q", "HEAD"], cwd=root, stdout=subprocess.DEVNULL
        ).returncode
        != 1
    ):
        raise SystemExit("Run at the physical root of the detached round027 donor.")
    worker = "c561004224f380d3c3b246d9b57f013a6559b2ff"
    # The paid search recheck request includes the repair reader identity.
    # Keep its original published identity on every retry, including after pricing changes.
    repair = "404a652c936eb7cfb4ab3303301edc4c9b5ee33c"
    grant = "sha256:ee51d1909fe93ce8928d1b930edb209d0e8582be7a20b776eab712bad539f34c"
    bank = "sha256:fca44fe0dc257caa17f4ea7c9e559891530794fcfba19574397684a79c9c658a"
    operator = "specs/016-measure-learning-velocity/validation/qualify_continuation.py"
    old, current = (git("show", rev + ":" + operator) for rev in (repair, revision))
    os.umask(0o077)
    out = Path(tempfile.mkdtemp(prefix="round029-launch-", dir=Path.home()))
    print(f"Saving this review in {out}", file=sys.stderr, flush=True)
    previous, latest = out / "recover.py", out / "review.py"
    previous.write_bytes(old)
    latest.write_bytes(current)
    recovered, result = out / "conditioning.zip", out / "launch-prices.zip"
    python = [sys.executable, "-I", "-S", "-B"]
    subprocess.run(
        [
            *python,
            str(previous),
            repair,
            "75",
            "conditioning",
            "--bank-receipt",
            bank,
            "--recover-grant",
            worker,
            grant,
            "--output",
            str(recovered),
        ],
        cwd=root,
    )
    # Independent work proceeds even when the original GPU evidence still fails verification.
    priced = subprocess.run(
        [
            *python,
            str(latest),
            revision,
            minutes,
            "launch-prices",
            "--from-result",
            str(recovered),
            "--output",
            str(result),
        ],
        cwd=root,
    )
    if result.is_file():
        subprocess.run(
            [*python, str(latest), "--verify-launch-zip", str(result)], cwd=root, check=True
        )
        print(f"Return {result}", file=sys.stderr, flush=True)
    return priced.returncode


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--prepare-trials":
        sys.exit(prepare_trials(*sys.argv[2:]))
    elif len(sys.argv) in (4, 5) and sys.argv[1] == "--recover-trial-preparation":
        sys.exit(recover_trial_preparation(*sys.argv[2:]))
    elif len(sys.argv) == 3 and sys.argv[1] == "--verify-trial-preparation-zip":
        print(json.dumps(verify_trial_preparation_zip(sys.argv[2]), indent=2))
    elif len(sys.argv) == 3 and sys.argv[1] == "--resume-initialization":
        sys.exit(resume_initialization(sys.argv[2]))
    elif len(sys.argv) == 4 and sys.argv[1] == "--review-launch":
        sys.exit(review_launch(*sys.argv[2:]))
    elif len(sys.argv) in (3, 4) and sys.argv[1] == "--verify-initialization-zip":
        print(json.dumps(verify_initialization_zip(*sys.argv[2:]), indent=2))
    elif len(sys.argv) == 3 and sys.argv[1] == "--verify-launch-zip":
        print(json.dumps(verify_launch_zip(sys.argv[2]), indent=2))
    elif len(sys.argv) == 3 and sys.argv[1] == "--verify-conditioning-zip":
        print(json.dumps(verify_conditioning_zip(sys.argv[2])))
    elif len(sys.argv) == 3 and sys.argv[1] == "--estimate-conditioning-zip":
        status = verify_conditioning_zip(sys.argv[2])["status"]
        if status not in {"COMPLETE", "PARTIAL"}:
            raise SystemExit("A settled complete or partial conditioning ZIP is required.")
        _, members = diagnostic_zip_members(sys.argv[2], CONDITIONING)
        print(json.dumps(launch_estimate(members), indent=2))
    elif len(sys.argv) == 3 and sys.argv[1] == "--verify-bank-zip":
        print(json.dumps(verify_bank_zip(sys.argv[2])))
    elif sys.argv[1:2] == ["--isolated-control-worker"]:
        isolated_control_worker(*sys.argv[2:])
    else:
        raise SystemExit(cli(sys.argv[1:]))
