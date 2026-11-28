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
ZIP_SCOPES = {"readiness": "round029-readiness/v1", "checkpoints": "checkpoint_matrix",
              "control-history": "round029-control-history/v1", "isolated-controls": "round029-isolated-control-prices/v1"}
MODES = ("donation", "children", "review-children", "supply", "timings", "broader", "fresh-start", *ZIP_SCOPES)


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
        from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
        from oeis_learn.synthesis.client import SynthesisClient

        if (
            not Path(provenance.__file__).resolve().is_relative_to(Path(tmp))
            or digest(provenance.source_snapshot()) != expected_source
        ):
            raise ValueError("installed application differs from its funded source")
        request = load(work / "inputs/request.json")
        data = load(work / "inputs/data.json")
        if (
            request != seal(request)
            or request["version"] != ISOLATED_CONTROLS
            or request["data"] != digest(data)
            or request["image"] != os.environ["OEIS_QUALIFICATION_IMAGE"]
            or request["source"] != expected_source
            or request["wheel"] != expected_hash
            or os.environ["OEIS_QUALIFICATION_ENGINE"] != "podman"
        ):
            raise ValueError("isolated control request differs from its binding")
        began = time.monotonic_ns()
        with SynthesisClient(digest([ISOLATED_CONTROLS, request["identity"]])) as client:
            native = client.runtime_identity("bigint-body/v1")
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
    )


def control_installation_peak(wheel, area):
    block = os.statvfs(area).f_frsize
    with zipfile.ZipFile(wheel) as archive:
        info = archive.infolist()
    dirs = {p for item in info for p in Path(item.filename).parents}
    size = sum(((item.file_size + block - 1) // block) * block for item in info) + block * len(dirs)
    # The role installs a checked pure-Python wheel offline, with no compiled bytecode.
    # Allow a staging tree and its installed copy on the same filesystem.
    return dict(bytes=2 * size, inodes=2 * (len(info) + len(dirs)))


def control_capacity(root, area, revision, floor, installation):
    from oeis_learn.generalization.progressive.contracts import CAMPAIGN_DOCUMENT_BYTES
    from oeis_learn.pipeline_validation.progressive_research_resources import velocity_capacity

    # One bounded candidate or scoring document is written at a time; prior results already occupy the mount.
    path = str(area.relative_to(root))
    owner = dict(
        owner="isolated_controls",
        path=path,
        scope="cpu_control_role",
        retained=[],
        observed=dict(bytes=0, inodes=0),
        projected=dict(
            bytes=installation["bytes"] + CAMPAIGN_DOCUMENT_BYTES,
            inodes=installation["inodes"] + 32,
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


def run_control_stage(
    root, source, area, grant, journal, host, runtime, request, data, cost, extra, proof, stage
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
    controls = area / "inputs" / method / phase
    rp, dp = controls / "request.json", controls / "data.json"
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
        )
        if any(m["deficit_bytes"] or m["deficit_inodes"] for m in mounts):
            raise ValueError("insufficient physical space for an isolated CPU role")

    def execute(out, allowance):
        timeout = min(660, allowance // 1_000_000 - 120)
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
            gpu=False,
            seconds=timeout,
            readonly_inputs=inputs,
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
            limits.get("Devices")
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
            or not 0 < producer["worker_us"] <= dispatch["elapsed_us"]
        ):
            raise ValueError("original CPU confinement, producer or supervision clock differs")
        name = "generated/candidates.json" if phase == "search" else "scored/scores.json"
        ref = producer["artifact"]
        if ref["path"] != name or ref["bytes"] > CAMPAIGN_DOCUMENT_BYTES:
            raise ValueError("unexpected isolated control artifact")
        value = read_bound_record(
            out,
            ref,
            version="discovery-candidates/v1" if phase == "search" else "discovery-scoring/v1",
        )
        return seal(
            dict(
                artifact=ref,
                elapsed_us=dispatch["elapsed_us"],
                worker_us=producer["worker_us"],
                native=producer["native"],
                **proof(value),
            )
        )

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
    balance = BudgetJournal.inspect_existing(
        area / "accounting", grant["identity"], grant["resources"]
    )
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


def donor_isolated_controls(action, root, source, revision, minutes):
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
            report = isolated_control_matrix(
                root, source, folder, revision, minutes, runtime, panel, wave, host, stage
            )
        report["files"].update(files)
        return report
    except (Exception, SystemExit, KeyboardInterrupt) as exc:
        print(f"{step}: {exc}", file=sys.stderr)
        result = dict(
            candidate=revision,
            scope=ISOLATED_CONTROLS,
            status="UNVERIFIED" if step == "cpu-roles" else "BLOCKED",
            stage=step,
            error=type(exc).__name__,
            grant_minutes=int(minutes),
            files=files,
            research_run_requested=False,
            scientific_verdict=None,
            gpu_dispatches=0,
        )
        if step != "cpu-roles":
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


def main(revision, minutes, mode="donation", supplied=None):
    if mode not in MODES:
        raise SystemExit("Unknown continuation qualification mode.")
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
    def action(name, *args, at=revision):
        return checked_json(["bash", str(source / LAUNCHER), at, name, *map(str, args)], root, name)

    if mode == "isolated-controls":
        return donor_isolated_controls(action, root, source, revision, minutes)
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
        choices=MODES,
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="save the result to a new file (ZIP for readiness, checkpoints and control reports; JSON for other modes)",
    )
    parser.add_argument("--from-result", type=Path, help="completed fresh-start JSON for read-only readiness")
    args = parser.parse_args(argv)
    readiness = args.mode == "readiness"
    zipped = args.mode in ZIP_SCOPES
    if zipped and (not args.output or args.output.suffix.lower() != ".zip"):
        parser.error(f"{args.mode} requires --output NEW.zip")
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
        os.link(out.name, args.output)
    print(f"Saved {args.output.resolve()}", file=sys.stderr)
    return code


if __name__ == "__main__":
    if sys.argv[1:2] == ["--isolated-control-worker"]:
        isolated_control_worker(*sys.argv[2:])
    else:
        raise SystemExit(cli(sys.argv[1:]))
