"""Run a pinned donor preparation or qualification and save its verified result."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

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


def donor_timings(action, root, revision, minutes, *, broader=False):
    from oeis_learn.pipeline_validation.progressive_velocity_timing_host import authorize_timing

    from oeis_learn.pipeline_validation.progressive_velocity_broad_host import (
        authorize as authorize_broader,
    )

    authority = (authorize_broader if broader else authorize_timing)(root, int(minutes), revision)
    try:
        first = action("qualify", authority)
    except SystemExit as exc:
        first = json.loads(str(exc))
    summary = dict(
        candidate=revision,
        scope="broader_update_timings" if broader else "ordinary_update_timings",
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
        return dict(summary, diagnostics=failure_details(root, summary["grant"], current))
    verification = action("verify")
    record = first["receipt"]
    if (
        verification["verification_scope"]
        != ("private_broader_timings" if broader else "private_update_timings")
        or verification["last_committed_ref"] != record["identity"]
        or first["research_dispatches"] != 0
    ):
        raise ValueError("original timing debit or private replay differs")
    inspected = action("preflight")
    prices = inspected["inventory"]["prices"]
    if broader:
        if (
            prices.get("broader_updates") != record
            or prices.get("timed_updates", {}).get("measurement") != record["ordinary_measurement"]
        ):
            raise ValueError(
                "preflight did not reopen both paid scopes and their shared calibration"
            )
        return dict(summary, broader=record, prices=prices, next_action=inspected["reason"])
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


def main(revision, minutes, mode="donation"):
    if mode not in {"donation", "children", "review-children", "supply", "timings", "broader"}:
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

    if mode in {"timings", "broader"}:
        return donor_timings(action, root, revision, minutes, broader=mode == "broader")
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


def cli(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("revision", metavar="FULL_COMMIT")
    parser.add_argument("minutes", metavar="MAXIMUM_MINUTES")
    parser.add_argument(
        "mode",
        nargs="?",
        default="donation",
        choices=("donation", "children", "review-children", "supply", "timings", "broader"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="save the verified summary or failure diagnostic to a new JSON file",
    )
    args = parser.parse_args(argv)
    if args.mode in {"supply", "timings", "broader"} and args.output is None:
        parser.error("Donor preparation requires an output file (--output PATH)")
    request = (args.revision, args.minutes, args.mode)
    if args.output is None:
        print(json.dumps(main(*request), indent=2))
        return 0
    if os.path.lexists(args.output):
        parser.error(f"Output already exists; choose a new file: {args.output}")
    # Open the destination directory before any funded work. Publish only complete bytes.
    with tempfile.NamedTemporaryFile(dir=args.output.parent, mode="w", encoding="utf-8") as out:
        try:
            report = main(*request)
            code = 1 if report.get("status") in ("FAILED", "BLOCKED") else 0
        except (SystemExit, Exception) as exc:
            report = dict(
                candidate=args.revision,
                mode=args.mode,
                status="UNVERIFIED",
                error=type(exc).__name__,
                reason=str(exc),
            )
            code = 1
        json.dump(report, out, indent=2, allow_nan=False)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
        os.link(out.name, args.output)
    print(f"Saved {args.output.resolve()}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(cli(sys.argv[1:]))
