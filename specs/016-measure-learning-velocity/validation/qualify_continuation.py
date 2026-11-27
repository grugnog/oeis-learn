"""Run and reopen either the donated ordinary comparison or the disposable parent/child rehearsal."""

import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

DONOR = "77667e67176ae32fa16457d52565bae1dbe41afc"
CACHE_HELPER = "specs/015-learned-program-search/validation/candidate_source.py"
LAUNCHER = "specs/016-measure-learning-velocity/validation/launch-round029.sh"


def checked_json(argv, cwd):
    proc = subprocess.run(argv, cwd=cwd, stdout=subprocess.PIPE)
    if proc.returncode:
        raise SystemExit(proc.stdout.decode().strip() or f"command exited {proc.returncode}")
    return json.loads(proc.stdout)


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
        charged_tokens=first["charged_tokens"],
        elapsed_us=first["elapsed_us"],
        remaining_us=first["remaining_authority"],
        lease_us=record["lease_us"],
        peak_device_bytes=record["peak_device_bytes"],
        verified_replays=2,
        research_dispatches=0,
        scientific_verdict=None,
    )


def main(revision, minutes, mode="donation"):
    if mode not in {"donation", "children"}:
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
    launcher = ["bash", str(source / LAUNCHER), revision]

    def action(name, *args):
        return checked_json([*launcher, name, *map(str, args)], root)

    inventory = action("preflight")["inventory"]
    if inventory["gpu_dispatches"] != 0:
        raise SystemExit("A read-only preflight issued device work.")
    sys.path.insert(0, str(source / "src"))
    if mode == "children":
        print(json.dumps(child_rehearsal(action, root, revision, minutes), indent=2))
        return
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
        )
    if (
        public["report_ref"] != private["report_ref"]
        or public["last_committed_ref"] != first["last_committed_ref"]
        or public["scientific_verdict"] is not None
    ):
        raise SystemExit("Independent diagnostic replay differs from the private workspace.")
    record = first["receipt"]
    print(
        json.dumps(
            dict(
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
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        raise SystemExit("Expected FULL_COMMIT MAXIMUM_MINUTES [donation|children].")
    main(*sys.argv[1:])
