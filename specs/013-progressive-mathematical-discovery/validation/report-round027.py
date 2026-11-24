"""Collect and independently verify a terminal round027 invocation using the reporting checkout."""
import argparse
from contextlib import ExitStack
import fcntl
import os
from pathlib import Path
import re
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
from oeis_learn.generalization.progressive.report import build, MAX_BYTES
from oeis_learn.generalization.progressive.verify import verify
from oeis_learn.pipeline_validation.identity import confined
from oeis_learn.pipeline_validation.progressive_report_collect import collect

CAMPAIGN = "sha256:b35444dcbdc68693917a61d05ac3dc11efbdebee568a074b80f2596191035304"
SESSION = "run-state/discovery/013-research-round-027"


def report(root, folder, session):
    run = load(folder / "run.json")
    result = load(folder / "result.json")
    if any(seal(v) != v or v.get("campaign") != CAMPAIGN for v in (run, result)):
        raise ValueError("expected the original terminal round027 run and result")
    control = confined(root, "run-state/discovery-control/" + CAMPAIGN[7:])
    print(f"{folder.name} terminal status: {result['mechanical_status']}", flush=True)
    tip = load(max(control.glob("transition-*.json")))
    if seal(tip) != tip:
        raise ValueError("original round027 transition differs")
    destination = session / "run-research" / "verified-reports" / folder.name / digest(
        [result["identity"], tip["identity"], "large-producers/v1"]
    )[7:]
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "report.zip"
    if not archive.exists():
        print("Collecting saved results for " + folder.name, flush=True)
        with tempfile.TemporaryDirectory(dir=destination, prefix="build-") as tmp:
            tmp = Path(tmp)
            request = collect(root, folder, tmp / "metadata")
            cap = min(MAX_BYTES, request["campaign_record"]["payload"]["storage"]["package_bytes"])
            staged = tmp / "report.zip"
            build(root, staged, request, max_bytes=cap)
            # Keep the immutable archive even when the verifier diagnoses an evidence problem.
            os.link(staged, archive)
    print("Independently verifying " + str(archive), flush=True)
    result = verify(archive)
    receipt = destination / "verification.json"
    if not receipt.exists():
        atomic_write(receipt, result)
    elif load(receipt) != result:
        raise ValueError("verification differs from saved report receipt")
    print(f"{result['status']}   scientific verdict {result['decision']}", flush=True)
    if result["missing"]:
        print("Missing evidence: " + ", ".join(result["missing"]), flush=True)
    print(receipt, flush=True)
    return 0 if result["status"] == "VERIFIED" else 2


def ready_report(root):
    session = confined(root, SESSION)
    control = confined(root, "run-state/discovery-control/" + CAMPAIGN[7:])
    with ExitStack() as stack:
        for path in (session / "session.lock", control / "run.lock"):
            lock = stack.enter_context(path.open("rb"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return None
        runs = sorted(p for p in (session / "run-research").glob("execution-*")
                      if re.fullmatch(r"execution-[0-9]{4}", p.name) and p.is_dir())
        if not runs or runs[-1].name == "execution-0000" or not (runs[-1] / "result.json").is_file():
            steps = load(session / "journal.json", maximum=32 << 20)["steps"]
            ordinal = runs[-1].name[-4:] if runs and runs[-1].name != "execution-0000" else "0001"
            if steps.get("run-research/invocation-" + ordinal, {}).get("status") in {"FAILED", "BLOCKED", "INTERRUPTED"}:
                raise ValueError("the resumed invocation ended without a terminal result; see the original session journal")
            return None
        return report(root, runs[-1], session)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait", action="store_true", help="wait for research to release the workstation and publish its terminal result")
    args = parser.parse_args()
    rc = ready_report(Path.cwd())
    if rc is None:
        print("Waiting for the resumed round027 invocation to finish and release its locks.", flush=True)
    while args.wait and rc is None:
        time.sleep(30)
        rc = ready_report(Path.cwd())
    raise SystemExit(2 if rc is None else rc)
