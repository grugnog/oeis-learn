"""Verify a terminal round027 invocation or compare the saved screening panel scores."""
import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

# git archive supplies the reporting package. A panel check streamed by git show uses the qualified checkout.
source_root = Path.cwd() if __file__ == "<stdin>" else Path(__file__).resolve().parents[3]
sys.path.insert(0, str(source_root / "src"))
from oeis_learn.generalization.contracts import verify_artifact
from oeis_learn.generalization.progressive.cohorts import STRATA
from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
from oeis_learn.generalization.progressive.report import build, MAX_BYTES
from oeis_learn.generalization.progressive.verify import verify
from oeis_learn.pipeline_validation.identity import confined
from oeis_learn.pipeline_validation.progressive_report_collect import collect
from oeis_learn.pipeline_validation.progressive_workstation_state import read

CAMPAIGN = "sha256:b35444dcbdc68693917a61d05ac3dc11efbdebee568a074b80f2596191035304"
SESSION = "run-state/discovery/013-research-round-027"
FOLLOWUP = "run-state/discovery/013-round027-trial00-followup"
TWO_M_REPORT = "sha256:15147c373c2a854bdc79fab9d4a10995b0337cd76b386eb23602deec47504302"
CAPABILITY_CUT = dict(path=f"run-state/discovery-control/{CAMPAIGN[7:]}/transition-0096.json",
                      bytes=26457, sha256="sha256:83150679881f59f0f3631871463ff2dea9114c505cada674ba4dc8d4ac7affd1")


def saved_look(folder, relative, expected):
    value = read(confined(folder, relative))
    if value["identity"] != expected:
        raise ValueError(f"screening evidence differs: {relative}")
    return value


def trial_panel(root):
    folder = confined(root, FOLLOWUP)
    final = saved_look(folder, "share/trial-00-2M.json", TWO_M_REPORT)
    first = saved_look(folder, "share/1M-replay.json", final["previous_report"])
    if len(first["measurements"]) != 6:
        raise ValueError("six original 1M measurements required")
    looks = [(i, 1000000, f"trial-{i:02d}-1M", ref) for i, ref in enumerate(first["measurements"])]
    looks.append((0, 2000000, "trial-00-2000000", final["measurement"]))
    roster, trials = {}, {}
    for ordinal, tokens, name, ref in looks:
        m = saved_look(folder, f"private/{name}.json", ref)
        if (m["trial"], m["coordinate"]) != (ordinal, tokens):
            raise ValueError("screening checkpoint differs")
        members = {row["task"]: (row["stratum"], row["group"]) for row in m["roster"]}
        if (len(m["roster"]) != 128 or Counter(s for s, _ in members.values()) != dict.fromkeys(STRATA, 32)
                or (roster and members != roster)):
            raise ValueError("screening problems or groups differ")
        roster = members
        values = {task: row["companion"] for task, row in m["panels"]["development"].items()}
        if values.keys() != roster.keys() or set(values.values()) - {"success", "failure", "unresolved"}:
            raise ValueError("development scores do not cover the screening panel")
        trials[ordinal, tokens] = values
    return roster, trials


def original_json(base, ref):
    path = confined(base, ref["path"])
    return path, json.loads(verify_artifact(base, ref, max_bytes=ref["bytes"]))


def control_panel(root, outputs, model, roster):
    rows = {}
    for i in range(8):
        name = f"capability-{model}-{i:02d}-score"
        path, receipt = original_json(root, outputs[name]["result"])
        if receipt["operation"] != "score" or receipt["mechanical_status"] != "COMPLETE":
            raise ValueError(f"original capability score missing: {name}")
        _, score = original_json(path.parent, receipt["metrics"]["scores"])
        if score["version"] != "discovery-scoring/v1":
            raise ValueError(f"unexpected score format: {name}")
        for task, methods in score["outcomes"].items():
            if task in roster:
                outcome = methods["equal_budget"]["outcome"]
                if task in rows or outcome not in {"success", "failure", "unresolved"}:
                    raise ValueError(f"duplicate or invalid control outcome: {name}")
                rows[task] = outcome
    if rows.keys() != roster.keys():
        raise ValueError(f"{model} scores do not cover the screening panel")
    return rows


def panel_report(root):
    roster, trials = trial_panel(root)
    controls = {}
    missing = []
    try:
        _, transition = original_json(root, CAPABILITY_CUT)
        for model in ("random", "retained_D"):
            try:
                controls[model] = control_panel(root, transition["state"]["outputs"], model, roster)
            except FileNotFoundError as exc:
                missing.append(f"{model}: {exc.filename}")
    except FileNotFoundError as exc:
        missing.append(str(exc))
    for detail in missing:
        print("Capability reference unavailable:", detail, file=sys.stderr)
    print("Each cell shows verified solves / unresolved cases, out of 32.")
    print("Controls use original equal-budget scoring. Trials use the follow-up rule for completed misses.")
    print(f"{'Model':<20}" + "".join(f"{s:>18}" for s in STRATA))
    values_by_model = {**controls, **{f"trial {i:02d} {tokens // 1000000}M": v for (i, tokens), v in trials.items()}}
    for label, values in values_by_model.items():
        cells = []
        for stratum in STRATA:
            count = Counter(values[t] for t, (s, _) in roster.items() if s == stratum)
            cells.append(f"{count['success']} / {count['unresolved']}")
        print(f"{label:<20}" + "".join(f"{cell:>18}" for cell in cells))
    print("Distinct problems verified by at least one of the six 1M trials:")
    solved = {t for (_, tokens), values in trials.items() if tokens == 1000000
              for t, outcome in values.items() if outcome == "success"}
    for stratum in STRATA:
        print(f"  {stratum:<18} {sum(roster[t][0] == stratum for t in solved)}/32")


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--wait", action="store_true", help="wait for research to release the workstation and publish its terminal result")
    mode.add_argument("--panel", action="store_true", help="compare original controls and trial checkpoints on the same screening cases")
    args = parser.parse_args()
    if args.panel:
        try:
            panel_report(Path.cwd().resolve())
        except (ValueError, KeyError, OSError) as exc:
            parser.exit(2, f"Screening panel check stopped: {exc}\n")
        raise SystemExit(0)
    rc = ready_report(Path.cwd())
    if rc is None:
        print("Waiting for the resumed round027 invocation to finish and release its locks.", flush=True)
    while args.wait and rc is None:
        time.sleep(30)
        rc = ready_report(Path.cwd())
    raise SystemExit(2 if rc is None else rc)
