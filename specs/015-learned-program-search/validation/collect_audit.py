"""Collect the selected workstation action and its shareable measurements in one visit."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def read_result(path, step, exit_code):
    """Read the CLI's final JSON line; Podman writes build progress to the same descriptor."""
    from oeis_learn.generalization.progressive.contracts import CAMPAIGN_DOCUMENT_BYTES
    with path.open("rb") as stream:
        length = stream.seek(0, os.SEEK_END)
        size = min(length, 8192)
        while True:
            stream.seek(length-size)
            tail = stream.read(size).rstrip()
            if b"\n" in tail or size == length:
                break
            if size >= CAMPAIGN_DOCUMENT_BYTES:
                raise ValueError("terminal command record exceeds reader allowance")
            size = min(length, size*2, CAMPAIGN_DOCUMENT_BYTES)
    value = json.loads(tail.rsplit(b"\n", 1)[-1])
    if not isinstance(value, dict) or not any(k in value for k in
            ("mechanical_status", "status", "private_replay", "producer_error")):
        raise ValueError("missing terminal command result")
    if exit_code == 0 and step == "rehearsal" and not (
            value.get("mechanical_status") == "SKIPPED" and value.get("reason") == "GENERATOR_NOT_SELECTED"
            or value.get("mechanical_status") == "COMPLETE" and value.get("reason") == "MATCHED_REHEARSAL_MEASURED"
            and all(isinstance(value.get(k), dict) and value[k] for k in ("cpu", "hip", "forecast"))):
        raise ValueError("successful rehearsal lacks its measurements")
    if exit_code == 0 and step == "verify-copied" and value.get("status") != "VERIFIED":
        raise ValueError("missing copied archive verification")
    return value


def prior_console(workspace, revision):
    """Return bounded evidence from the original collection even if its receipts are incomplete."""
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_round028 import SESSION
    from oeis_learn.pipeline_validation.progressive_round028_rehearsal import training_forecast

    tools = confined(workspace, "run-state/discovery-tools")
    captures = []
    for path in sorted(tools.glob("round028-rehearsal-*/summary.json"), key=lambda p:p.stat().st_mtime_ns, reverse=True):
        try:
            summary = json.loads(confined(tools, path.relative_to(tools)).read_bytes())
            if summary.get("candidate") == revision and "rehearsal" in summary.get("steps", {}):
                output = confined(tools, (path.parent / "rehearsal.json").relative_to(tools))
                captures.append(read_result(output, "rehearsal", summary["steps"]["rehearsal"]["exit"]))
                break
        except (OSError, ValueError, KeyError):
            continue
    try:
        progress = confined(workspace, Path(SESSION) / "private/generator-rehearsal" / revision / "progress.json")
        from oeis_learn.generalization.progressive.contracts import load, seal
        value = load(progress)
        if seal(value) == value:
            captures.append(value)
    except (OSError, ValueError):
        pass
    report = []
    for value in captures:
        if value.get("candidate") != revision:
            continue
        allowed = {"phase": {"cpu", "hip", "complete"}, "mechanical_status": {"COMPLETE", "SKIPPED", "BLOCKED"},
                   "reason": {"GENERATOR_NOT_SELECTED", "NO_TRAINING_SUPPORT_OUTSIDE_PROBES",
                              "FREE_SPACE_FLOOR_DEFICIT", "POST_BUILD_STORAGE_DEFICIT", "CPU_REHEARSAL_FAILED",
                              "HIP_REHEARSAL_FAILED", "MATCHED_REHEARSAL_MEASURED"}}
        state = {k:value[k] for k, options in allowed.items() if isinstance(value.get(k), str) and value[k] in options}
        cpu = value.get("cpu", {})
        if type(cpu.get("gradient_l2")) in (int,float) and 0 < cpu["gradient_l2"] < float("inf"):
            state["reported_gradient_l2"] = cpu["gradient_l2"]
        try:
            arms = value["hip"]["arms"]
            measures = {a:{k:arms[a][k] for k in ("charged_tokens", "update_us")} for a in ("ordinary","first_divergence")}
            if all(isinstance(v,list) and len(v)==2 and all(type(x) is int and x>0 for x in v)
                   for arm in measures.values() for v in arm.values()):
                state["reported_updates"] = measures
                state["reported_forecast"] = training_forecast(measures)
        except (TypeError, KeyError, ArithmeticError):
            pass
        report.append(state)
    return report


def collect(workspace, candidate, action="collect-audit", producer_revision=None):
    source = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(source / "src"))
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_round028 import ARCHIVE, SESSION

    destination = confined(workspace, "run-state/discovery-tools")
    rehearsal = action == "collect-rehearsal"
    prefix = {"collect-audit":"round028-cpu-", "collect-rehearsal":"round028-rehearsal-",
              "collect-panels":"round028-panels-", "collect-receipts":"round028-receipts-"}[action]
    folder = Path(tempfile.mkdtemp(prefix=prefix, dir=destination))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    cli = [sys.executable, "-B", str(source / "scripts/foundation_run.py"), "--engine", "podman",
           "discovery", "round028"]
    summary = dict(version="round028-workstation-collection/v1", candidate=candidate, steps={})

    def run(name, action, *authority):
        print(f"Running {name}", file=sys.stderr, flush=True)
        output = folder / (name + ".json")
        with output.open("xb") as out, (folder / (name + ".log")).open("xb") as log:
            process = subprocess.run([*cli, "--action", action, *authority], cwd=folder, env=env,
                                     stdout=out, stderr=log)
        try:
            raw = read_result(output, name, process.returncode)
        except (ValueError, UnicodeError):
            code = process.returncode or 2
            summary["steps"][name] = dict(mechanical_status="INVALID_COMMAND_RESULT",
                                          process_exit=process.returncode, exit=code)
            return code, {}
        # Raw local output may contain a private path or case key in an exception.
        # Only fixed command fields, numeric counts and tracked code locations travel.
        fields = ("mechanical_status", "scientific_status", "reason", "scope", "admission", "denominators",
                  "expected_cases", "authenticated_cases", "qualified_checkout", "filesystem", "gpu_dispatches",
                  "unfinished_trial05_composition", "private_replay", "status", "decision", "owner_active")
        step = {k: raw[k] for k in fields if k in raw}
        if name == "rehearsal":
            step.update({k: raw[k] for k in ("candidate", "audit", "phase", "support", "original_scale", "cpu",
                        "hip", "forecast", "required_bytes", "deficit_bytes", "error_type", "failure_locations",
                        "next_action") if k in raw})
        if name == "status" and "rehearsal" in raw:
            step["rehearsal"] = {k: raw["rehearsal"][k] for k in ("candidate", "phase", "reason", "next_action")
                                 if k in raw["rehearsal"]}
        if name == "supply":
            step.update({k:raw[k] for k in ("error_type","failure_locations") if k in raw})
        if name == "recover":
            step.update({k:raw[k] for k in ("stage_requests", "published_stages", "matched_boundaries",
                                             "failure_locations", "cpu") if k in raw})
        if name == "preflight" and "files" in raw:
            step.update(donor=raw["donor"], prerequisites=raw["files"])
        error = raw.get("producer_error") or {}
        if error:
            step["error"] = {k: error[k] for k in ("type", "code") if k in error}
            step["locations"] = raw.get("source_locations", [])
        step["exit"] = process.returncode
        summary["steps"][name] = step
        return process.returncode, raw

    root = ["--workspace", str(workspace)]
    if action == "collect-panels":
        code, raw = run("supply", "panels", *root)
        if not code:
            try:
                expected = Path("share/panel-supply") / candidate / "round028-panels.zip"
                if raw.get("archive") != str(expected):
                    raise ValueError("missing completed panel supply publication")
                copied = folder / "round028-panels.zip"
                shutil.copyfile(confined(workspace,Path(SESSION)/expected),copied)
                code, standalone = run("verify-copied","verify","--archive",str(copied))
                if not code and standalone == raw["receipt"]:
                    print(f"Verified panel supply and historical costing. Return this file:\n{copied}",flush=True)
                    return 0
                code = code or 2
            except (KeyError,TypeError,ValueError,OSError,zipfile.BadZipFile) as exc:
                summary["steps"]["publication"] = dict(error_type=type(exc).__name__,exit=2)
                code = 2
        summary.update(exit=code, gpu_dispatches=0)
        from oeis_learn.generalization.progressive.contracts import load, seal
        area = workspace / SESSION / "private/panel-supply" / candidate
        try:
            progress = load(confined(workspace,area/"progress.json"))
            phases = {"authenticated_historical_clocks","connecting_historical_families","retained_supply",
                      "native_supply","conventional_calibration","complete"}
            allowed = {"timed","candidates","historical_rows","eligible","components","deficient_cells",
                       "calibration_families","accepted","attempted"}
            if seal(progress) == progress and progress["phase"] in phases:
                summary["progress"] = dict(phase=progress["phase"], **{k:v for k,v in progress.items()
                    if k in allowed and type(v) is int and v >= 0})
        except (KeyError,OSError,ValueError):
            pass
        diagnostics = folder / "round028-panel-diagnostics.zip"
        with zipfile.ZipFile(diagnostics,"x",zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("summary.json",json.dumps(summary,sort_keys=True,indent=2,allow_nan=False))
        print(f"Panel collection exit {code}. Return this file:\n{diagnostics}",flush=True)
        return code
    if action == "collect-receipts":
        from oeis_learn.generalization.progressive.program_search_rehearsal_report import publish
        code, raw = run("recover", "recover-rehearsal", *root, "--producer-revision", producer_revision)
        if not code:
            try:
                archive = folder / "round028-rehearsal.zip"
                if raw.get("reason") != "SAVED_REHEARSAL_VERIFIED":
                    raise ValueError("missing original physical rehearsal")
                receipt = publish(archive, raw["evidence"], confined(workspace, Path(SESSION) / ARCHIVE))
                code, standalone = run("verify-copied", "verify", "--archive", str(archive))
                if not code and standalone == receipt:
                    print(f"Verified saved rehearsal. Return this file:\n{archive}", flush=True)
                    return 0
                code = code or 2
            except (KeyError, TypeError, ValueError, OSError, zipfile.BadZipFile) as exc:
                summary["steps"]["publication"] = dict(error_type=type(exc).__name__, exit=2)
                code = 2
        summary.update(exit=code, producer_revision=producer_revision, gpu_dispatches=0,
                       prior_observations=prior_console(workspace, producer_revision))
        report_path = folder / "round028-receipt-diagnostics.zip"
        with zipfile.ZipFile(report_path, "x", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("summary.json", json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
        print(f"Saved-receipt recovery exit {code}. Return this file:\n{report_path}", flush=True)
        return code
    preflight, _ = run("preflight", "preflight", *root)
    stage = "rehearse" if rehearsal else "audit"
    audit, value = run("rehearsal" if rehearsal else "audit", stage, *root) if preflight == 0 else (preflight, {})
    code = audit
    include_archive = False
    if value.get("archive") == ARCHIVE or (rehearsal and (workspace / SESSION / ARCHIVE).is_file()):
        private = 0
        if not rehearsal:
            private, _ = run("verify-private", "verify", *root)
        copied = folder / "round028.zip"
        shutil.copyfile(confined(workspace, Path(SESSION) / ARCHIVE), copied)
        public, _ = run("verify-copied", "verify", "--archive", str(copied))
        include_archive = public == 0
        code = private or public or audit
    status, _ = run("status", "status", *root)
    code = code or status
    summary["exit"] = code
    summary["includes_verified_audit"] = include_archive
    (folder / "summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    result = folder / "round028-workstation.zip"
    with zipfile.ZipFile(result, "x", zipfile.ZIP_DEFLATED) as archive:
        archive.write(folder / "summary.json", "summary.json")
        if include_archive:
            archive.write(copied, "round028.zip")
    print(f"{stage} exit {code}. Return this file:\n{result}", flush=True)
    return code


if __name__ == "__main__":
    os.umask(0o077)
    raise SystemExit(collect(Path(sys.argv[1]), *sys.argv[2:]))
