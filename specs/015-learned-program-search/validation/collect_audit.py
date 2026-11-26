"""Collect the selected workstation action and its shareable measurements in one visit."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def collect(workspace, candidate, action="collect-audit"):
    source = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(source / "src"))
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_round028 import ARCHIVE, SESSION

    destination = confined(workspace, "run-state/discovery-tools")
    rehearsal = action == "collect-rehearsal"
    folder = Path(tempfile.mkdtemp(prefix="round028-rehearsal-" if rehearsal else "round028-cpu-", dir=destination))
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
            raw = json.loads(output.read_bytes())
        except (ValueError, UnicodeError):
            raw = {}
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
