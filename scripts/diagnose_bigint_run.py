#!/usr/bin/env python3
"""Collect existing diverse32 evidence; optionally replay one visible candidate.

Host operations use only the standard library. Collection never starts an engine.
Replay uses a fresh CPU container, no private truth, no weights and no parent mount.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def sha(path):
    with Path(path).open("rb") as handle:
        return "sha256:" + hashlib.file_digest(handle, "sha256").hexdigest()


def safe(root, relative):
    root, relative = Path(root), Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("relative artifact path required")
    path = root
    for part in relative.parts:
        path /= part
        if path.is_symlink():
            raise ValueError("symlink artifact rejected: " + str(path))
    return path


def load(path, cap=4 << 20):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > cap:
        raise ValueError("bounded regular JSON artifact required: " + str(path))
    return json.loads(
        path.read_bytes(), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))
    )


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def fresh(output):
    output = Path(output)
    if output.exists() or output.with_suffix(".zip").exists():
        raise ValueError("fresh output directory and archive required")
    output.mkdir(parents=True)
    return output


def archive(output):
    """Only utility-created reports; never recursively archive a run or cache."""
    files = []
    size = 0
    for base, dirs, names in os.walk(output, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d != ".runtime")
        if any((Path(base) / d).is_symlink() for d in dirs):
            raise ValueError("symlink report directory rejected")
        for name in sorted(names):
            path = Path(base) / name
            if path.is_symlink():
                raise ValueError("symlink report rejected")
            size += path.stat().st_size
            if size > 16 << 20:
                raise ValueError("diagnostic reports exceed16 MiB")
            files.append(
                dict(
                    path=str(path.relative_to(output)), bytes=path.stat().st_size, sha256=sha(path)
                )
            )
    write(Path(output) / "archive-manifest.json", dict(files=files))
    target = Path(output).with_suffix(".zip")
    with zipfile.ZipFile(target, "x", zipfile.ZIP_DEFLATED) as bundle:
        for row in files:
            bundle.write(Path(output) / row["path"], row["path"])
        bundle.write(Path(output) / "archive-manifest.json", "archive-manifest.json")
    return target


def collect(root, run, prompt_id, output):
    root = Path(root).resolve()
    run = safe(root, run)
    output = Path(output).resolve()
    if output == run or run in output.parents:
        raise ValueError("diagnostic output must be outside the parent run")
    output = fresh(output)
    references = {}

    def record(path, *, cap=4 << 20):
        value = load(path, cap)
        references[str(path.relative_to(root))] = dict(bytes=path.stat().st_size, sha256=sha(path))
        return value

    parent = record(run / "run.json")
    if parent.get("version") != "bigint-run/v1":
        raise ValueError("bigint run required")
    for name in ("status.json", "budget.json", "checkpoint-identity.json"):
        record(run / name)
    comparison = record(run / "comparison.json")
    # This utility diagnoses the already exposed synthetic tiny panel only.
    split = "0:tiny"
    observed = comparison["observations"][split][prompt_id]
    role_name = Path(comparison["splits"][split]["archive"]["role_output"]).name
    if not role_name.startswith("role-") or len(role_name) != 37:
        raise ValueError("unexpected generation role identity")
    job = safe(root, "run-state/" + digest(str(run.relative_to(root)))[7:])
    role = safe(job, "roles/" + role_name)
    inputs = safe(job, "inputs/" + role_name)
    settings = record(safe(inputs, "settings.json"))
    if (
        settings.get("final_access", "sealed") != "sealed"
        or settings["contract_sha256"] != parent["contract_sha256"]
    ):
        raise ValueError("parent/generation contract mismatch or final role")
    rows = record(safe(inputs, "prompts.json"))
    row = next(row for row in rows if row["id"] == prompt_id)
    if len(row["indices"]) != 20 or len(row["values"]) != 20:
        raise ValueError("exactly20 visible terms required")
    prompt = dict(indices=row["indices"], values=row["values"])
    result = record(safe(role, "result.json"))
    selections = None
    if len(result["files"]) > 64:
        raise ValueError("selection chunk count cap")
    for name, ref in result["files"].items():
        if not name.startswith("selections/"):
            raise ValueError("selection manifest path required")
        path = safe(role, name)
        chunk = record(path)
        if sha(path) != ref["sha256"] or path.stat().st_size != ref["bytes"]:
            raise ValueError("sealed selection chunk mismatch")
        if prompt_id in chunk:
            if selections is not None:
                raise ValueError("duplicate prompt selection")
            selections = chunk[prompt_id]
    if selections is None:
        raise ValueError("prompt missing from sealed selections")
    for chosen in selections["selections"].values():
        if chosen is not None and chosen["prompt_sha256"] != digest(prompt):
            raise ValueError("selection prompt mismatch")
        if chosen is not None:
            selected = chosen["selected"]
            seal_body = {
                key: value
                for key, value in selected.items()
                if key not in {"selection", "seal_sha256"}
            }
            if (
                selected.get("selection") != "visible-only/v1"
                or selected.get("seal_sha256") != digest(seal_body)
                or selected["source_sha256"]
                != "sha256:" + hashlib.sha256(selected["source"].encode()).hexdigest()
            ):
                raise ValueError("candidate seal/source mismatch")
    detail = None
    detailed = safe(role, "details/" + digest(prompt_id)[7:] + ".json")
    if detailed.exists():
        detail = record(detailed, cap=1 << 20)
    compact = None
    for path in sorted(safe(role, "evidence").glob("*.json"))[:64]:
        chunk = record(safe(role, str(path.relative_to(role))), cap=1 << 20)
        if prompt_id in chunk:
            compact = chunk[prompt_id]
    raw = selections["selections"].get("raw")
    greedy_source = (compact or {}).get("greedy_source")
    if greedy_source is None and raw:
        greedy_source = raw["original_source"]
    if greedy_source is None and detail:
        attempts = detail.get("details", {}).get("correct", {}).get("details", [])
        greedy_source = attempts[0]["source"] if attempts else None
    recipe = settings["recipe"]
    seed = (recipe["model"]["seed"] + int(digest(prompt_id)[7:14], 16)) % 2**32
    case = dict(
        version="bigint-candidate-replay/v1",
        prompt_id=prompt_id,
        prompt=prompt,
        original_source=greedy_source,
        original_source_sha256="sha256:" + hashlib.sha256(greedy_source.encode()).hexdigest()
        if greedy_source
        else None,
        parent_contract_sha256=parent["contract_sha256"],
        parent_image=parent["image"],
        parent_checkpoint_identity_sha256=references[
            str((run / "checkpoint-identity.json").relative_to(root))
        ]["sha256"],
        expected_runtime_sha256=raw["verification"].get("runtime_sha256") if raw else None,
        resources=recipe["execution"]["resources"],
        service_limits=recipe["fitting"]["service_limits"],
        fitting_disabled=recipe["fitting"]["disabled"],
        seed=seed,
        fitting_seconds=4,
        raw_seconds=4,
        policy="none"
        if greedy_source and "(param.get " in greedy_source
        else "eligible-literals/v1",
    )
    write(output / "case.json", case)
    write(output / "selection.json", selections)
    write(output / "observation.json", observed)
    if detail is not None:
        write(output / "detail.json", detail)
    if compact is not None:
        write(output / "evidence.json", compact)
    # Successful fits alone enter this cache; absence is never a failure diagnosis.
    cached, scanned, scanned_count = [], 0, 0
    cache_files = sorted(safe(role, "native-cache").glob("*.json"))
    for path in cache_files[:256]:
        path = safe(role, str(path.relative_to(role)))
        if path.stat().st_size > 4 << 20 or scanned + path.stat().st_size > 32 << 20:
            break
        scanned += path.stat().st_size
        scanned_count += 1
        cached_row = load(path)
        value = cached_row.get("value", {})
        if any(
            item.get("original_source_sha256") == case["original_source_sha256"]
            and item.get("prompt_sha256") == digest(prompt)
            for item in value.get("assignments", [])
        ):
            cached.append(cached_row)
            references[str(path.relative_to(root))] = dict(
                bytes=path.stat().st_size, sha256=sha(path)
            )
    write(output / "cached-fits.json", cached)
    score_role = None
    # Identify the scoring sibling by its sealed selection manifest. Never open truth.json.
    role_inputs = sorted(safe(job, "inputs").glob("role-*"))
    if len(role_inputs) > 256:
        raise ValueError("role search count cap")
    for source in role_inputs:
        candidate = safe(job, str(source.relative_to(job)) + "/selections.json")
        if candidate.exists() and record(candidate) == result["files"]:
            if score_role is not None:
                raise ValueError("ambiguous scoring sibling")
            score_role = safe(job, "roles/" + source.name)
            scored_settings = record(safe(source, "settings.json"))
            write(
                output / "scorer-settings.json",
                {key: scored_settings[key] for key in ("contract_sha256", "seconds", "device")},
            )
            outcomes = record(safe(score_role, "outcomes.json"))
            write(output / "scorer-observation.json", outcomes[prompt_id])
    for label, source in (("generation", role), ("scoring", score_role)):
        if source is None:
            continue
        for name in ("stdout.log", "stderr.log"):
            path = safe(source, name)
            if path.exists():
                references[str(path.relative_to(root))] = dict(
                    bytes=path.stat().st_size, sha256=sha(path)
                )
                with path.open("rb") as handle:
                    handle.seek(max(0, path.stat().st_size - (64 << 10)))
                    payload = handle.read(64 << 10)
                (output / (label + "-" + name)).write_bytes(payload)
    for name, ref in references.items():
        if sha(safe(root, name)) != ref["sha256"]:
            raise ValueError("parent evidence changed during read-only collection")
    write(
        output / "summary.json",
        dict(
            version="bigint-evidence-collection/v1",
            status="COLLECTED",
            qualified=False,
            parent=str(run.relative_to(root)),
            prompt_id=prompt_id,
            generation_role=role_name,
            original_outcomes=observed,
            sources=references,
            detailed_trace_available=detail is not None or compact is not None,
            candidate_replay_ready=bool(greedy_source),
            cached_fit_matches=len(cached),
            scoring_role=score_role.name if score_role else None,
            cache_scan_bytes=scanned,
            cache_scan_complete=scanned_count == len(cache_files),
            limitation="Cached fits are successful provisional results only; missing failed traces remain unknown. Replay is new analysis, never a replacement score.",
        ),
    )
    return archive(output)


def trace_worker_calls(client, events):
    """Observe original-image transport before legacy clients discard its reason."""
    worker = getattr(client, "worker", None)
    original = getattr(worker, "call", None)
    if not callable(original):
        return False

    def call(request, *, deadline):
        started = time.monotonic()
        event = dict(operation=request["operation"], allowance_seconds=max(0, deadline - started))
        try:
            response = original(request, deadline=deadline)
            event.update(
                {
                    key: response.get(key)
                    for key in ("complete", "outcome", "reason", "generation", "metrics")
                }
            )
            event.update(
                results_count=len(response.get("results", [])),
                unstarted_count=len(response.get("unstarted", [])),
            )
            return response
        except BaseException as exc:
            event.update(
                complete=False, outcome="exception", reason=f"{type(exc).__name__}: {exc}"[:512]
            )
            raise
        finally:
            event["elapsed_seconds"] = time.monotonic() - started
            if len(events) < 32:
                events.append(event)

    worker.call = call
    return True


def inside(case_path, output, seconds, image):
    """Visible-only CPU replay against the implementation packaged in the image."""
    from oeis_learn.synthesis.admission import VisiblePrompt, source_identity
    from oeis_learn.synthesis.client import SynthesisClient
    from oeis_learn.synthesis.contracts import ResourceProfile
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.synthesis.service_policy import ServiceLimits
    from oeis_learn.synthesis.state import BudgetLedger

    case = load(case_path)
    if case.get("version") != "bigint-candidate-replay/v1" or set(case["prompt"]) != {
        "indices",
        "values",
    }:
        raise ValueError("candidate replay case required")
    source = case["original_source"]
    if not source or source_identity(source) != case["original_source_sha256"]:
        raise ValueError("missing or mismatched original candidate")
    prompt = VisiblePrompt(tuple(case["prompt"]["indices"]), tuple(case["prompt"]["values"]))
    if len(prompt.indices) != 20 or case["fitting_seconds"] != 4 or case["raw_seconds"] != 4:
        raise ValueError("frozen visible replay limits required")
    output = Path(output)
    ledger = BudgetLedger(seconds)
    ledger.start("candidate-replay", seconds)
    write(output / "budget-start.json", ledger.snapshot())
    started = time.monotonic()
    deadline = started + seconds
    report = dict(
        version="bigint-visible-replay/v1",
        status="FAIL",
        qualified=False,
        parent_contract_sha256=case["parent_contract_sha256"],
        parent_image=case["parent_image"],
        image=image,
        same_image=image == case["parent_image"],
        case_sha256=sha(case_path),
        analysis_identity=digest([digest(case), seconds, image, uuid.uuid4().hex]),
        purpose="visible-only fitting diagnosis; no continuation score or promotion",
        worker_calls=[],
    )
    try:
        with SynthesisClient(
            case["parent_contract_sha256"],
            limits=ResourceProfile(**case["resources"]),
            service_limits=ServiceLimits(**case["service_limits"]),
            cache_root=output / ".runtime/native-cache",
        ) as client:
            report["transport_trace_available"] = trace_worker_calls(client, report["worker_calls"])
            report["runtime_sha256"] = client.runtime_identity("bigint-body/v1")
            report["same_runtime"] = (
                report["runtime_sha256"] == case["expected_runtime_sha256"]
                if case.get("expected_runtime_sha256")
                else None
            )
            until = min(deadline, time.monotonic() + 4)
            production = client.execute(source, prompt.indices, deadline=until)
            reference = evaluate(
                source, prompt.indices, client.limits, deadline=min(until, time.monotonic() + 2)
            )
            report["raw"] = dict(
                production=production,
                reference=dict(
                    outcome=reference.outcome,
                    reason=reference.reason,
                    coverage=reference.coverage,
                    visible_match=reference.values == prompt.values,
                ),
                visible_verified=(
                    production.get("outcome") == reference.outcome == "ok"
                    and production.get("values") == list(reference.values) == list(prompt.values)
                ),
            )
            if case["fitting_disabled"]:
                report["fitting"] = dict(outcome="disabled")
            else:
                report["fitting"] = client.fit(
                    source,
                    prompt,
                    policy=case["policy"],
                    deadline=min(deadline, time.monotonic() + 4),
                    seed=case["seed"],
                )
        report["status"] = "COMPLETED"
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        ledger.finish(
            "candidate-replay", "completed" if report["status"] == "COMPLETED" else "interrupted"
        )
        report["elapsed_seconds"] = time.monotonic() - started
        write(output / "budget.json", ledger.snapshot())
        write(output / "summary.json", report)
    return 0 if report["status"] == "COMPLETED" else 1


def replay(args):
    from foundation_local import Host

    case_path = Path(args.evidence).resolve() / "case.json"
    manifest = load(Path(args.evidence).resolve() / "archive-manifest.json")
    ref = next(row for row in manifest["files"] if row["path"] == "case.json")
    if sha(case_path) != ref["sha256"] or case_path.stat().st_size != ref["bytes"]:
        raise ValueError("collected replay case changed")
    case = load(case_path)
    if not case.get("original_source"):
        raise ValueError("original candidate unavailable; collect retained trace before replay")
    output = fresh(Path(args.output).resolve())
    host = Host(args)
    image = host.engine.image(args.image or case["parent_image"])
    # The portable adapter accepts directories only. Stage precisely the two
    # permitted files, outside the writable output mount and original inputs.
    with tempfile.TemporaryDirectory(prefix=".bigint-replay-", dir=output.parent) as directory:
        staged = Path(directory)
        staged.chmod(0o755)
        shutil.copyfile(case_path, staged / "case.json")
        shutil.copyfile(Path(__file__).resolve(), staged / "diagnose.py")
        for path in staged.iterdir():
            path.chmod(0o644)
        if sha(staged / "case.json") != ref["sha256"]:
            raise ValueError("collected replay case changed during staging")
        return _replay_staged(args, host, image, case, case_path, output, staged)


def _replay_staged(args, host, image, case, case_path, output, staged):
    from qualify_bigint_phase5 import launch
    from oeis_learn.container_runtime import private_runtime_environment
    from oeis_learn.synthesis.state import BudgetLedger

    name = "oeis-bigint-replay-" + uuid.uuid4().hex
    command = host.engine.command(
        image=image,
        name=name,
        mounts=[
            (staged, "/work/input", False),
            (output, "/work/output", True),
        ],
        argv=[
            "python",
            "/work/input/diagnose.py",
            "_inside",
            "--case",
            "/work/input/case.json",
            "--output",
            "/work/output",
            "--seconds",
            str(args.seconds),
            "--image",
            image,
        ],
        memory_bytes=host.policy["cpu_memory_bytes"],
        cpus=host.policy["cpus"],
        workdir="/opt/oeis-build",
        auto_remove=False,
        environment={**private_runtime_environment(output), "PYTHONPATH": "/opt/oeis-build/src"},
    )
    error = None
    ledger = BudgetLedger(args.seconds + 45)
    ledger.start("visible-replay-role", args.seconds + 45)
    write(output / "host-budget-start.json", ledger.snapshot())
    write(
        output / "analysis.json",
        dict(
            version="bigint-candidate-analysis/v1",
            case_sha256=sha(case_path),
            parent_contract_sha256=case["parent_contract_sha256"],
            image=image,
            analysis_identity=digest([sha(case_path), image, args.seconds, uuid.uuid4().hex]),
            work_seconds=args.seconds,
            startup_seconds=15,
            cleanup_seconds=30,
            script_sha256=sha(Path(__file__)),
            qualified=False,
        ),
    )
    try:
        launch(command, output, seconds=args.seconds + 15)
        result = load(output / "summary.json")
        if (
            result.get("status") != "COMPLETED"
            or result.get("case_sha256") != sha(case_path)
            or result.get("image") != image
        ):
            raise ValueError("completed image/case-bound replay summary required")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        started_cleanup = time.monotonic()
        try:
            host.engine.remove(name)
        except BaseException as exc:
            error = (error + "; " if error else "") + f"cleanup {type(exc).__name__}: {exc}"
        ledger.finish("visible-replay-role", "interrupted" if error else "completed")
        write(output / "host-budget.json", ledger.snapshot())
        write(
            output / "host-summary.json",
            dict(
                status="FAIL" if error else "COMPLETED",
                error=error,
                cleanup_seconds=time.monotonic() - started_cleanup,
                qualified=False,
            ),
        )
        archive(output)
    if error:
        raise RuntimeError(error)
    return output.with_suffix(".zip")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    c = commands.add_parser("collect", help="read original synthetic tiny run; no engine")
    c.add_argument("--run", required=True, help="workspace-relative completed run directory")
    c.add_argument("--prompt", required=True)
    c.add_argument("--output", required=True)
    r = commands.add_parser(
        "replay", help="one isolated visible candidate; no GPU or scoring truth"
    )
    r.add_argument("--evidence", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--engine", choices=("auto", "docker", "podman"), default="auto")
    r.add_argument("--image", help="default: original run's pinned image")
    r.add_argument("--seconds", type=int, choices=range(10, 121), default=30)
    r.add_argument("--gpu-base-image", default=None)
    r.add_argument("--render-node", default=None)
    r.add_argument("--memory-gib", type=float)
    r.add_argument("--cpus", type=int)
    r.add_argument("--free-disk-gib", type=int, default=5)
    r.add_argument("--host-free-memory-gib", type=float, default=1)
    i = commands.add_parser("_inside", help=argparse.SUPPRESS)
    i.add_argument("--case", required=True)
    i.add_argument("--output", required=True)
    i.add_argument("--image", required=True)
    i.add_argument("--seconds", type=int, choices=range(10, 121), required=True)
    args = p.parse_args()
    if args.command == "collect":
        result = collect(ROOT, args.run, args.prompt, args.output)
    elif args.command == "replay":
        result = replay(args)
    else:
        return inside(args.case, args.output, args.seconds, args.image)
    print(json.dumps(dict(archive=str(result), qualified=False)))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, StopIteration) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
