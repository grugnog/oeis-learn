"""Bounded installed all-stage rehearsal and exact original HIP capability reuse.

Software diagnostics, device capability and research authority remain separate.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    record,
    reference,
    seal,
)


def inventory():
    import json
    from importlib.resources import files

    return json.loads(
        files("oeis_learn.generalization")
        .joinpath("policies", "progressive-acceptance.json")
        .read_text()
    )


GPU_CASES = tuple(c["id"] for c in inventory()["cases"] if c["device"] == "actual_HIP")


def suite_nodes():
    """Fixed, source-closed roster; never accept an arbitrary command or test list."""
    return tuple(c["node"] for c in inventory()["integrated"]["cases"])


def reconcile_suite(path):
    """Replay individual outcomes, not pytest's caller-supplied summary counts."""
    import math
    import xml.etree.ElementTree as ET

    registered = {c["node"]: c["stage"] for c in inventory()["integrated"]["cases"]}
    path = Path(path)
    if path.is_symlink() or path.stat().st_size > inventory()["integrated"]["maximum_junit_bytes"]:
        raise ValueError("bounded regular JUnit required")
    raw = path.read_bytes()
    if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise ValueError("JUnit declarations forbidden")
    tree = ET.fromstring(raw)
    if tree.tag not in {"testsuites", "testsuite"}:
        raise ValueError("JUnit suite required")
    cases, invalid, seen = [], [], set()
    for entry in tree.iter("testcase"):
        node = entry.get("classname", "").replace(".", "/") + ".py::" + entry.get("name", "")
        if node not in registered or node in seen:
            invalid.append(node)
            continue
        seen.add(node)
        try:
            seconds = float(entry.get("time", ""))
            if not math.isfinite(seconds) or not 0 <= seconds <= inventory()["integrated"]["timeout_seconds"]:
                raise ValueError("invalid duration")
        except ValueError:
            invalid.append(node)
            seconds = 0
        outcomes = [child.tag for child in entry if child.tag in {"failure", "error", "skipped"}]
        cases.append(dict(node=node, stage=registered[node], status="FAIL" if outcomes else "PASS",
                          outcomes=outcomes, elapsed_us=int(seconds * 1e6)))
    missing = sorted(set(registered) - seen)
    # Collection errors are sometimes emitted outside a testcase.
    for outcome in ("error", "failure", "skipped"):
        if len(list(tree.iter(outcome))) > sum(c["outcomes"].count(outcome) for c in cases):
            invalid.append("unattributed-" + outcome)
    return seal(dict(version="progressive-integrated-cases/v1",
                     status="PASS" if not missing and not invalid and all(c["status"] == "PASS" for c in cases) else "FAIL",
                     stages=sorted(set(registered.values())), cases=cases, missing=missing, invalid=invalid))


def _bounded_process(argv, *, cwd, env, log, seconds, maximum):
    """Own the subprocess group through timeout/failure; keep local bounded logs."""
    import signal

    began = time.monotonic()
    with Path(log).open("xb") as stream:
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stream, stderr=stream,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                if time.monotonic() - began >= seconds:
                    raise TimeoutError("acceptance subprocess deadline")
                if Path(log).stat().st_size > maximum:
                    raise ValueError("acceptance subprocess log cap")
                time.sleep(0.1)
            if Path(log).stat().st_size > maximum:
                raise ValueError("acceptance subprocess log cap")
            return process.returncode
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=5)


def run_suite(source_root, output):
    """Build once, run the fixed native/CPU cases installed, export public summaries.

    Independent mechanical fixtures force useful positive and negative paths even
    when a real campaign cannot freeze. This never issues a research launch receipt.
    """
    import zipfile
    import xml.etree.ElementTree as ET
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.pipeline_validation.identity import file_hash

    root, output = Path(source_root).resolve(strict=True), Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    policy = inventory()["integrated"]
    source = digest(collect_snapshot(root))
    began = time.monotonic()
    installed, first_error, cases, code = None, None, None, None
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OEIS_REQUIRE_NATIVE="1",
               OEIS_REQUIRE_BIGINT_BUILD="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    try:
        (output / "metadata").mkdir()
        build = [sys.executable, "setup.py", "egg_info", "--egg-base", str(output / "metadata"),
                 "build", "--build-base", str(output / "build"), "bdist_wheel", "--bdist-dir",
                 str(output / "bdist"), "--dist-dir", str(output / "dist")]
        if _bounded_process(build, cwd=root, env=env, log=output / "build.log",
                            seconds=policy["build_timeout_seconds"], maximum=policy["maximum_log_bytes"]):
            raise ValueError("wheel build failed")
        [wheel] = (output / "dist").glob("*.whl")
        package = output / "installed"
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(package)
        env["PYTHONPATH"] = os.pathsep.join((str(package), str(root)))
        # Probe from outside the checkout. Its exact installed closure must match
        # the current checkout, including tests, locks, contracts and native code.
        probe = (
            "from pathlib import Path; import oeis_learn; "
            "from oeis_learn.experiments.source_snapshot import source_snapshot; "
            "from oeis_learn.generalization.progressive.contracts import atomic_write,digest; "
            "import sys; p=Path(oeis_learn.__file__).resolve(); "
            "assert p.is_relative_to(Path(sys.argv[1])); "
            "atomic_write(Path(sys.argv[2]),dict(source=digest(source_snapshot()),package=str(p)))"
        )
        if _bounded_process([sys.executable, "-c", probe, str(package), str(output / "installed.json")],
                            cwd=output, env=env, log=output / "probe.log", seconds=30,
                            maximum=policy["maximum_log_bytes"]):
            raise ValueError("installed package probe failed")
        installed = load(output / "installed.json")
        if installed["source"] != source:
            raise ValueError("installed source closure mismatch")
        installed = dict(source=installed["source"], wheel_sha256=file_hash(wheel))
        code = _bounded_process(
            [sys.executable, "-m", "pytest", "-q", "-o", "addopts=", "-p", "no:cacheprovider",
             "--basetemp", str(output / "work"), "--junitxml", str(output / "junit.xml"), *suite_nodes()],
            cwd=root, env=env, log=output / "suite.log", seconds=policy["timeout_seconds"],
            maximum=policy["maximum_log_bytes"])
        cases = reconcile_suite(output / "junit.xml")
        if code != 0 or cases["status"] != "PASS":
            raise ValueError("installed case roster failed")
        if digest(collect_snapshot(root)) != source:
            raise ValueError("source changed during acceptance")
    except (OSError, ValueError, TimeoutError, ET.ParseError) as error:
        # Public metadata excludes tracebacks, tensor data and truth-bearing logs.
        first_error = type(error).__name__
    if cases is None:
        cases = seal(dict(version="progressive-integrated-cases/v1", status="FAIL", cases=[],
                          stages=[], missing=list(suite_nodes()), invalid=["incomplete-execution"]))
    atomic_write(output / "cases.json", cases)
    result = seal(dict(version="progressive-integrated-acceptance/v1", source=source,
                       scope="DIAGNOSTIC_ONLY", status="COMPLETE" if first_error is None else "FAILED",
                       qualified=False, hardware_status="PENDING", sustained_status="PENDING",
                       scientific_status="UNMEASURED", installed=installed, cases=reference(output, "cases.json"),
                       exit_code=code, first_error=first_error, actual_elapsed_us=int((time.monotonic()-began)*1e6),
                       limits=policy, junit=reference(output, "junit.xml") if (output / "junit.xml").is_file() else None))
    atomic_write(output / "acceptance.json", result)
    with zipfile.ZipFile(output / "diagnostics.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in ("acceptance.json", "cases.json"):
            raw = (output / name).read_bytes()
            if len(raw) > policy["maximum_export_bytes"] // 2:
                raise ValueError("diagnostic export cap")
            archive.writestr(name, raw)
    return result


def retain_suite(output, destination):
    """Retain the bounded public ZIP before successful pytest temp cleanup."""
    import zipfile

    source = Path(output) / "diagnostics.zip"
    cap = inventory()["integrated"]["maximum_export_bytes"]
    if source.is_symlink() or source.stat().st_size > cap:
        raise ValueError("bounded diagnostic archive required")
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if (len(names) != 2 or set(names) != {"acceptance.json", "cases.json"}
                or sum(info.file_size for info in archive.infolist()) > cap):
            raise ValueError("public acceptance members only")
    with Path(destination).open("xb") as stream:
        stream.write(source.read_bytes())


HARDWARE_CHECKS = ("finite_update", "parameter_change", "full_restore", "deterministic_continuation",
                   "generation", "native_reference_score", "cleanup")


def hardware_plan(root, request, producers):
    """Only original exact Q capabilities can be reused, never campaign authority."""
    import json
    from oeis_learn.generalization.progressive.contracts import verify_refs
    from oeis_learn.pipeline_validation.identity import validate
    from .progressive_qualification import matrix

    validate(request, ("version", "source", "image", "engine", "serial", "hardware", "cases"),
             "progressive-hip-matrix/v1")
    expected = matrix(
        list({c["profile"]["identity"]: c["profile"] for c in request["cases"]}.values()),
        source=request["source"], image=request["image"], engine=request["engine"], hardware=request["hardware"],
        input_lengths=tuple(sorted({c["input_length"] for c in request["cases"]})),
        microbatches=tuple(sorted({c["microbatch"] for c in request["cases"]})))
    if request != expected or len(producers) > 48:
        raise ValueError("exact bounded current matrix required")
    payloads = verify_refs(root, producers, max_bytes=4 << 20)
    by_scope = {}

    def unique_keys(items):
        value = dict(items)
        if len(value) != len(items):
            raise ValueError("duplicate original producer key")
        return value

    for ref, payload in zip(producers, payloads):
        # Parse the verified original bytes, never a second mutable file read.
        producer = json.loads(payload, object_pairs_hook=unique_keys)
        validate(producer, ("version", "source", "scope", "device", "status", "requested", "executed", "measurements"),
                 "progressive-hip-case/v1")
        if (producer["device"] != "actual_HIP" or producer["status"] != "PASS"
                or producer["requested"] != list(HARDWARE_CHECKS) or producer["executed"] != list(HARDWARE_CHECKS)
                or producer["scope"] in by_scope):
            raise ValueError("original complete unique HIP case required")
        measures = producer["measurements"]
        numeric = ("parameters", "peak_allocated_bytes", "cleanup_allocated_bytes", "baseline_allocated_bytes", "elapsed_us")
        if (set(measures) != set(HARDWARE_CHECKS) | set(numeric)
                or any(measures.get(k) is not True for k in HARDWARE_CHECKS)
                or any(type(measures.get(k)) is not int or measures[k] < 0 for k in numeric)
                or not measures["parameters"] or not measures["elapsed_us"]
                or measures["cleanup_allocated_bytes"] > measures["baseline_allocated_bytes"]):
            raise ValueError("original successful HIP measurements required")
        by_scope[producer["scope"]] = (producer, ref)
    reused, delta, elapsed = [], [], 0
    for case in request["cases"]:
        scope = digest([request[k] for k in ("source", "image", "engine", "hardware")] + [case])
        pair = by_scope.get(scope)
        if pair is None or pair[0]["source"] != request["source"]:
            delta.append(case)
        else:
            producer, ref = pair
            reused.append(dict(case=case["identity"], scope=scope, producer=ref))
            elapsed += producer["measurements"]["elapsed_us"]
    return seal(dict(version="progressive-hardware-reuse/v1", matrix=request["identity"],
                     scope="model-update-capability-only", qualified=False, reused=reused, delta=delta,
                     original_execution_us=elapsed, campaign_device_status="PENDING", sustained_status="PENDING"))


def _current_hardware(request):
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from .progressive_qualification import require_hip, hardware_facts

    require_hip()
    if (request["source"] != digest(source_snapshot()) or request["hardware"] != hardware_facts()
            or request["image"] != os.environ.get("OEIS_QUALIFICATION_IMAGE")
            or request["engine"] != os.environ.get("OEIS_QUALIFICATION_ENGINE")):
        raise ValueError("current source/image/engine/HIP runtime differs from requested acceptance")


def run_hardware_delta(root, request, producers, output, *, seconds=900):
    """One finite serial batch; reuse original bytes, execute only absent Q cases.

    This capability delta deliberately cannot authorize campaign launch. Integrated
    device lifecycle/target consumption and sustained runs retain their own owners.
    """
    import shutil

    if type(seconds) is not int or not 1 <= seconds <= 900:
        raise ValueError("bounded hardware delta allowance required")
    began = time.monotonic()
    plan = hardware_plan(root, request, producers)
    _current_hardware(request)
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    atomic_write(output / "request.json", request)
    atomic_write(output / "plan.json", plan)
    originals = []
    # Copy exactly, retaining origin reference and identity; do not reseal PASS.
    for item in plan["reused"]:
        destination = output / "reused" / item["case"].split(":")[1] / "producer.json"
        destination.parent.mkdir(parents=True)
        shutil.copyfile(Path(root) / item["producer"]["path"], destination)
        copied = reference(output, destination)
        if {k: copied[k] for k in ("sha256", "bytes")} != {k: item["producer"][k] for k in ("sha256", "bytes")}:
            raise ValueError("original hardware bytes changed during reuse")
        originals.append(copied)
    executed, first_error = [], None
    for case in plan["delta"]:
        try:
            remaining = seconds - (time.monotonic() - began)
            if remaining <= 0:
                raise TimeoutError("hardware delta allowance exhausted")
            target = output / "delta" / case["identity"].split(":")[1]
            target.mkdir(parents=True)
            code = (
                "from pathlib import Path;import sys;"
                "from oeis_learn.generalization.progressive.contracts import load,digest;"
                "from oeis_learn.pipeline_validation.progressive_qualification import run_case;"
                "from oeis_learn.synthesis.client import SynthesisClient;"
                "r=load(sys.argv[1]);c=next(c for c in r['cases'] if c['identity']==sys.argv[2]);"
                "\nwith SynthesisClient(digest('progressive-acceptance-delta/v1')) as client:"
                "run_case(r,c,Path(sys.argv[3]),client=client)"
            )
            status = _bounded_process(
                [sys.executable, "-c", code, str(output / "request.json"), case["identity"], str(target)],
                cwd=output, env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2])),
                log=target / "process.log", seconds=remaining, maximum=4 << 20)
            if status:
                raise ValueError("hardware delta subprocess failed")
            producer = reference(output, target / "producer.json")
            # Reconcile new originals through the same validator as reused cases.
            check = hardware_plan(output, request, [producer])
            if [c["case"] for c in check["reused"]] != [case["identity"]]:
                raise ValueError("delta producer does not cover requested case")
            originals.append(producer)
            executed.append(case["identity"])
        except (OSError, ValueError, TimeoutError) as error:
            first_error = type(error).__name__
            break
    coverage = hardware_plan(output, request, originals)
    result = seal(dict(version="progressive-hardware-delta/v1", matrix=request["identity"],
                       status="PASS" if not first_error and not coverage["delta"] else "FAIL",
                       scope="model-update-capability-only", qualified=False,
                       campaign_device_status="PENDING", sustained_status="PENDING",
                       reused=plan["reused"], executed=executed, producers=originals,
                       omitted=[c["identity"] for c in coverage["delta"]], first_error=first_error,
                       original_execution_us=plan["original_execution_us"],
                       actual_elapsed_us=int((time.monotonic()-began)*1e6), allowance_seconds=seconds))
    atomic_write(output / "delta.json", result)
    return result


def fixture(root, *, interrupt=True, first_cap_updates=8, weights_only=False, fail=False):
    """Hand-checkable mechanical sources, separate purpose and bounded horizons."""
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.curriculum import Controller
    from oeis_learn.generalization.progressive.generator import structure_id
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.pipeline_validation.progressive_preparation import verify_draw
    from oeis_learn.pipeline_validation.progressive_training import resolve_update_policy

    from .progressive_learner import compatibility

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sources = [
        "(program (return (add $index (int 1))))",
        "(program (return (add $index (int 2))))",
        "(program (return (select (lt $index (int 2)) $index (int 8))))",
        "(program (return (select (lt $index (int 3)) $index (int 9))))",
    ]
    rows = []
    for i, source in enumerate(sources):
        row, reason = verify_draw(
            {
                "source": source,
                "ordinal": i,
                "draw_identity": digest(["acceptance", i]),
                "source_policy": digest("hand-checked diagnostic"),
                "structure": structure_id(source),
                "detector_limit": "forced diagnostic sources; no autonomous discoveries",
            }
        )
        if row is None:
            raise ValueError(reason)
        rows.append(row)
    p = resolve_profile(
        resolve({"version": "bigint-bootstrap/v1"}), shape="smaller_width", seed=13, dropout=0.1
    )
    from .progressive_device_acceptance import selected_profile
    p = selected_profile(p)
    from .progressive_acceptance_limits import measure, storage
    measured = measure(p)
    storage_limits = storage(root, measured, tasks=2)
    atomic_write(root / 'acceptance-limits.json', measured)
    policy = resolve_update_policy(
        p, schedule_horizon_tokens=400000000, schedule="constant", microbatch_size=7
    )
    layout = {
        "bins": ["familiar", "frontier"],
        "probes": {
            "familiar": {"role": "training", "id": rows[0]["draw_identity"]},
            "frontier": {"role": "training", "id": rows[2]["draw_identity"]},
        },
    }
    if fail:
        rows[1] = seal(dict(rows[1], target_source="(program (return (int 99)))"))
    package = seal(
        {
            "version": "progressive-training-input/v1",
            "profile": p,
            "update_policy": policy,
            "shards": [rows],
            "layout": layout,
            "refresh_updates": 8,
            "scope": "DIAGNOSTIC_ONLY",
            "diagnostic_fault": {"slot": "first", "update": 1, "point": "after_reserve"}
            if interrupt
            else None,
        }
    )
    weights_ref = origin_ref = None
    if weights_only:
        # Build an actual original strict checkpoint, then import its exact
        # weights blob. A tensor hash alone is no longer origin evidence.
        import shutil

        ancestor = root / "ancestor"
        ac, at, ach, _ = fixture(ancestor, interrupt=False)
        atomic_write(
            ancestor / "control" / ("pause-" + digest("invocation-0").split(":")[1] + ".json"),
            seal({"campaign": ac["identity"], "requested": True}),
        )
        result, produced = invoke_cpu(ancestor, ac, at[0], ach, ordinal=0)
        if result["mechanical_status"] != "PARTIAL":
            raise ValueError("original diagnostic checkpoint failed: " + str(result))
        checkpoint = load(produced / "role/checkpoint.json")
        state_root = ancestor / "learning" / at[0]["identity"].split(":")[1] / "checkpoints"
        full = load(state_root / checkpoint["payload"]["full_state"]["path"])
        shutil.copyfile(
            state_root / full["states"]["weights"]["path"], root / "imported-weights.pt"
        )
        weights_ref = reference(root, "imported-weights.pt")
        atomic_write(ancestor / "task.json", at[0])
        origin = seal(
            dict(
                version="progressive-initializer-origin/v1",
                checkpoint=weights_ref,
                campaign=reference(root, ancestor / "campaign.json"),
                task=reference(root, ancestor / "task.json"),
                state=reference(root, produced / "role/checkpoint.json"),
                state_root=str(state_root.relative_to(root)),
                exposure=reference(root, ancestor / "exposure.json"),
                parent=None,
            )
        )
        atomic_write(root / "origin.json", origin)
        origin_ref = reference(root, "origin.json")
        package = seal(dict(package, weights=weights_ref, origin=origin_ref))
    atomic_write(root / "training-input.json", package)
    channel = seal(
        {
            "version": "discovery-channel/v1",
            "role": "learner",
            "kind": "training_request",
            "data": {
                "membership": "training",
                "artifact": reference(root, "training-input.json"),
                "compatibility": compatibility(package),
                "profile": "bigint-body/v1",
            },
        }
    )
    atomic_write(root / "training-channel.json", channel)
    controller = Controller(layout, seed=13)
    charge = 16 * len(rows[1 if controller.select()["bin"] == "familiar" else 3]["tokens"])

    def caps(tokens, seconds=120):
        return {
            "tokens": tokens,
            "draws": 0,
            "rejections": 0,
            "attempts": 0,
            "native_work": 0,
            "elapsed_us": seconds * 1000000,
            "disk_bytes": measured['task_bytes'],
            "archive_bytes": 0,
        }

    plans = []
    for i, name in enumerate(("first", "second")):
        plans.append(
            {
                "id": name,
                "stage": "capability" if i == 0 else "diversity",
                "kind": "training",
                "arm": "static",
                "seed": 13,
                "initializer": {
                    "mode": "weights_only" if weights_only else "random",
                    "artifact": weights_ref,
                    "compatibility": compatibility(package),
                }
                if i == 0
                else {
                    "mode": "full_state",
                    "artifact": None,
                    "compatibility": compatibility(package),
                    "parent_slot": "first",
                },
                "token_target": (3 if i == 0 else 5) * charge,
                "caps": caps(first_cap_updates * charge if i == 0 else 8 * charge),
                "startup_us": 1000000,
                "finalization_us": 4000000,
                "policy": digest("diagnostic-request"),
                "slot": name,
                "output": name,
            }
        )
    if origin_ref is not None:
        plans[0]["initializer"]["origin"] = origin_ref
    exposure = record(
        "exposure",
        dict(previous=None, detector_limits=["diagnostic mechanical source"], events=[]),
        created_at="2026-11-08T00:00:00Z",
        purpose="acceptance",
        track="synthetic_strict",
    )
    atomic_write(root / "exposure.json", exposure)
    graph = {
        "selected_stages": ["capability", "diversity"],
        "nodes": [
            {
                "id": n,
                "stage": plans[i]["stage"],
                "kind": "routine",
                "dependencies": [] if i == 0 else ["first"],
                "maximum_visits": 1,
                "caps": caps(8 * charge),
            }
            for i, n in enumerate(("first", "second"))
        ],
        "edges": [
            {
                "source": "first",
                "target": "second",
                "condition": "COMPLETE",
                "action": "ADVANCE_REGISTERED",
            }
        ],
        "diagnostic_pool": caps(0, 0) | {"disk_bytes": 0},
    }
    request_refs = []
    for i, plan in enumerate(plans):
        name = f"request-{i}.json"
        atomic_write(
            root / name,
            {
                "operation": "train",
                "artifacts": {"training": ["training-channel.json"]},
                "image": "sha256:" + "a" * 64,
                "seconds": 120,
                "reservation": caps(0) | {"disk_bytes": 0},
            },
        )
        plan["policy"] = reference(root, name)["sha256"]
        request_refs.append(
            {"name": name, "role": "custodian", "kind": "policy", "artifact": reference(root, name)}
        )
    c = record(
        "campaign",
        {
            "blueprint": digest("diagnostic-lifecycle"),
            "runtime": digest("UNQUALIFIED_CPU_FIXTURE"),
            "graph": graph,
            "tasks": plans,
            "references": [
                dict(
                    name="original-exposure",
                    role="custodian",
                    kind="groups",
                    artifact=reference(root, "exposure.json"),
                )
            ]
            + request_refs
            + (
                [
                    {
                        "name": "imported-weights",
                        "role": "learner",
                        "kind": "checkpoint",
                        "artifact": weights_ref,
                    }
                ]
                if weights_ref
                else []
            )
            + [
                {
                    "name": "training",
                    "role": "learner",
                    "kind": "prepared",
                    "artifact": reference(root, "training-channel.json"),
                }
            ],
            "caps": caps(16 * charge, 240) | {"disk_bytes": storage_limits['required_bytes']},
            "storage": storage_limits,
            "target_policy": digest("diagnostic-no-scientific-authority"),
        },
        created_at="2026-11-08T00:00:00Z",
        purpose="acceptance",
        track="synthetic_strict",
    )
    atomic_write(root / "campaign.json", c)
    tasks = [
        record(
            "task",
            dict(p, campaign=c["identity"]),
            created_at=c["created_at"],
            purpose=c["purpose"],
            track=c["track"],
        )
        for p in plans
    ]
    return c, tasks, channel, charge


def invoke_cpu(
    root, campaign, task, channel, *, ordinal, pythonpath=None, channel_path="training-channel.json"
):
    """Real isolated Python process calls the same installed role dispatcher."""
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.generalization.progressive.projections import describe, project

    from .progressive_roles import REGISTRY

    root = Path(root)
    accounting = root / "accounting"
    control = root / "control"
    control.mkdir(parents=True, exist_ok=True)
    journal = BudgetJournal(accounting, campaign["identity"], campaign["payload"]["caps"])
    remaining = journal.task_remaining(task)
    name = "invocation-" + str(ordinal)
    output = root / name
    from oeis_learn.generalization.progressive.storage import (
        register_invocation,
        usage,
        owned_guard,
    )
    import json

    storage_proof = register_invocation(root, campaign, task, accounting, output)
    state = root / "learning" / task["identity"].split(":")[1]
    guard = owned_guard(
        dict(campaign=campaign, task=task, storage=storage_proof), output=output, state_root=state
    )
    metadata_bytes = len(json.dumps(dict(campaign=campaign, task=task), indent=2).encode()) + (
        64 << 10
    )
    guard.reserve_bytes += metadata_bytes
    guard.check(0, 64)
    output.mkdir()
    with guard.activate():
        project(
            root,
            output / "inputs",
            "train",
            {"training": [describe(root, channel_path)]},
            registry=REGISTRY,
            authorized=[reference(root, channel_path)],
        )
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    start = time.monotonic_ns() // 1000
    cost = dict.fromkeys(remaining, 0)
    cost["elapsed_us"] = remaining["elapsed_us"]
    charge = journal.reserve(
        task["payload"]["id"], "train", ordinal, cost, boot=boot, monotonic_us=start
    )
    projected_bytes, projected_inodes = usage([output / "inputs"])
    storage_proof["prior_bytes"] += projected_bytes + metadata_bytes
    storage_proof["prior_inodes"] += projected_inodes + 64
    authority = seal(
        {
            "version": "progressive-supervisor/v1",
            "campaign": campaign,
            "task": task,
            "clock": {
                "reservation_key": charge["key"],
                "boot": boot,
                "start_us": start,
                "end_us": start + cost["elapsed_us"],
            },
            "invocation": name,
            "storage": storage_proof,
        }
    )
    atomic_write(output / "authority.json", authority)
    states = root / "learning"
    state = states / task["identity"].split(":")[1]
    state.mkdir(parents=True, exist_ok=True)
    parent_state = root / "no-predecessor"
    if task["payload"]["initializer"].get("parent_slot"):
        parent_plan = next(
            p
            for p in campaign["payload"]["tasks"]
            if p["slot"] == task["payload"]["initializer"]["parent_slot"]
        )
        if 'trial_choices' in parent_plan:
            from oeis_learn.generalization.progressive.trial_scopes import resolve
            parent_plan=resolve(parent_plan,task['payload']['trial_choice'])
        parent = record(
            "task",
            dict(parent_plan, campaign=campaign["identity"]),
            created_at=campaign["created_at"],
            purpose=campaign["purpose"],
            track=campaign["track"],
        )
        parent_state = states / parent["identity"].split(":")[1]
    code = """import sys
from pathlib import Path
from oeis_learn.pipeline_validation.progressive_role import execute
from oeis_learn.pipeline_validation.progressive_evidence import failure
try:
 r=execute('train',inputs=Path(sys.argv[1]),output=Path(sys.argv[2]),authority=Path(sys.argv[3]),accounting=Path(sys.argv[4]),control=Path(sys.argv[5]),state_root=Path(sys.argv[6]),predecessor_state=Path(sys.argv[7]))
except Exception as exc:
 from oeis_learn.generalization.progressive.storage import owned_guard
 from oeis_learn.generalization.progressive.contracts import load
 with owned_guard(load(Path(sys.argv[3])),output=Path(sys.argv[2]),state_root=Path(sys.argv[6])).activate():
  r=failure(Path(sys.argv[2]),'train',exc,producer='installed-train')
raise SystemExit(0 if r['mechanical_status']=='COMPLETE' else 2 if r['mechanical_status']=='PARTIAL' else 1)
"""
    env = dict(os.environ)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(output / "inputs"),
            str(output / "role"),
            str(output / "authority.json"),
            str(accounting),
            str(control),
            str(state),
            str(parent_state),
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(1, (remaining["elapsed_us"] // 1000000)),
    )
    # subprocess.run returns only after owned process termination. This is a
    # measured proof, not a worker timestamp or a simulated multi-hour interval.
    end = time.monotonic_ns() // 1000
    journal.commit(charge["key"], boot=boot, monotonic_us=end, termination_proven=True)
    atomic_write(
        output / "process.json",
        seal(
            {
                "returncode": process.returncode,
                "elapsed_us": end - start,
                "stdout": process.stdout[-4096:],
                "stderr": process.stderr[-4096:],
            }
        ),
    )
    from .progressive_evidence import finalize_learning

    if (state / "binding.json").is_file():
        finalize_learning(journal, task, state, output / "role")
    return load(output / "role/result.json"), output


def export_lifecycle(root, outputs, destination):
    """Bounded public metadata export; private tensors, inputs and traces stay local."""
    import json
    import zipfile

    root, destination = Path(root), Path(destination)
    members = {}
    for i, output in enumerate(outputs):
        output = Path(output)
        result = load(output / "role/result.json")
        if seal(result) != result:
            raise ValueError("changed original lifecycle result")
        error = result.get("producer_error")
        public = {
            k: result[k]
            for k in (
                "version",
                "operation",
                "campaign",
                "stage",
                "task",
                "mechanical_status",
                "scientific_status",
                "qualification_status",
                "claim_status",
                "reason",
                "metrics",
            )
        }
        public["original_result"] = reference(root, output / "role/result.json")
        public["producer_error"] = (
            {k: error[k] for k in ("type", "code", "producer", "local_traceback")}
            if error
            else None
        )
        members[f"invocation-{i}/result.json"] = seal(public)
        process = load(output / "process.json")
        members[f"invocation-{i}/process.json"] = seal(
            {k: process[k] for k in ("returncode", "elapsed_us")}
        )
        for name in (
            "milestone.json",
            "accounting-final.json",
            "decision-bundle.json",
            "gain-cost.json",
        ):
            if (output / "role" / name).is_file():
                members[f"invocation-{i}/{name}"] = load(output / "role" / name)
    if len(members) > 4096:
        raise ValueError("finite lifecycle export required")
    encoded = {
        name: json.dumps(value, sort_keys=True, allow_nan=False).encode()
        for name, value in members.items()
    }
    if sum(map(len, encoded.values())) > 32 << 20:
        raise ValueError("lifecycle export exceeds metadata cap")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with (
        destination.open("xb") as stream,
        zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive,
    ):
        for name, blob in encoded.items():
            archive.writestr(name, blob)
    return reference(root, destination)


def receipt(root, outputs, *, checks, archive):
    """Short CPU evidence cannot authorize a research run or simulated duration."""
    root = Path(root)
    measured = [load(Path(p) / "process.json")["elapsed_us"] for p in outputs]
    results = [load(Path(p) / "role/result.json") for p in outputs]
    if any(seal(r) != r for r in results):
        raise ValueError("original sealed role results required")
    paused = [r for r in results if r.get("metrics", {}).get("task_status") == "PAUSED"]
    completed = [r for r in results if r["mechanical_status"] == "COMPLETE"]
    if (
        not paused
        or not any(r.get("task") == p.get("task") for p in paused for r in completed)
        or not any(
            (Path(p) / "role/accounting-final.json").is_file()
            and load(Path(p) / "role/accounting-final.json")["milestone"]["ancestor_tokens"] > 0
            for p in outputs
        )
        or not {"PARTIAL", "FAILED"} <= {r["mechanical_status"] for r in results}
        or not any(r.get("next_action") == "EXPLORE_RESERVED" for r in results)
    ):
        raise ValueError(
            "actual interrupted/resumed/continued and diagnostic terminal evidence required"
        )
    required = {
        "interruption",
        "resume",
        "continuation",
        "partial",
        "failure",
        "eligible_diagnostic",
        "export",
    }
    if set(checks) != required or any(v is not True for v in checks.values()):
        raise ValueError("every shortened lifecycle obligation must pass")
    value = seal(
        {
            "version": "progressive-lifecycle-acceptance/v1",
            "status": "COMPLETE",
            "scope": "CPU DIAGNOSTIC_ONLY",
            "qualified": False,
            "scientific_status": "UNMEASURED",
            "actual_elapsed_us": sum(measured),
            "actual_elapsed_hours": sum(measured) / 3600000000,
            "duration_evidence": "short real subprocesses; no simulated sustained hours",
            "sustained_duration_status": "PENDING",
            "hardware_status": "PENDING",
            "gpu_cases": list(GPU_CASES),
            "sustained_protocol": {
                "tasks": 2,
                "maximum_task_hours": 24,
                "finalization_minutes_minimum": 30,
                "checks": [
                    "original campaign/task caps across invocation changes",
                    "owned boundary pause and interrupt/recovery",
                    "committed vs charged vs discarded exposure",
                    "read-only predecessor and truth isolation",
                    "bounded export preserving first error",
                    "continuous <=60s progress and measured resources",
                ],
                "evidence": [
                    "original supervisor cleanup receipts",
                    "monotonic active intervals and boot identity",
                    "checkpoint and budget journal cuts",
                    "actual HIP device/runtime producer",
                    "public export",
                ],
            },
            "checks": checks,
            "evidence": [reference(root, Path(p) / "role/result.json") for p in outputs],
            "archive": archive,
        }
    )
    atomic_write(root / "lifecycle-acceptance.json", value)
    return value


def invoke_decision_cpu(root, request, *, ordinal, pythonpath=None):
    """Run the actual installed public-only decision producer in a new process."""
    from oeis_learn.generalization.progressive.projections import describe, project

    from .progressive_roles import REGISTRY

    root = Path(root)
    source = root / f"decision-input-{ordinal}.json"
    atomic_write(
        source,
        seal(
            {
                "version": "discovery-channel/v1",
                "kind": "public_evidence",
                "role": "verifier",
                "data": request,
            }
        ),
    )
    output = root / f"decision-{ordinal}"
    output.mkdir()
    project(
        root,
        output / "inputs",
        "publish",
        {"public": [describe(root, source)]},
        registry=REGISTRY,
        authorized=[reference(root, source)],
    )
    env = dict(os.environ)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    code = "from pathlib import Path;import sys;from oeis_learn.pipeline_validation.progressive_role import execute;execute('publish',inputs=Path(sys.argv[1]),output=Path(sys.argv[2]))"
    started = time.monotonic_ns() // 1000
    result = subprocess.run(
        [sys.executable, "-c", code, str(output / "inputs"), str(output / "role")],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
        env=env,
    )
    atomic_write(
        output / "process.json",
        seal(
            {
                "returncode": result.returncode,
                "elapsed_us": time.monotonic_ns() // 1000 - started,
                "stdout": result.stdout[-4096:],
                "stderr": result.stderr[-4096:],
            }
        ),
    )
    if result.returncode:
        raise RuntimeError("installed decision producer failed: " + result.stderr[-2048:])
    return load(output / "role/result.json"), output
