"""Immutable bounded measurement entry point; CPU fixtures never qualify device decisions.

No arbitrary commands, imported scores, or private truth paths in its input schema.
Live accelerator/pilot orchestration is pending and fails explicitly, without CPU fallback.
"""

from pathlib import Path
import time
from oeis_learn.experiments.artifacts import (
    canonical_bytes,
    load_json,
    compute_canonical_digest as digest,
)
from oeis_learn.experiments.models import check_digest
from oeis_learn.experiments.decisions import frozen_protocol, decision_records
from oeis_learn.tracking.foundation_metrics import EventLog


def implementation_identity():
    root = Path(__file__).resolve().parents[1]
    return digest(
        {str(p.relative_to(root)): digest(p.read_text()) for p in sorted(root.rglob("*.py"))}
    )


def fixture_manifest():
    from oeis_learn.sandbox.pipeline import runtime_identity

    sources = [f"i256.const {n}" for n in range(4)]
    return dict(
        version="measurement/v1",
        mode="cpu_fixture",
        protocol=frozen_protocol(),
        identities=dict(
            code=implementation_identity(),
            runtime=digest(runtime_identity()),
            data=digest(sources),
            model=digest("no_model_cpu_runtime_fixture"),
        ),
        sources=sources,
    )


def validate_manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {
        "version",
        "mode",
        "protocol",
        "identities",
        "sources",
    }:
        raise ValueError(
            "unknown manifest fields; final truth and external score inputs are forbidden"
        )
    if manifest["version"] != "measurement/v1" or manifest["mode"] not in (
        "cpu_fixture",
        "qualified_device",
    ):
        raise ValueError("unknown measurement version/mode")
    if manifest["protocol"] != frozen_protocol():
        raise ValueError("frozen budget, priority or sample schedule mismatch")
    identities = manifest["identities"]
    if not isinstance(identities, dict) or set(identities) != {"code", "runtime", "data", "model"}:
        raise ValueError("complete immutable identities required")
    for name, value in identities.items():
        check_digest(value, name)
    if identities["code"] != implementation_identity():
        raise ValueError("stale code identity")
    sources = manifest["sources"]
    from oeis_learn.sandbox.wat_ast import parse
    from oeis_learn.sandbox.pipeline import runtime_identity

    if not isinstance(sources, list) or not 1 <= len(sources) <= 128:
        raise ValueError("bounded frozen source sample required")
    for source in sources:
        parse(source)
    if identities["data"] != digest(sources) or identities["runtime"] != digest(runtime_identity()):
        raise ValueError("source or runtime identity mismatch")


def _engineering(sources):
    from oeis_learn.sandbox.pipeline import Runtime, Limits
    from oeis_learn.tracking.foundation_metrics import fuel_summary
    from statistics import median

    durations, fuel = {"uncached": [], "cached": []}, []
    service = Runtime()
    for repeat in range(40):
        order = ("cached", "uncached") if repeat % 2 else ("uncached", "cached")
        for variant in order:
            elapsed = 0
            for source in sources:
                active = service if variant == "cached" else Runtime(cache_bytes=0)
                start = time.monotonic_ns()
                result = active.evaluate(source, range(20), Limits())
                elapsed += time.monotonic_ns() - start
                if (
                    result.outcome is not None
                    or result.outputs != result.reference_outputs
                    or len(result.outputs) != 20
                ):
                    raise RuntimeError("correctness gate failed")
                if repeat >= 10:
                    fuel.extend(result.fuel_terms)
            if repeat >= 10:
                durations[variant].append(elapsed)
    return dict(
        warmups=10,
        repeats=30,
        prompts=len(sources),
        declared_prompts=128,
        sample_shortfall=128 - len(sources),
        variant_order="alternating",
        phase="runtime_end_to_end",
        timings_ns=durations,
        medians_ns={k: median(v) for k, v in durations.items()},
        fuel=fuel_summary(fuel, limit=1000000),
        semantic_equivalence=True,
        thermal=dict(state="unavailable", reason="no supported sensor probe"),
        power=dict(state="unavailable", reason="no supported sensor probe"),
    )


def measure(manifest, output, *, dry_run=False):
    validate_manifest(manifest)
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    binding = {"manifest": manifest, "dry_run": dry_run}
    frozen = root / "manifest.json"
    if frozen.exists():
        if load_json(frozen.read_bytes()) != binding:
            raise ValueError("immutable measurement manifest cannot change")
    else:
        from oeis_learn.experiments.artifacts import ArtifactPath, atomic_write

        atomic_write(ArtifactPath(root, "manifest.json"), canonical_bytes(binding))
    target = root / "measurement.json"
    if target.exists():
        return load_json(target.read_bytes())
    log = EventLog(root / "events")
    report = dict(
        version="measurement-result/v1",
        manifest_sha256=digest(binding),
        protocol_sha256=digest(frozen_protocol()),
        qualified=False,
        executed=False,
        measurements=None,
        status="dry_run",
        stop_reason="validated immutable schedule; no work issued",
        decisions=decision_records(manifest["identities"]),
        charged_ns=0,
    )
    if not dry_run:
        if log.events:
            start = log.events["started"]["utc_ns"]
            report.update(
                status="interrupted",
                stop_reason="prior issued work cannot be regenerated or refunded",
                charged_ns=max(0, time.time_ns() - start),
            )
        else:
            log.append("started", dict(utc_ns=time.time_ns(), manifest_sha256=digest(binding)))
            start = time.monotonic_ns()
            if manifest["mode"] == "qualified_device":
                report.update(
                    status="hardware_unavailable",
                    stop_reason="qualified GPU and coordinated pilot adapter pending; no CPU fallback",
                )
            else:
                from oeis_learn.sandbox.contained import run_contained

                state, value = run_contained(
                    _engineering, (manifest["sources"],), timeout_ms=10000, proof=True
                )
                report.update(
                    executed=True,
                    status="diagnostic_complete" if state == "ok" else "interrupted",
                    stop_reason="CPU fixture; declared engineering/GPU/pilot evidence pending"
                    if state == "ok"
                    else value,
                    measurements=value if state == "ok" else None,
                )
            report["charged_ns"] = time.monotonic_ns() - start
    log.append("completed", dict(status=report["status"], charged_ns=report["charged_ns"]))
    from oeis_learn.experiments.artifacts import ArtifactPath, atomic_write

    atomic_write(ArtifactPath(root, "measurement.json"), canonical_bytes(report))
    return report
