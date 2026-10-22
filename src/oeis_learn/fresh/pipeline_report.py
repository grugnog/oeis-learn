"""Diagnostic wall attribution and explicitly partial schedule-weighted projection."""

from collections import defaultdict
import html
import json
from pathlib import Path
import statistics

from .common import read, write
from oeis_learn.tracking.pipeline_profile import exclusive_seconds


def event_rows(root):
    result = []
    for p in Path(root).rglob("events.jsonl"):
        if "trace-events" in p.parts:
            continue
        for line in p.read_text().splitlines():
            try:
                result.append(dict(json.loads(line), path=str(p.relative_to(root))))
            except json.JSONDecodeError:
                continue  # Torn tail from interruption remains in archived raw evidence.
    return result


def pair_attempts(attempts):
    def key(x):
        return x["prompt"], x["mode"], x["position"], x["seed"]

    arms = [{key(x): x for x in attempts.get(arm, [])} for arm in ("ordinary", "telemetry")]
    matching, changed, unpaired = [], [], []
    for k in sorted(set(arms[0]) | set(arms[1])):
        a, b = (x.get(k) for x in arms)
        if a is None or b is None:
            unpaired.append(k)
        elif any(
            a[f] != b[f] for f in ("tokens_sha256", "batch_size", "generation", "fitting", "raw")
        ):
            changed.append(k)
        else:
            matching.append(k)
    return dict(
        matching=len(matching),
        changed=len(changed),
        unpaired=len(unpaired),
        changed_keys=changed,
        unpaired_keys=unpaired,
    )


def report(output):
    output = Path(output)
    work = read(output / "input/work.json")
    recipe = work["recipe"]
    rows = event_rows(output)
    groups, cpu, busy, memory, telemetry = (
        {},
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
        [],
    )
    for arm in ("ordinary", "telemetry"):
        selected = [x for x in rows if f"/arms/{arm}/" in "/" + x["path"]]
        # Compute per recording; timestamps from different roles are not nested.
        totals = defaultdict(float)
        for path in {x["path"] for x in selected}:
            for k, v in exclusive_seconds([x for x in selected if x["path"] == path]).items():
                totals[k] += v
        groups[arm] = dict(totals)
        for x in selected:
            if x["depth"] != 0 or "boundaries" not in x:
                continue
            a, b = x["boundaries"]["start"], x["boundaries"]["end"]
            delta = b["role_cpu_seconds"] - a["role_cpu_seconds"]
            delta += b["reaped_children_cpu_seconds"] - a["reaped_children_cpu_seconds"]
            delta += sum(
                p["cpu_seconds"] for p in b["processes"] if p["pid"] != b["processes"][0]["pid"]
            )
            delta -= sum(
                p["cpu_seconds"] for p in a["processes"] if p["pid"] != a["processes"][0]["pid"]
            )
            cpu[f"{arm}:{x['name']}"].append(max(0, delta) / max(1e-9, x["end"] - x["start"]))
    for p in output.rglob("samples.jsonl"):
        if "trace-events" in p.parts:
            continue
        for line in p.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            label = str(p.parent.relative_to(output))
            gpu = row.get("gpu") or {}
            try:
                value = float(gpu["gpu_busy_percent"])
                if 0 <= value <= 100:
                    busy[label].append(value)
            except (TypeError, ValueError, KeyError):
                pass
            try:
                memory[label].append(int(row["cgroup"]["memory.current"]))
            except (TypeError, ValueError, KeyError):
                pass
    for p in output.rglob("telemetry.json"):
        telemetry.append(dict(path=str(p.relative_to(output)), **read(p)))
    attempts = read(output / "gpu/attempts.json")
    overhead = []
    for group in range(4):
        pair = {}
        for arm in ("ordinary", "telemetry"):
            match = [
                x
                for x in rows
                if x["depth"] == 0
                and x["name"] == "portfolio"
                and x.get("metadata", {}).get("group") == group
                and f"/arms/{arm}/" in "/" + x["path"]
            ]
            if match:
                pair[arm] = match[0]["end"] - match[0]["start"]
        if len(pair) == 2 and pair["ordinary"] > 0:
            overhead.append(
                dict(
                    group=group,
                    ratio=pair["telemetry"] / pair["ordinary"],
                    interpretation="descriptive only until work/dispositions match",
                )
            )
    gpu_summary = read(output / "gpu/summary.json")
    update_pair = gpu_summary.get("updates", {})
    # Compare deterministic fields only; memory/timings intentionally differ.
    keys = ("block", "completed", "losses_sha256", "parameters_sha256")
    update_equal = [
        [{k: x[k] for k in keys} for x in update_pair.get(arm, [])]
        for arm in ("ordinary", "telemetry")
    ]
    update_equal = bool(update_equal[0]) and update_equal[0] == update_equal[1]
    paired = pair_attempts(attempts)
    median = statistics.median([x["ratio"] for x in overhead]) if overhead else None
    projection = schedule_projection(rows, attempts, recipe, gpu_summary)
    result = dict(
        version="bigint-pipeline-analysis/v1",
        qualified=False,
        diagnostic_exclusive_seconds=groups,
        cpu_core_equivalents={k: statistics.mean(v) for k, v in cpu.items()},
        gpu_busy={
            k: dict(mean=statistics.mean(v), maximum=max(v), samples=len(v))
            for k, v in busy.items()
        },
        memory_current_peaks_bytes={k: max(v) for k, v in memory.items()},
        memory_note="shared counters overlap; do not sum RSS/cgroup/GPU allocation",
        telemetry=telemetry,
        paired_attempts=paired,
        paired_updates_equal=update_equal,
        group_observer_ratios=overhead,
        median_descriptive_ratio=median,
        observer_review_required=(median is not None and median > 1.05)
        or paired["changed"] > 0
        or paired["unpaired"] > 0
        or not update_equal,
        schedule_weighted=projection,
        missing_metrics="absent GPU busy or scope entries mean unavailable, never idle/zero",
        uncertainty="one paired pass; no precise tail latency; caps and frozen-checkpoint distribution limit extrapolation",
        recommendation="review matching-work wall shares, GPU gaps and quota/memory evidence before choosing one optimization",
    )
    write(output / "analysis.json", result)
    timeline(output, rows, result)
    return result


def schedule_projection(rows, attempts, recipe, gpu_summary):
    """No invented complete projection when controls/checkpoint cost are missing."""
    ordinary = [x for x in rows if x["depth"] == 0 and "/arms/ordinary/" in "/" + x["path"]]
    updates = sum(x["end"] - x["start"] for x in ordinary if x["name"] == "updates")
    completed = gpu_summary.get("work", {}).get("ordinary", {}).get("updates", 0)
    validation = sum(x["end"] - x["start"] for x in ordinary if x["name"] == "validation")
    cadence, count = recipe["learning"]["validation_every"], recipe["learning"]["max_updates"]
    schedule = (count + cadence - 1) // cadence
    training_seconds = updates / completed * count if completed else None
    validation_seconds = (
        validation * schedule
        if gpu_summary.get("validation", {}).get("ordinary", {}).get("complete")
        else None
    )
    observed = [
        dict(
            mode=x["mode"],
            position=x["position"],
            tokens=x["tokens"],
            batch=x["batch_size"],
            seconds=x["generation_seconds"],
            complete=x["generation"]["complete"],
        )
        for x in attempts.get("ordinary", [])
        if x["mode"] != "template"
    ]
    return dict(
        status="PARTIAL_PROJECTION",
        intended_scope="parent diverse32 schedule; not new cost admission",
        max_updates=count,
        cadence=cadence,
        validations=schedule,
        measured_weighted_seconds=dict(training=training_seconds, validation=validation_seconds),
        observed_decode_distribution=observed,
        unknown_categories=[
            "full eight-attempt shuffled/absent portfolios",
            "periodic greedy-only scheduling and cache history",
            "real checkpoint/export and production logging",
            "early-stop trajectory and changing output-length distribution",
        ],
        total_seconds=None,
        total_fraction=None,
        amdahl_bound=None,
        reason="reduced controls and frozen weights cannot establish complete training critical-path shares; do not normalize known subtotal to100%",
    )


def timeline(output, rows, result):
    top = [x for x in rows if x["depth"] == 0]
    origin = min((x["start"] for x in top), default=0)
    end = max((x["end"] for x in top), default=origin + 1)
    width = max(1, end - origin)
    parts = [
        '<!doctype html><meta charset="utf-8"><title>Bigint pipeline diagnostic</title>',
        "<style>body{font:14px sans-serif;margin:2em}svg{width:100%;min-width:800px}pre{white-space:pre-wrap}rect{fill:#3875aa}</style>",
        "<h1>Pipeline diagnostic</h1><p>Hover over spans for wall times. Diagnostic mix differs from training; missing sensors are not zero utilization.</p>",
        f'<svg viewBox="0 0 1200 {len(top) * 25 + 30}" role="img" aria-label="Exclusive top-level role timeline">',
    ]
    for i, x in enumerate(sorted(top, key=lambda x: x["start"])):
        label = html.escape(x["path"] + ": " + x["name"])
        left = 420 + (x["start"] - origin) / width * 760
        size = max(1, (x["end"] - x["start"]) / width * 760)
        parts.append(
            f'<text x="0" y="{i * 25 + 16}" font-size="10">{label}</text><rect x="{left}" y="{i * 25 + 3}" width="{size}" height="17"><title>{label}: {x["end"] - x["start"]:.3f}s</title></rect>'
        )
    parts.extend(
        [
            "</svg><h2>Measurement summary</h2><pre>",
            html.escape(json.dumps(result, indent=2)),
            "</pre>",
        ]
    )
    (output / "timeline.html").write_text("\n".join(parts))
