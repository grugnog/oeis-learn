#!/usr/bin/env python3
"""Image-side measurements of qualified APIs, never a production execution profile."""

import argparse
import json
import math
import os
import re
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def summarize(samples):
    if not samples or any(
        type(x) not in (float, int) or not math.isfinite(x) or x < 0 for x in samples
    ):
        raise ValueError("nonempty finite measured seconds required")
    ordered = sorted(samples)
    return dict(
        samples=len(samples),
        median_seconds=statistics.median(ordered),
        p95_seconds=ordered[math.ceil(0.95 * len(ordered)) - 1],
    )


def cases():
    from oeis_learn.synthesis.contracts import decimal_integer

    result = []

    def add(name, source, indices, values, *, reason=None, old=None, result_bits=None):
        input_bits = max(abs(n).bit_length() for n in indices)
        literal_bits = max(
            [abs(int(n)).bit_length() for n in re.findall(r"\(int (-?[0-9]+)\)", source)] or [0]
        )
        result.append(
            dict(
                name=name,
                source=source,
                indices=[decimal_integer(n) for n in indices],
                supplied_bits=max(input_bits, literal_bits),
                input_bits=input_bits,
                literal_bits=literal_bits,
                result_bits=result_bits or max(abs(n).bit_length() for n in values),
                expected=[decimal_integer(n) for n in values] if reason is None else [],
                expected_reason=reason,
                old_source=old,
            )
        )

    add(
        "shared-small-add",
        "(program (return (add (int 42) (int 7))))",
        [0],
        [49],
        old="i256.const 42 i256.const 7 i256.add",
    )
    add(
        "shared-small-cancel",
        "(program (return (sub (int -42) (int -42))))",
        [0],
        [0],
        old="i256.const -42 i256.const -42 i256.sub",
    )
    for width in (64, 128, 255):
        n = (1 << (width - 1)) + 31
        add(
            f"shared-add-{width}",
            f"(program (return (add (int {n}) (int 7))))",
            [0],
            [n + 7],
            old=f"i256.const {n} i256.const 7 i256.add",
        )
    for width in (64, 256, 512, 1024, 4096):
        n = (1 << (width - 1)) + 31
        for sign in (1, -1):
            x = sign * n
            add(f"add-{width}-{sign}", "(program (return (add $index (int 1))))", [x], [x + 1])
        add(f"full-mul-{width}", "(program (return (mul $index $index)))", [n], [n * n])
    n = (1 << 4096) - 1001
    square = "(program (local $b0 bigint) (set $b0 (mul $index $index)) "
    add("generated-16384", square + "(return (mul (get $b0) (get $b0))))", [n], [n**4])
    add("generated-cancellation", square + "(return (sub (get $b0) (get $b0))))", [n], [0])
    add(
        "expected-overflow",
        square + "(return (mul (mul (get $b0) (get $b0)) $index)))",
        [n],
        [],
        reason="integer_limit",
        result_bits=20480,
    )
    add(
        "non-scoring-loop",
        "(program (local $b0 bigint) (local $c0 control) "
        "(set $b0 (int 1)) (loop $c0 (to_control $index) "
        "(set $b0 (mul (get $b0) (to_bigint (increment (get $c0)))))) (return (get $b0)))",
        list(range(100)),
        [math.factorial(n) for n in range(100)],
    )
    add(
        "non-scoring-branch",
        "(program (return (select (lt $index (int 50)) "
        "(mul $index $index) (add (mul $index (int 2)) (int 1)))))",
        list(range(100)),
        [n * n if n < 50 else 2 * n + 1 for n in range(100)],
    )
    return result


def write(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


PROFILE_PHASES = (
    "literal_decode",
    "store_setup",
    "clock_start",
    "guest_init",
    "input_load",
    "link_instantiate",
    "compute",
    "serialize",
    "clock_stop",
    "store_drop",
)


def profile_summary(report, batch):
    """Validate paired evidence independently of native parity labels."""
    samples = report.get("samples", [])
    if report.get("version") != "bigint-measurements/v2" or len(samples) < 2:
        raise ValueError("paired native profile required")
    mismatches = []
    for index, sample in enumerate(samples):
        control = sample["control"]
        if sample.get("order") != ("profiled-first" if index % 2 == 0 else "control-first"):
            raise ValueError("profile pair order mismatch")
        if type(sample.get("parity")) is not bool or "stages" in control["timings"]:
            raise ValueError("ordinary control and explicit parity flag required")
        for timings in (sample["timings"], control["timings"]):
            for key in ("prepare_seconds", "execution_seconds", "total_seconds"):
                summarize([timings[key]])
            if (
                abs(
                    timings["prepare_seconds"]
                    + timings["execution_seconds"]
                    - timings["total_seconds"]
                )
                > 1e-6
            ):
                raise ValueError("aggregate timing mismatch")
        stages = sample["timings"]["stages"]
        if stages.get("policy") != "bigint-runtime-profile/v1" or set(stages["phases"]) != set(
            PROFILE_PHASES
        ):
            raise ValueError("unknown or incomplete phase profile")
        for key in ("terms_started", "terms_finished"):
            if type(stages[key]) is not int or not 0 <= stages[key] <= batch:
                raise ValueError("invalid term count")
        if stages["terms_finished"] > stages["terms_started"]:
            raise ValueError("invented completed terms")
        for phase in stages["phases"].values():
            if type(phase["calls"]) is not int or not 0 <= phase["calls"] <= batch:
                raise ValueError("invalid phase calls")
            summarize([phase["seconds"]])
        summarize([stages["unattributed_seconds"]])
        accounted = (
            sum(p["seconds"] for p in stages["phases"].values()) + stages["unattributed_seconds"]
        )
        if abs(accounted - sample["timings"]["execution_seconds"]) > 1e-6:
            raise ValueError("phase accounting mismatch")
        fields = sorted(
            k
            for k in set(sample["result"]) | set(control["result"])
            if k not in sample["result"]
            or k not in control["result"]
            or sample["result"][k] != control["result"][k]
        )
        if fields or not sample["parity"]:
            mismatches.append(dict(sample=index, fields=fields or ["native_parity_flag"]))
    warm = samples[1:]
    control_warm = {
        key: summarize([s["control"]["timings"][key] for s in warm])
        for key in ("prepare_seconds", "execution_seconds", "total_seconds")
    }
    baseline = control_warm["execution_seconds"]["median_seconds"]
    return dict(
        policy="bigint-runtime-profile/v1",
        paired_results_match=not mismatches,
        mismatches=mismatches,
        pair_order=[s["order"] for s in samples],
        control_warm=control_warm,
        phases={
            name: dict(
                seconds=summarize(
                    [s["timings"]["stages"]["phases"][name]["seconds"] for s in warm]
                ),
                calls=[s["timings"]["stages"]["phases"][name]["calls"] for s in samples],
            )
            for name in PROFILE_PHASES
        },
        unattributed=summarize([s["timings"]["stages"]["unattributed_seconds"] for s in warm]),
        warm_median_execution_ratio=statistics.median(
            s["timings"]["execution_seconds"] for s in warm
        )
        / baseline
        if baseline
        else None,
        overhead_unavailable_reason=None if baseline else "zero-duration control",
        warm_paired_delta_seconds=statistics.median(
            s["timings"]["execution_seconds"] - s["control"]["timings"]["execution_seconds"]
            for s in warm
        ),
        interpretation="Review observer cost and pair order before causal attribution; exclusions are not successful throughput.",
    )


INPUT_LOAD_PHASES = (
    "index_parse",
    "begin_load",
    "magnitude_encode",
    "byte_transfer",
    "finish_load",
    "status_check",
    "seal",
)
INPUT_COUNTERS = (
    "literal_values_started",
    "index_values_started",
    "values_finished",
    "byte_calls_attempted",
    "byte_calls_returned",
)


MEMORY_SCOPES = {
    "memory_before_first": "before_first",
    "memory_in_first": "first_byte",
    "memory_after_first": "after_first",
}
MEMORY_FIELDS = (
    "requests",
    "allowed",
    "denied",
    "grow_failed_callbacks",
    "requested_increase_bytes",
)


def first_byte_summary(samples):
    """Limiter requests are observations, not proof of physical allocation or causality."""
    probes = []
    later_attempted, later_returned = [], []
    for sample in samples:
        stages = sample["timings"]["stages"]
        detail = stages["input_load"]
        p = detail["first_byte_probe"]
        if p.get("policy") != "bigint-first-byte-probe/v1":
            raise ValueError("unknown first-byte policy")

        def count(value, maximum=2**64 - 1):
            if type(value) is not int or not 0 <= value <= maximum:
                raise ValueError("invalid first-byte/memory counter")
            return value

        terms = count(p["terms_observed"], stages["phases"]["input_load"]["calls"])
        if (
            terms != stages["phases"]["input_load"]["calls"]
            or count(p["memory_probe_terms"]) != terms
        ):
            raise ValueError("missing input memory probes")
        first = count(p["first_byte"]["calls"], min(terms, detail["byte_calls_attempted"]))
        returned = count(p["first_byte_calls_returned"], min(first, detail["byte_calls_returned"]))
        later_attempted.append(detail["byte_calls_attempted"] - first)
        later_returned.append(detail["byte_calls_returned"] - returned)
        if detail["byte_calls_attempted"] and not first:
            raise ValueError("attempted bytes without a first-byte call")
        if later_returned[-1] > later_attempted[-1]:
            raise ValueError("invented later byte returns")
        summarize([p["first_byte"]["seconds"], p["remaining_transfer_seconds"]])
        if not first and p["first_byte"]["seconds"] != 0:
            raise ValueError("first-byte time without calls")
        if (
            abs(
                p["first_byte"]["seconds"]
                + p["remaining_transfer_seconds"]
                - detail["phases"]["byte_transfer"]["seconds"]
            )
            > 1e-6
        ):
            raise ValueError("first-byte times do not partition transfer")
        total_requests = 0
        traced = {scope: dict.fromkeys(MEMORY_FIELDS, 0) for scope in MEMORY_SCOPES.values()}
        for name in MEMORY_SCOPES:
            c = p[name]
            for field in MEMORY_FIELDS:
                count(c[field])
            if c["allowed"] + c["denied"] != c["requests"]:
                raise ValueError("unbalanced memory request counters")
            if c["requested_increase_bytes"] > c["requests"] * 2**32:
                raise ValueError("invented wasm32 growth")
            total_requests += c["requests"]
        if not first and any(p["memory_in_first"][field] for field in MEMORY_FIELDS):
            raise ValueError("growth in a missing first byte")
        events = p["memory_events"]
        omitted = count(p["memory_events_omitted"])
        if (
            not isinstance(events, list)
            or not min(total_requests, 4) <= len(events) <= 16
            or len(events) + omitted != total_requests
        ):
            raise ValueError("unbounded or unaccounted memory trace")
        previous_term, previous_ordinal, previous_scope = 0, 0, -1
        for e in events:
            term = count(e["term"], terms)
            ordinal = count(e["ordinal"], 4)
            scope = e["scope"]
            if scope not in traced or term == 0 or ordinal == 0:
                raise ValueError("unknown memory trace location")
            scope_index = list(MEMORY_SCOPES.values()).index(scope)
            if (
                term < previous_term
                or ordinal != (previous_ordinal + 1 if term == previous_term else 1)
                or (term == previous_term and scope_index < previous_scope)
            ):
                raise ValueError("unordered memory trace")
            previous_term, previous_ordinal, previous_scope = term, ordinal, scope_index
            current = count(e["current_bytes"], 2**32)
            desired = count(e["desired_bytes"], 2**32)
            if current > desired or type(e["allowed"]) is not bool:
                raise ValueError("invalid memory request")
            c = traced[scope]
            c["requests"] += 1
            c["allowed" if e["allowed"] else "denied"] += 1
            c["requested_increase_bytes"] += desired - current
        for name, scope in MEMORY_SCOPES.items():
            for field in ("requests", "allowed", "denied", "requested_increase_bytes"):
                if traced[scope][field] > p[name][field] or (
                    not omitted and traced[scope][field] != p[name][field]
                ):
                    raise ValueError("memory trace disagrees with scope counters")
        probes.append(p)
    return dict(
        policy="bigint-first-byte-probe/v1",
        first_byte_warm=summarize([p["first_byte"]["seconds"] for p in probes[1:]]),
        remaining_transfer_warm=summarize([p["remaining_transfer_seconds"] for p in probes[1:]]),
        first_byte_calls=[p["first_byte"]["calls"] for p in probes],
        first_byte_calls_returned=[p["first_byte_calls_returned"] for p in probes],
        later_byte_calls_attempted=later_attempted,
        later_byte_calls_returned=later_returned,
        **{
            name: {field: [p[name][field] for p in probes] for field in MEMORY_FIELDS}
            for name in MEMORY_SCOPES
        },
        **{
            name: [p[name] for p in probes]
            for name in ("terms_observed", "memory_probe_terms", "memory_events_omitted")
        },
        interpretation="First-call time is inside byte_transfer; the remainder includes later calls and loop/observer overhead. Growth scopes stop at input completion. Allowed limiter requests do not prove successful physical allocation, copy time or the cause of latency.",
    )


def input_load_summary(report, batch):
    """Nested observation partitions its parent; never count it twice in execution."""
    samples = report.get("samples", [])
    if report.get("profile_input_load") is not True or len(samples) < 2:
        raise ValueError("requested paired input-load profiling required")
    policies = {s["timings"]["stages"]["input_load"].get("policy") for s in samples}
    if len(policies) != 1 or not policies <= {
        "bigint-input-load-profile/v1",
        "bigint-input-load-profile/v2",
    }:
        raise ValueError("mixed or unknown input-load policies")
    policy = next(iter(policies))
    for sample in samples:
        parent = sample["timings"]["stages"]["phases"]["input_load"]
        detail = sample["timings"]["stages"]["input_load"]
        if (policy == "bigint-input-load-profile/v2") != ("first_byte_probe" in detail):
            raise ValueError("first-byte record/policy mismatch")
        if set(detail["phases"]) != set(INPUT_LOAD_PHASES):
            raise ValueError("unknown or incomplete input-load profile")
        max_values = batch * (4096 + 1)  # bounded AST literals plus one actual index per term
        for name in INPUT_COUNTERS:
            if type(detail[name]) is not int or not 0 <= detail[name] <= max_values * 512:
                raise ValueError("invalid input-load counter")
        literals, indices = detail["literal_values_started"], detail["index_values_started"]
        started = literals + indices
        if literals > batch * 4096 or indices > batch or detail["values_finished"] > started:
            raise ValueError("invented input values")
        if (
            not 0
            <= detail["byte_calls_returned"]
            <= detail["byte_calls_attempted"]
            <= started * 512
        ):
            raise ValueError("invented byte calls")
        for phase in detail["phases"].values():
            if type(phase["calls"]) is not int or not 0 <= phase["calls"] <= max_values:
                raise ValueError("invalid input phase calls")
            summarize([phase["seconds"]])
            if phase["calls"] == 0 and phase["seconds"] != 0:
                raise ValueError("time without input phase calls")
        counts = {k: v["calls"] for k, v in detail["phases"].items()}
        if counts["begin_load"] != started or not (
            counts["begin_load"]
            >= counts["magnitude_encode"]
            >= counts["byte_transfer"]
            >= counts["finish_load"]
            >= counts["status_check"]
            >= detail["values_finished"]
        ):
            raise ValueError("input phase order/count mismatch")
        if not indices <= counts["index_parse"] <= parent["calls"] or counts["seal"] > indices:
            raise ValueError("input index/seal count mismatch")
        if counts["byte_transfer"] > detail["byte_calls_attempted"]:
            raise ValueError("byte transfer without attempted calls")
        summarize([parent["seconds"]])
        summarize([detail["unattributed_seconds"]])
        accounted = (
            sum(p["seconds"] for p in detail["phases"].values()) + detail["unattributed_seconds"]
        )
        if abs(accounted - parent["seconds"]) > 1e-6:
            raise ValueError("nested input timings overlap or exceed their parent")
    warm = samples[1:]
    parent = summarize([s["timings"]["stages"]["phases"]["input_load"]["seconds"] for s in warm])
    parent_median = parent["median_seconds"]
    phases = {}
    for name in INPUT_LOAD_PHASES:
        summary = summarize(
            [s["timings"]["stages"]["input_load"]["phases"][name]["seconds"] for s in warm]
        )
        phases[name] = dict(
            seconds=summary,
            calls=[s["timings"]["stages"]["input_load"]["phases"][name]["calls"] for s in samples],
            warm_fraction_of_input_load=summary["median_seconds"] / parent_median
            if parent_median
            else None,
        )
    return dict(
        policy=policy,
        **(
            {"first_byte_probe": first_byte_summary(samples)}
            if policy == "bigint-input-load-profile/v2"
            else {}
        ),
        parent_input_load_warm=parent,
        phases=phases,
        counters={
            name: [s["timings"]["stages"]["input_load"][name] for s in samples]
            for name in INPUT_COUNTERS
        },
        unattributed=summarize(
            [s["timings"]["stages"]["input_load"]["unattributed_seconds"] for s in warm]
        ),
        interpretation="Nested times partition input_load and are not additional execution; review detailed/ordinary distortion and partial calls. Returned byte calls do not prove bytes were accepted after a sticky failure.",
    )


def lookup_summary(report):
    """Compare ordinary cached/baseline execution, separately from observation cost."""
    if (
        report.get("host_lookup_comparison")
        != dict(version="per-term-host-lookups/v1", cached="lazy-per-term", baseline="per-call")
        or len(report.get("samples", [])) < 2
    ):
        raise ValueError("complete host lookup comparison required")
    mismatches = []
    for index, sample in enumerate(report["samples"]):
        baseline = sample["lookup_baseline"]
        if baseline.get("order") != ("baseline-first" if index % 2 == 0 else "baseline-last"):
            raise ValueError("host lookup baseline order mismatch")
        if type(baseline.get("parity")) is not bool:
            raise ValueError("explicit lookup parity flag required")
        for arm in (baseline, sample["control"]):
            timings = arm["timings"]
            if "stages" in timings:
                raise ValueError("lookup comparison requires ordinary calls")
            for key in ("prepare_seconds", "execution_seconds", "total_seconds"):
                summarize([timings[key]])
            if (
                abs(
                    timings["prepare_seconds"]
                    + timings["execution_seconds"]
                    - timings["total_seconds"]
                )
                > 1e-6
            ):
                raise ValueError("lookup baseline timing mismatch")
        fields = sorted(
            k
            for k in set(baseline["result"]) | set(sample["control"]["result"])
            if k not in baseline["result"]
            or k not in sample["control"]["result"]
            or baseline["result"][k] != sample["control"]["result"][k]
        )
        if fields or not baseline["parity"]:
            mismatches.append(dict(sample=index, fields=fields or ["native_parity_flag"]))
    warm = report["samples"][1:]
    summaries = {
        arm: {
            key: summarize([s[arm]["timings"][key] for s in warm])
            for key in ("prepare_seconds", "execution_seconds", "total_seconds")
        }
        for arm in ("control", "lookup_baseline")
    }
    old = summaries["lookup_baseline"]["execution_seconds"]["median_seconds"]
    return dict(
        policy="per-term-host-lookups/v1",
        all_results_match=not mismatches,
        mismatches=mismatches,
        baseline_order=[s["lookup_baseline"]["order"] for s in report["samples"]],
        ordinary_cached_warm=summaries["control"],
        ordinary_baseline_warm=summaries["lookup_baseline"],
        cached_to_lookup_execution_ratio=summaries["control"]["execution_seconds"]["median_seconds"]
        / old
        if old
        else None,
        unavailable_reason=None if old else "zero-duration lookup baseline",
        warm_paired_savings_seconds=statistics.median(
            s["lookup_baseline"]["timings"]["execution_seconds"]
            - s["control"]["timings"]["execution_seconds"]
            for s in warm
        ),
        interpretation="Ordinary calls isolate lookup caching from detailed observation cost; review order/noise and exclusions before claiming a speedup.",
    )


def measurement_results(sample):
    rows = [sample["result"]]
    rows += [sample[name]["result"] for name in ("control", "lookup_baseline") if name in sample]
    return rows


def growth_comparison_summary(zero, reserved, order, batch):
    """A distinct engine experiment; no control result can silently qualify production."""
    from oeis_learn.synthesis.contracts import identity

    if order not in ("zero-first", "reserved-first") or len(zero["samples"]) != len(
        reserved["samples"]
    ):
        raise ValueError("invalid growth arm order/count")
    configurations, profiles, nested, rss = {}, {}, {}, {}
    guest_hash = zero["engine_configuration"]["manifest"]["runtime_wasm_sha256"]
    if not isinstance(guest_hash, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", guest_hash):
        raise ValueError("invalid growth guest identity")
    for name, report, reserve in (("zero", zero, 0), ("reserved", reserved, 65536)):
        record = report["engine_configuration"]
        expected = dict(
            version="bigint-engine-configuration/v1",
            purpose="qualification-growth-control" if reserve else "production",
            runtime_wasm_sha256=guest_hash,
            wasmtime="20.0.2",
            consume_fuel=True,
            epoch_interruption=True,
            wasm_threads=False,
            parallel_compilation=False,
            static_memory_maximum_size=0,
            static_memory_guard_size=0,
            dynamic_memory_guard_size=0,
            dynamic_memory_reserved_for_growth=reserve,
        )
        # Canonical identity compares JSON types too (true must not masquerade as1).
        if identity(record["manifest"]) != identity(expected) or record["sha256"] != identity(
            expected
        ):
            raise ValueError("unbound or changed growth engine configuration")
        configurations[name] = record["sha256"]
        if (
            type(report["peak_process_rss_bytes"]) is not int
            or not 0 < report["peak_process_rss_bytes"] <= 16 << 30
        ):
            raise ValueError("bounded Linux process RSS required for each arm")
        rss[name] = report["peak_process_rss_bytes"]
        summarize([report["runtime_compile_seconds"]])
        profiles[name] = profile_summary(report, batch)
        nested[name] = input_load_summary(report, batch)
        if nested[name]["policy"] != "bigint-input-load-profile/v2":
            raise ValueError("growth comparison requires first-byte v2 evidence")
    mismatches = []
    for index, (a, b) in enumerate(zip(zero["samples"], reserved["samples"])):
        fields = sorted(
            k
            for k in set(a["control"]["result"]) | set(b["control"]["result"])
            if k not in a["control"]["result"]
            or k not in b["control"]["result"]
            or a["control"]["result"][k] != b["control"]["result"][k]
        )
        if fields:
            mismatches.append(dict(sample=index, fields=fields))
    ordinary = {name: p["control_warm"] for name, p in profiles.items()}
    denominator = ordinary["zero"]["execution_seconds"]["median_seconds"]
    policy = dict(
        version="bigint-growth-reservation-comparison/v1",
        configurations=configurations,
        resource_identity=zero["samples"][0]["control"]["result"]["resource_identity"],
        process_policy="separate-native-process-per-setting-and-row",
        order_policy="alternate-corpus-row",
        repetitions=len(zero["samples"]),
    )
    return dict(
        **policy,
        comparison_sha256=identity(policy),
        order=order,
        all_results_match=not mismatches
        and all(p["paired_results_match"] for p in profiles.values()),
        mismatches=mismatches,
        ordinary_warm=ordinary,
        ordinary_cold={
            name: report["samples"][0]["control"]["timings"]
            for name, report in (("zero", zero), ("reserved", reserved))
        },
        runtime_compile_seconds={
            name: report["runtime_compile_seconds"]
            for name, report in (("zero", zero), ("reserved", reserved))
        },
        peak_process_rss_bytes=rss,
        first_byte_warm={
            name: n["first_byte_probe"]["first_byte_warm"] for name, n in nested.items()
        },
        observer_ratios={name: p["warm_median_execution_ratio"] for name, p in profiles.items()},
        warm_execution_ratio_reserved_over_zero=ordinary["reserved"]["execution_seconds"][
            "median_seconds"
        ]
        / denominator
        if denominator
        else None,
        interpretation="Qualification-only counterfactual. Review ordinary prepare/execute/total and lifecycle, first byte, observer distortion and exclusions. Per-process VmHWM includes compilation/JIT and both observations, measured before report serialization; it is not aggregate RSS or guest-only allocation. Production remains zero reservation.",
    )


def memory_growth_measurements(
    case, batch, repetitions, *, order, failure_output, progress_output=None
):
    """Separate subprocesses prevent one setting's JIT/RSS from contaminating the other."""
    if order not in ("zero-first", "reserved-first"):
        raise ValueError("invalid growth arm order")
    arms = {}
    for name in ("zero", "reserved") if order == "zero-first" else ("reserved", "zero"):
        if progress_output is not None:
            write(
                progress_output,
                dict(
                    status="FAIL",
                    case=case["name"],
                    batch=batch,
                    order=order,
                    active_arm=name,
                    arms=arms,
                ),
            )
        arms[name] = runtime_measurements(
            case,
            batch,
            repetitions,
            profile=True,
            profile_input_load=True,
            growth_reservation_control=name == "reserved",
            failure_output=failure_output,
            native_output=Path(failure_output).with_name("growth-latest-native.log"),
        )
        if progress_output is not None:
            write(
                progress_output,
                dict(
                    status="FAIL",
                    case=case["name"],
                    batch=batch,
                    order=order,
                    completed_arm=name,
                    arms=arms,
                ),
            )
    comparison = growth_comparison_summary(arms["zero"], arms["reserved"], order, batch)
    if progress_output is not None:
        write(
            progress_output,
            dict(
                status="PASS" if comparison["all_results_match"] else "FAIL",
                case=case["name"],
                batch=batch,
                order=order,
                comparison_sha256=comparison["comparison_sha256"],
                arms=arms,
            ),
        )
    comparison["reserved"] = arms["reserved"]
    arms["zero"]["memory_growth"] = comparison
    return arms["zero"]


def runtime_measurements(
    case,
    batch,
    repetitions,
    *,
    profile=False,
    compare_host_lookups=False,
    profile_input_load=False,
    growth_reservation_control=False,
    failure_output=None,
    native_output=None,
):
    """Native compilation and execution are timed inside the production runtime."""
    from oeis_learn.synthesis.reference import evaluate

    if (compare_host_lookups or profile_input_load) and not profile:
        raise ValueError("host lookup/input-load details require runtime profiling")

    if growth_reservation_control and (
        not profile or not profile_input_load or compare_host_lookups
    ):
        raise ValueError("growth control requires isolated input profiling")

    # Repeated exact indices measure batch overhead, never pretend to be a dataset.
    indices = (case["indices"] * batch)[:batch]
    started = time.monotonic()
    binary = os.environ.get("OEIS_BENCHMARK_RUNTIME", "/opt/oeis-build-info/benchmark-runtime")
    proc = subprocess.run(
        [binary],
        input=json.dumps(
            dict(
                source=case["source"],
                indices=indices,
                repetitions=repetitions,
                **({"profile": True} if profile else {}),
                **({"compare_host_lookups": True} if compare_host_lookups else {}),
                **({"profile_input_load": True} if profile_input_load else {}),
                **({"growth_reservation_control": True} if growth_reservation_control else {}),
            )
        ).encode(),
        capture_output=True,
        timeout=120,
    )
    subprocess_seconds = time.monotonic() - started
    if native_output is not None:
        target = Path(native_output)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(proc.stdout[: 4 << 20])
        temporary.replace(target)
    if proc.returncode or len(proc.stdout) > 4 << 20:
        raise RuntimeError(
            "bounded native measurement failed: " + proc.stderr[:1000].decode(errors="replace")
        )
    try:
        report = json.loads(proc.stdout)
        if not isinstance(report, dict):
            raise ValueError("native measurement report must be an object")
        if (
            report.get("version")
            != ("bigint-measurements/v2" if profile else "bigint-measurements/v1")
            or len(report.get("samples", [])) != repetitions
        ):
            raise ValueError("incomplete native measurement report")
        if profile_input_load:
            report["input_load_profiling"] = input_load_summary(report, batch)
            if report["input_load_profiling"]["policy"] != "bigint-input-load-profile/v2":
                raise ValueError("current native input profiling requires first-byte v2 evidence")
        elif report.get("profile_input_load") is True or any(
            s.get("timings", {}).get("stages", {}).get("input_load") is not None
            for s in report["samples"]
        ):
            raise ValueError("unrequested nested input profiling")
        if compare_host_lookups:
            report["host_lookups"] = lookup_summary(report)
        elif report.get("host_lookup_comparison") is not None or any(
            "lookup_baseline" in s for s in report["samples"]
        ):
            raise ValueError("unrequested host lookup comparison")
        if profile:
            report["profiling"] = profile_summary(report, batch)
            if not report["profiling"]["paired_results_match"] or (
                compare_host_lookups and not report["host_lookups"]["all_results_match"]
            ):
                report.update(case=case["name"], batch=batch)
                from oeis_learn.synthesis.contracts import identity

                for sample in report["samples"]:
                    for row in measurement_results(sample):
                        row["values_sha256"] = identity(row.pop("values"))
                return report
    except (ValueError, KeyError, TypeError, AttributeError, OverflowError):
        if failure_output is not None:
            target = Path(failure_output)
            temporary = target.with_suffix(target.suffix + ".tmp")
            temporary.write_bytes(proc.stdout)  # already bounded to4MiB above
            temporary.replace(target)
        raise
    reference_times = []
    for _ in range(repetitions):
        start = time.monotonic()
        reference = evaluate(case["source"], indices)
        reference_times.append(time.monotonic() - start)
        if case["expected_reason"]:
            if reference.reason != case["expected_reason"]:
                raise RuntimeError("independent failure classification disagrees")
        elif (
            reference.outcome != "ok"
            or list(reference.values) != (case["expected"] * batch)[:batch]
        ):
            raise RuntimeError("independent source interpreter disagrees")
    reference_stats = summarize(reference_times)
    expected = (case["expected"] * batch)[:batch]
    exclusions = []
    for sample in report["samples"]:
        row = sample["result"]
        if case["expected_reason"]:
            if row.get("reason") != case["expected_reason"] or row.get("values"):
                raise RuntimeError("expected overflow was not classified correctly")
        elif row.get("outcome") != "ok":
            if row.get("reason") not in {
                "term_fuel",
                "aggregate_fuel",
                "deadline",
                "arena_limit",
                "memory_limit",
                "output_limit",
            } or row.get("values"):
                raise RuntimeError("unexpected production failure on a legal arithmetic case")
            exclusions.append(dict(reason=row["reason"], coverage=row.get("coverage")))
        elif row.get("values") != expected:
            raise RuntimeError("production/independent expectation disagreement")
    if case["expected_reason"]:
        if reference.reason != case["expected_reason"]:
            raise RuntimeError("independent failure classification disagrees")
    elif reference.outcome != "ok" or list(reference.values) != expected:
        raise RuntimeError("independent source interpreter disagrees")
    report.update(
        case=case["name"],
        batch=batch,
        supplied_bits=case["supplied_bits"],
        input_bits=case.get("input_bits"),
        literal_bits=case.get("literal_bits"),
        result_bits=case["result_bits"],
        expected_reason=case["expected_reason"],
        resource_exclusions=exclusions,
        subprocess_seconds=subprocess_seconds,
        reference_seconds=reference_stats["median_seconds"],
        reference_summary=reference_stats,
        cold=report["samples"][0]["timings"],
        warm={
            key: summarize([s["timings"][key] for s in report["samples"][1:]])
            for key in ("prepare_seconds", "execution_seconds", "total_seconds")
        },
    )
    # Full outputs are independently checked above; preserve hashes, not megabytes.
    from oeis_learn.synthesis.contracts import identity

    for sample in report["samples"]:
        for result in measurement_results(sample):
            result["values_sha256"] = identity(result.pop("values"))
    if case["old_source"]:
        from oeis_learn.sandbox.pipeline import Runtime, PreparedCandidate

        runtime = Runtime(backend="rust")
        old_prepare, old_execute = [], []
        for _ in range(repetitions):
            start = time.monotonic()
            prepared = runtime.prepare(case["old_source"])
            old_prepare.append(time.monotonic() - start)
            if not isinstance(prepared, PreparedCandidate):
                raise RuntimeError("retained native compilation failed")
            start = time.monotonic()
            values = [runtime.production_term(prepared, 0, 1_000_000) for _ in range(batch)]
            old_execute.append(time.monotonic() - start)
            if any(
                failure is not None or str(v) != case["expected"][0] for v, _, failure in values
            ):
                raise RuntimeError("retained small-value semantics disagreed")
        old = summarize(old_execute[1:])
        comparison = report["profiling"]["control_warm"] if profile else report["warm"]
        ratio = comparison["execution_seconds"]["median_seconds"] / max(
            old["median_seconds"], 1e-12
        )
        report["retained"] = dict(
            prepare=summarize(old_prepare[1:]),
            execution=old,
            small_value_slowdown=ratio,
            ratio_basis="ordinary-control" if profile else "aggregate-measured",
            investigation_required=ratio > 5,
        )
    else:
        report["retained"] = dict(
            excluded=True, reason="outside declared shared add/sub constant subset"
        )
    return report


def fitting_measurements(repetitions):
    """Time provisional native fit and independent admission separately, no trial callbacks."""
    from oeis_learn.synthesis.client import SynthesisClient
    from oeis_learn.synthesis.contracts import identity, decimal_integer
    from oeis_learn.synthesis.admission import VisiblePrompt, admit

    fixtures = [
        (
            "affine-large",
            "(program (return (add (mul (param.get p0) $index) (param.get p1))))",
            [str(7 * n + 100000000000000000001) for n in range(20)],
            "assignment_found",
        ),
        (
            "coefficient4096",
            "(program (return (param.get p0)))",
            [decimal_integer((1 << 4096) - 1)] * 20,
            "assignment_found",
        ),
        (
            "rank-deficient",
            "(program (return (add (param.get p0) (param.get p1))))",
            ["3"] * 20,
            "assignment_found",
        ),
        (
            "nonlinear-shared",
            "(program (return (mul (param.get p0) (param.get p0))))",
            ["4"] * 20,
            "assignment_found",
        ),
        (
            "inconsistent",
            "(program (return (param.get p0)))",
            list(map(str, range(20))),
            "unsat_in_scope",
        ),
        (
            "tainted-control",
            "(program (return (select (lt (param.get p0) $index) (int 1) (int 2))))",
            ["1"] * 20,
            "unsupported",
        ),
    ]
    samples = []
    complete = True
    with SynthesisClient(identity("phase5-measurement-fit/v1")) as client:
        runtime = client.runtime_identity("bigint-body/v1")
        for name, source, values, expected in fixtures:
            prompt = VisiblePrompt(tuple(map(str, range(20))), tuple(values))
            for i in range(repetitions):
                start = time.monotonic()
                response = client.request(
                    "fit",
                    [dict(candidate_id=str(i), source=source)],
                    prompt.indices,
                    prompt.values,
                    deadline=start + 2,
                )
                native_seconds = time.monotonic() - start
                rows = response["results"]
                native = (
                    rows[0]
                    if len(rows) == 1 and response["complete"]
                    else dict(outcome="resource_exhausted")
                )
                outcome = native["outcome"]
                if outcome not in {expected, "unknown", "resource_exhausted"}:
                    raise RuntimeError(
                        "native fitting fixture unexpected outcome: " + name + ":" + outcome
                    )
                complete &= outcome == expected
                reference_seconds = None
                if outcome == "assignment_found":
                    start = time.monotonic()
                    admitted = [
                        admit(
                            row,
                            prompt,
                            runtime_sha256=runtime,
                            limits=client.limits,
                            deadline=start + 2,
                        )
                        for row in native["assignments"]
                    ]
                    reference_seconds = time.monotonic() - start
                    if not admitted or not all(r.outcome == "verified_solution" for r in admitted):
                        raise RuntimeError("non-scoring fit independent admission failed: " + name)
                samples.append(
                    dict(
                        fixture=name,
                        outcome=outcome,
                        native_seconds=native_seconds,
                        reference_seconds=reference_seconds,
                    )
                )
    return dict(
        samples=samples,
        native=summarize([r["native_seconds"] for r in samples]),
        reference=summarize(
            [r["reference_seconds"] for r in samples if r["reference_seconds"] is not None]
        ),
        complete=complete,
    )


def neural_fixtures(shapes=None):
    """Capacity matrix plus a separate shape-matched, non-scoring cost fixture."""
    from oeis_learn.decoder.bigint_codec import encode_body

    matrix = [
        ("capacity", 64, 33),
        ("capacity", 256, 152),
        ("capacity", 1024, 560),
        ("capacity", 4096, 1024),
    ]
    if shapes is not None:
        width, tokens = shapes["max_input_bits"], shapes["max_tokens"]
        if (
            type(width) is not int
            or not 1 <= width <= 4096
            or type(tokens) is not int
            or not 1 <= tokens <= 1024
        ):
            raise ValueError("bounded prepared input/token shapes required")
        matrix.append(("preflight", width, tokens))
    fixtures = []
    for role, width, goal in matrix:
        count = min(59, max(1, math.ceil((goal - 16) / 17)))
        source = (
            "(program (local $b0 bigint) "
            + "(set $b0 (add (get $b0) (int 1))) " * count
            + "(return (get $b0)))"
        )
        tokens = encode_body(source)
        if len(tokens) < goal:
            source = source.replace("(int 1)", "(int " + "1" * (1 + goal - len(tokens)) + ")", 1)
            tokens = encode_body(source)
        base = 1 << (width - 1)
        fixtures.append(
            dict(
                fixture_role=role,
                values=[str((-1 if n % 2 else 1) * (base + n % min(base, 20))) for n in range(20)],
                tokens=tokens,
            )
        )
    return fixtures


def parameter_digest(*models):
    """Hash parameters independently of inference cache/position buffer growth."""
    import hashlib
    import torch

    digest = hashlib.sha256()
    for index, model in enumerate(models):
        for name, value in model.named_parameters():
            digest.update(str((index, name, tuple(value.shape), str(value.dtype))).encode())
            digest.update(
                value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
    return "sha256:" + digest.hexdigest()


def neural_measurements(recipe, output, device, repetitions, shapes=None):
    """Calibrate fixed states, then time restored training and frozen generation."""
    from copy import deepcopy
    import random
    import shutil
    import numpy as np
    import torch
    from oeis_learn.fresh.bigint_worker import models, _batch_loss
    from oeis_learn.decoder.bigint_generation import (
        GenerationRequest,
        sample_batch,
        GENERATION_BATCH_SIZE,
        GENERATION_POLICY,
    )
    from oeis_learn.fresh.conditioning_controls import condition_memory
    from oeis_learn.synthesis.contracts import identity
    from oeis_learn.synthesis.state import ExperimentContract, BudgetLedger
    from oeis_learn.fresh.bigint_state import CheckpointStore

    fixtures = neural_fixtures(shapes)
    records, generation_records, frozen_states = [], [], []
    contract = ExperimentContract(
        **{
            k: identity(["non-scoring-fixed-state/v2", k, recipe["identity"]])
            for k in (
                "data",
                "split",
                "model",
                "inputs",
                "decoder",
                "solver",
                "runtime",
                "resources",
                "policy",
            )
        },
        track="generic_only",
        foundation=None,
        seed=recipe["model"]["seed"],
        budget=7200,
    )
    store = CheckpointStore(Path(output) / "measurement-checkpoints", contract)
    ledger = BudgetLedger(7200)

    def progress():
        write(
            Path(output) / "neural-partial.json",
            dict(
                status="PARTIAL",
                qualified=False,
                measurement_protocol="fixed-state-batched/v2",
                frozen_states=frozen_states,
                samples=records,
                generation_samples=generation_records,
            ),
        )

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    def elapsed(function):
        sync()
        start = time.monotonic()
        value = function()
        sync()
        return time.monotonic() - start, value

    def key(result):
        return {k: result[k] for k in ("tokens", "complete", "source", "failure")}

    try:
        # These calibration counts are protocol constants, never timing repetitions.
        # Probe both previously observed short- and long-output synthetic states.
        for calibration in (3, 5):
            encoder, decoder, counts = models(recipe, device)
            optimizer = torch.optim.AdamW([*encoder.parameters(), *decoder.parameters()], lr=1e-4)
            for row in fixtures:
                encoder.train()
                decoder.train()
                for _ in range(calibration):
                    _batch_loss(encoder, decoder, [row] * 8, device, optimizer=optimizer)
            encoder.eval()
            decoder.eval()
            frozen = parameter_digest(encoder, decoder)
            frozen_states.append(
                dict(
                    calibration=calibration,
                    updates=len(fixtures) * calibration,
                    parameters_sha256=frozen,
                )
            )
            progress()
            snapshot = deepcopy(
                dict(
                    weights=dict(encoder=encoder.state_dict(), decoder=decoder.state_dict()),
                    optimizer=optimizer.state_dict(),
                    rng=dict(
                        python=random.getstate(),
                        numpy=np.random.get_state(),
                        torch=torch.get_rng_state(),
                        cuda=torch.cuda.get_rng_state_all() if device == "cuda" else None,
                    ),
                    data_order=dict(cursor=0, permutation=list(range(8))),
                    selection={},
                    curriculum={},
                    replay={},
                )
            )

            def restore():
                encoder.load_state_dict(snapshot["weights"]["encoder"])
                decoder.load_state_dict(snapshot["weights"]["decoder"])
                optimizer.load_state_dict(deepcopy(snapshot["optimizer"]))
                random.setstate(snapshot["rng"]["python"])
                np.random.set_state(snapshot["rng"]["numpy"])
                torch.set_rng_state(snapshot["rng"]["torch"])
                if device == "cuda":
                    torch.cuda.set_rng_state_all(snapshot["rng"]["cuda"])
                encoder.eval()
                decoder.eval()

            for row in fixtures:
                # Warm up training outside the timer, restoring the optimizer as well.
                encoder.train()
                decoder.train()
                _batch_loss(encoder, decoder, [row] * 8, device, optimizer=optimizer)
                restore()
                for _ in range(repetitions):
                    encoder.train()
                    decoder.train()
                    training, _ = elapsed(
                        lambda: _batch_loss(
                            encoder, decoder, [row] * 8, device, optimizer=optimizer
                        )
                    )
                    restore()
                    teacher, _ = elapsed(lambda: _batch_loss(encoder, decoder, [row] * 8, device))

                    def save_checkpoint():
                        value = store.save(snapshot, ledger)
                        store.prune()
                        return value

                    checkpoint, _ = elapsed(save_checkpoint)
                    records.append(
                        dict(
                            fixture_role=row["fixture_role"],
                            calibration=calibration,
                            parameters_sha256=frozen,
                            input_bits=max(abs(int(v)).bit_length() for v in row["values"]),
                            difference_bits=max(
                                abs(int(b) - int(a)).bit_length()
                                for a, b in zip(row["values"], row["values"][1:])
                            ),
                            target_tokens=len(row["tokens"]),
                            training=training,
                            teacher_forcing=teacher / 8,
                            checkpoint=checkpoint,
                        )
                    )
                progress()
            if parameter_digest(encoder, decoder) != frozen:
                raise RuntimeError("restored training/teacher forcing changed frozen parameters")

            # Cost only shape-matched non-scoring inputs. Without a manifest these
            # are capacity observations and can never admit a recipe.
            probes = [r for r in fixtures if r["fixture_role"] == "preflight"] or fixtures
            for row in probes:
                # Different visible prompts, with the same audited maximum width.
                values = [int(v) for v in row["values"]]
                visible = [values[i:] + values[:i] for i in range(GENERATION_BATCH_SIZE)]
                with torch.no_grad():
                    memory = encoder.forward_from_sequences(visible, device=device)
                padding = torch.zeros(memory.shape[:2], device=memory.device, dtype=torch.bool)
                for mode in ("correct", "shuffled", "absent"):
                    controlled, _ = condition_memory(
                        memory,
                        padding,
                        mode,
                        order=list(range(1, GENERATION_BATCH_SIZE)) + [0]
                        if mode == "shuffled"
                        else None,
                    )
                    for temperature in (0, 0.8):
                        print(
                            "MEASURING frozen generation",
                            calibration,
                            mode,
                            temperature,
                            flush=True,
                        )
                        seeds = [
                            recipe["model"]["seed"]
                            + int(identity(["cost", i])[7:14], 16)
                            + int(temperature > 0)
                            for i in range(GENERATION_BATCH_SIZE)
                        ]

                        def generate(batch_indices):
                            # Include production encoder work in measured end-to-end cost.
                            with torch.no_grad():
                                fresh = encoder.forward_from_sequences(visible, device=device)
                                fresh, _ = condition_memory(
                                    fresh,
                                    padding,
                                    mode,
                                    order=list(range(1, GENERATION_BATCH_SIZE)) + [0]
                                    if mode == "shuffled"
                                    else None,
                                )
                            deadline = time.monotonic() + 60
                            return sample_batch(
                                decoder,
                                [
                                    GenerationRequest(
                                        fresh[i : i + 1], seeds[i], temperature, deadline
                                    )
                                    for i in batch_indices
                                ],
                            )

                        # Single-row references also conservatively measure the tail.
                        references, tail_times = [], []
                        for i in range(GENERATION_BATCH_SIZE):
                            seconds, value = elapsed(lambda i=i: generate([i]))
                            references.append(key(value[0]))
                            tail_times.append(seconds)
                        _, warm = elapsed(lambda: generate(range(GENERATION_BATCH_SIZE)))
                        parity = [key(r) for r in warm] == references
                        samples = []
                        for _ in range(repetitions):
                            seconds, trajectories = elapsed(
                                lambda: generate(range(GENERATION_BATCH_SIZE))
                            )
                            equal = [key(r) for r in trajectories] == references
                            parity &= equal
                            samples.append(
                                dict(
                                    seconds=seconds,
                                    generated_tokens=[len(r["tokens"]) for r in trajectories],
                                    complete=[r["complete"] for r in trajectories],
                                    failures=[r["failure"] for r in trajectories],
                                    outputs_sha256=identity([key(r) for r in trajectories]),
                                    parity=equal,
                                )
                            )
                        unchanged = parameter_digest(encoder, decoder) == frozen
                        if not unchanged:
                            raise RuntimeError("generation changed frozen parameters")
                        generation_records.append(
                            dict(
                                fixture_role=row["fixture_role"],
                                calibration=calibration,
                                parameters_sha256=frozen,
                                mode=mode,
                                temperature=temperature,
                                seeds=seeds,
                                batch_size=GENERATION_BATCH_SIZE,
                                visible_sha256=identity(visible),
                                memory_sha256=parameter_tensor_digest(controlled),
                                samples=samples,
                                tail_seconds=tail_times,
                                parity=parity,
                                weights_unchanged=unchanged,
                                deadline_complete=all(
                                    r["failure"] != "deadline" for r in trajectories
                                ),
                            )
                        )
                        progress()
                        print(
                            "MEASURED frozen generation", calibration, mode, temperature, flush=True
                        )
    finally:
        shutil.rmtree(store.root, ignore_errors=True)
    measured = [r for r in records if r["fixture_role"] == "preflight"] or records
    generation_summary = summarize([t for r in generation_records for t in r["tail_seconds"]])
    batch_summary = summarize([s["seconds"] for r in generation_records for s in r["samples"]])
    # Pooled quantiles must not hide a slower control, seed group or model state.
    generation_summary["p95_seconds"] = max(
        summarize(r["tail_seconds"])["p95_seconds"] for r in generation_records
    )
    batch_summary["p95_seconds"] = max(
        summarize([s["seconds"] for s in r["samples"]])["p95_seconds"] for r in generation_records
    )
    return dict(
        measurement_protocol="fixed-state-batched/v2",
        generation_policy=GENERATION_POLICY,
        generation_batch_size=GENERATION_BATCH_SIZE,
        frozen_states=frozen_states,
        shape_sha256=identity(shapes) if shapes is not None else None,
        device=device,
        hip=torch.version.hip,
        parameters=counts,
        fixture_sha256=identity(
            dict(fixtures=fixtures, calibrations=[3, 5], policy=GENERATION_POLICY)
        ),
        scoring_data=False,
        samples=records,
        generation_samples=generation_records,
        generation_parity=all(r["parity"] and r["deadline_complete"] for r in generation_records),
        summaries={
            **{
                k: summarize([r[k] for r in measured])
                for k in ("training", "teacher_forcing", "checkpoint")
            },
            "generation": generation_summary,
            "generation_batch": batch_summary,
        },
    )


def parameter_tensor_digest(tensor):
    import hashlib
    import torch

    return (
        "sha256:"
        + hashlib.sha256(
            tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    )


def admit_cost_recipe(report, recipe, shapes, output, *, scope="transfer-screen"):
    """Bind observed costs to the actual image/profile and complete prepared shape audit."""
    from copy import deepcopy
    from oeis_learn.synthesis.contracts import identity
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.fresh.cost_policy import screen_cost, schedule_identity
    import yaml

    counts = {"training": 512, "parameter_development": 128, "structural_development": 128}
    neural = report.get("neural", {})
    if neural.get("measurement_protocol") != "fixed-state-batched/v2" or not neural.get(
        "frozen_states"
    ):
        raise ValueError("fixed-state model calibration and parameter identities required")
    if neural.get("generation_parity") is not True:
        raise ValueError(
            "mixed-prompt greedy/stochastic batching parity and deadline checks required"
        )
    if (
        neural.get("generation_batch_size") != 8
        or report["costs"].get("generation_batch_size") != 8
        or "generation_batch_p95" not in report["costs"]
    ):
        raise ValueError("measured full batch8 latency and conservative singleton tail required")
    forecast = screen_cost(
        **report["costs"],
        scope=scope,
        updates=recipe["learning"]["max_updates"],
        cadence=recipe["learning"]["validation_every"],
        budget=recipe["budgets"]["training_seconds"],
    )
    if not forecast["within_budget"]:
        raise ValueError(
            f"measured {scope} cost {forecast['projected_seconds']:.3f}s exceeds "
            f"80% budget {forecast['budget'] * 0.8:.3f}s"
        )
    if (
        report.get("status") != "PASS"
        or not report.get("neural", {}).get("hip")
        or not report["neural"]["device"].startswith("cuda")
        or not forecast["within_budget"]
        or report.get("fitting", {}).get("complete") is not True
        or report["neural"].get("shape_sha256") != identity(shapes)
        or shapes.get("counts") != counts
        or shapes.get("tiny_count") != 32
        or shapes.get("native_admitted") is not True
        or any(not shapes.get("stages", {}).get(s) for s in ("A", "B", "C", "D"))
        or shapes.get("max_input_bits", 4097) > 4096
        or shapes.get("max_tokens", 1025) > 1024
        or not shapes.get("prepared_sha256")
        or not shapes.get("split_sha256")
    ):
        raise ValueError(
            "HIP, complete shape-audited512/128/128+diverse32 and passing80% cost screen required"
        )
    value = deepcopy(recipe)
    value.pop("identity", None)
    value["learning"]["cost_preflight"] = dict(
        image=report["image"],
        profile_sha256=identity(
            {k: recipe[k] for k in ("execution", "inputs", "model", "fitting", "search")}
        ),
        scoring_data=False,
        costs=report["costs"],
        fixture_sha256=identity(
            dict(
                fixtures=report["neural"]["fixture_sha256"],
                measurement_protocol=report["neural"]["measurement_protocol"],
                frozen_states=report["neural"]["frozen_states"],
                generation_policy=report["neural"].get("generation_policy"),
                shapes=shapes,
                runtime_cases=[r["case"] for r in report["cases"]],
            )
        ),
        scope=scope,
        source_sha256=report["source_sha256"],
        shape_sha256=identity(shapes),
        schedule_sha256=schedule_identity(recipe, scope),
    )
    value = resolve(value)
    path = Path(output) / f"measured-{scope}-recipe.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return dict(
        path=path.name,
        recipe=value,
        scope=scope,
        shape_sha256=identity(shapes),
        qualification="bounded cost estimate, not transfer evidence",
    )


def run(
    output,
    *,
    repetitions=3,
    batches=(1, 20, 100),
    neural=False,
    device="cpu",
    config=None,
    shapes=None,
    cost_scope="diverse32",
    profile_runtime=False,
    compare_host_lookups=False,
    profile_input_load=False,
    compare_memory_growth=False,
):
    from oeis_learn.synthesis.contracts import identity
    from oeis_learn.experiments.source_snapshot import collect_snapshot

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "benchmark.json").exists():
        raise ValueError("new output directory required; preserve previous measurements")
    report = dict(
        version="bigint-runtime-profile/v1" if profile_runtime else "bigint-performance/v1",
        compare_host_lookups=compare_host_lookups,
        profile_input_load=profile_input_load,
        **({"compare_memory_growth": True} if compare_memory_growth else {}),
        status="FAIL",
        qualified=False,
        python=sys.version,
        source_sha256=identity(collect_snapshot(ROOT)),
        image=os.environ.get("OEIS_QUALIFICATION_IMAGE"),
        cases=[],
        semantics="production bigint-resources/v1; no wider-input profile",
        accounting="prepare includes parse/lower/candidate compilation; execute includes fresh stores, loading and serialization; runtime compilation separate; warm reuses runtime module only",
    )
    try:
        if compare_memory_growth and (
            not profile_runtime or not profile_input_load or compare_host_lookups
        ):
            raise ValueError("memory growth comparison requires isolated CPU input profiling")
        if (compare_host_lookups or profile_input_load) and not profile_runtime:
            raise ValueError("host lookup/input-load details require runtime profiling")
        if profile_runtime and (
            neural or device != "cpu" or config is not None or shapes is not None
        ):
            raise ValueError(
                "runtime profiling is CPU-only and excludes neural/data/configuration work"
            )
        from oeis_learn.fresh.bigint_config import packaged_artifacts
        from oeis_learn.fresh.common import file_hash

        if neural and device == "cuda":
            import torch

            if not torch.version.hip or not torch.cuda.is_available():
                raise RuntimeError("actual HIP required; CPU fallback prohibited")

        report["artifacts"] = packaged_artifacts()
        report["measurement_binary_sha256"] = file_hash(
            os.environ.get("OEIS_BENCHMARK_RUNTIME", "/opt/oeis-build-info/benchmark-runtime")
        )
        if not 2 <= repetitions <= 5 or not batches or any(b not in (1, 20, 100) for b in batches):
            raise ValueError("2..5 repetitions and batches1/20/100 required")
        for case in cases():
            for batch in batches:
                report["active_measurement"] = dict(case=case["name"], batch=batch)
                if compare_memory_growth:
                    row = memory_growth_measurements(
                        case,
                        batch,
                        repetitions,
                        order="zero-first" if len(report["cases"]) % 2 == 0 else "reserved-first",
                        failure_output=output / "failed-native.log",
                        progress_output=output / "growth-active.json",
                    )
                elif profile_runtime:
                    row = runtime_measurements(
                        case,
                        batch,
                        repetitions,
                        profile=True,
                        **({"compare_host_lookups": True} if compare_host_lookups else {}),
                        **(
                            {
                                "profile_input_load": True,
                                "failure_output": output / "failed-native.log",
                            }
                            if profile_input_load
                            else {}
                        ),
                    )
                else:
                    row = runtime_measurements(case, batch, repetitions)
                report["cases"].append(row)
                write(output / "partial.json", report)
                if profile_runtime and (
                    not row["profiling"]["paired_results_match"]
                    or (compare_host_lookups and not row["host_lookups"]["all_results_match"])
                    or (compare_memory_growth and not row["memory_growth"]["all_results_match"])
                ):
                    raise RuntimeError(
                        "profiled/control/baseline semantic, fuel or coverage disagreement; partial.json retains all results"
                    )
                print("MEASURED", case["name"], "batch", batch, flush=True)
        if not profile_runtime:
            report["active_measurement"] = dict(phase="fitting")
            report["fitting"] = fitting_measurements(repetitions)
        import resource

        report["resource_metrics"] = dict(
            peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
            accounting="Linux RUSAGE_CHILDREN maximum among reaped children, not aggregate resident bytes; per-evaluation fuel is retained",
        )
        if neural:
            report["active_measurement"] = dict(phase="neural")
            from oeis_learn.fresh.bigint_config import load, packaged_artifacts

            recipe = load(
                config
                or ROOT
                / "configs/foundation"
                / (
                    "bigint_smoke_phase5.yaml"
                    if cost_scope == "diverse32"
                    else "bigint_literal_screen.yaml"
                ),
                artifacts=packaged_artifacts(),
            )
            report["neural"] = neural_measurements(
                recipe, output, device, repetitions, shapes=shapes
            )
            from oeis_learn.fresh.cost_policy import screen_cost

            summaries = report["neural"]["summaries"]
            costs = {k + "_p95": summaries[k]["p95_seconds"] for k in summaries}
            costs["generation_batch_size"] = report["neural"]["generation_batch_size"]
            costs.update(
                fit_p95=report["fitting"]["native"]["p95_seconds"],
                reference_p95=report["fitting"]["reference"]["p95_seconds"],
                hidden_scoring_p95=max(
                    c["runtime_compile_seconds"]
                    + c["warm"]["total_seconds"]["p95_seconds"]
                    + c["reference_summary"]["p95_seconds"]
                    for c in report["cases"]
                    if c["batch"] == 100
                ),
            )
            forecast = screen_cost(
                **costs,
                scope=cost_scope,
                updates=recipe["learning"]["max_updates"],
                cadence=recipe["learning"]["validation_every"],
                budget=recipe["budgets"]["training_seconds"],
            )
            report["cost_projection"] = forecast
            report["costs"] = costs
            report["preflight_eligible"] = False
            report["preflight_reason"] = (
                "tensor cost screen only; matched complete dataset size/complexity and hardware binding must be reviewed before recipe admission"
            )
        if profile_runtime:
            report["profiling_overview"] = dict(
                rows=len(report["cases"]),
                paired_results_match=all(
                    r["profiling"]["paired_results_match"] for r in report["cases"]
                ),
                phase_ranking=sorted(
                    [
                        dict(
                            phase=name,
                            summed_warm_median_seconds=sum(
                                r["profiling"]["phases"][name]["seconds"]["median_seconds"]
                                for r in report["cases"]
                            ),
                        )
                        for name in PROFILE_PHASES
                    ],
                    key=lambda row: row["summed_warm_median_seconds"],
                    reverse=True,
                ),
                interpretation="Descriptive ranking across the fixed corpus; inspect per-case overhead, order and residual before attributing a bottleneck.",
            )
        report["slowdown_investigations"] = [
            dict(case=r["case"], batch=r["batch"], ratio=r["retained"]["small_value_slowdown"])
            for r in report["cases"]
            if r["retained"].get("investigation_required")
        ]
        report["status"] = "PASS"
        report.pop("active_measurement", None)
        if neural and shapes is not None:
            other = "transfer-screen" if cost_scope == "diverse32" else "diverse32"
            other_path = (
                ROOT
                / "configs/foundation"
                / (
                    "bigint_literal_screen.yaml"
                    if other == "transfer-screen"
                    else "bigint_smoke_phase5.yaml"
                )
            )
            recipes = {cost_scope: recipe, other: load(other_path, artifacts=packaged_artifacts())}
            qualify_scopes(report, recipes, shapes, output, primary=cost_scope)
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["report_sha256"] = identity(report)
        write(output / "benchmark.json", report)
    return report


def qualify_scopes(report, recipes, shapes, output, *, primary):
    """Record each registered workload independently; one pass grants no other scope."""
    from oeis_learn.fresh.cost_policy import screen_cost, schedule_identity
    from oeis_learn.synthesis.contracts import identity

    receipts = {}
    profile = lambda recipe: identity(
        {k: recipe[k] for k in ("execution", "inputs", "model", "fitting", "search")}
    )
    for scope, recipe in recipes.items():
        forecast = screen_cost(
            **report["costs"],
            scope=scope,
            updates=recipe["learning"]["max_updates"],
            cadence=recipe["learning"]["validation_every"],
            budget=recipe["budgets"]["training_seconds"],
        )
        receipt = dict(
            version="scoped-preflight/v1",
            scope=scope,
            status="BLOCKED",
            source_sha256=report["source_sha256"],
            image=report["image"],
            profile_sha256=profile(recipe),
            shape_sha256=identity(shapes),
            schedule_sha256=schedule_identity(recipe, scope),
            recipe_sha256=identity(recipe),
            projection=forecast,
            scoring_data=False,
        )
        try:
            if profile(recipe) != profile(recipes[primary]):
                raise ValueError("secondary scope profile differs from the measured model")
            receipt["measured_recipe"] = admit_cost_recipe(
                report, recipe, shapes, output, scope=scope
            )
            receipt["status"] = "PASS"
        except ValueError as exc:
            receipt["reason"] = str(exc)
        receipt["receipt_sha256"] = identity(receipt)
        write(Path(output) / f"cost-{scope}.json", receipt)
        receipts[scope] = receipt
    report["cost_scope"] = primary
    report["cost_receipts"] = receipts
    report["cost_projection"] = receipts[primary]["projection"]
    report["preflight_eligible"] = receipts[primary]["status"] == "PASS"
    report["preflight_reason"] = receipts[primary].get(
        "reason", "matching scoped cost and complete shape audit passed"
    )
    if report["preflight_eligible"]:
        report["measured_recipe"] = receipts[primary]["measured_recipe"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--batches", default="1,20,100")
    parser.add_argument("--neural", action="store_true")
    parser.add_argument(
        "--runtime-profile",
        action="store_true",
        help="paired CPU runtime phases and observer overhead; no fitting/neural work",
    )
    parser.add_argument(
        "--compare-host-lookups",
        action="store_true",
        help="add an ordinary pre-cache lookup baseline to the CPU runtime profile",
    )
    parser.add_argument(
        "--profile-input-load",
        action="store_true",
        help="nested CPU transfer timings and partial byte/value counters; requires --runtime-profile",
    )
    parser.add_argument(
        "--compare-memory-growth",
        action="store_true",
        help="qualification-only zero/64KiB separate-process comparison; requires input runtime profiling",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--config")
    parser.add_argument("--shape-manifest")
    parser.add_argument(
        "--cost-scope", choices=("diverse32", "transfer-screen"), default="diverse32"
    )
    args = parser.parse_args()
    try:
        run(
            args.output,
            repetitions=args.repetitions,
            batches=tuple(map(int, args.batches.split(","))),
            neural=args.neural,
            device=args.device,
            config=args.config,
            shapes=json.loads(Path(args.shape_manifest).read_text())
            if args.shape_manifest
            else None,
            cost_scope=args.cost_scope,
            profile_runtime=args.runtime_profile,
            compare_host_lookups=args.compare_host_lookups,
            profile_input_load=args.profile_input_load,
            compare_memory_growth=args.compare_memory_growth,
        )
    except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    raise SystemExit(main())
