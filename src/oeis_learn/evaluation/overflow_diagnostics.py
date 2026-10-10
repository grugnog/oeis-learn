"""Isolated wider logical-reference diagnostics and benchmark-only unchecked modules.

Only explicit logical pure programs are supported. Origin labels are descriptive,
not verified archive admission. No private truth, model access or admission output.
"""

from pathlib import Path
import re
import time
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from oeis_learn.experiments.models import check_digest
from oeis_learn.sandbox.reference import LogicalReference
from oeis_learn.sandbox.wat_ast import ExecutionFailure

STRATA = {
    "random_generic": "generic_sample",
    "model_samples": "model_sample",
    "verified_solutions": "verified_solution",
}


class WiderReference(LogicalReference):
    def __init__(self, source):
        super().__init__(source)
        self.maximum_bits, self.violations = 0, 0
        self.max_steps = 50000000
        self.deadline = time.monotonic() + 2

    def checked(self, value):
        bits = abs(value).bit_length()
        self.maximum_bits = max(self.maximum_bits, bits)
        if bits > 4096:
            raise ExecutionFailure("unknown", "wider_reference_4096_bit_cap")
        if not -(1 << 255) <= value < (1 << 255):
            self.violations += 1
        return value


def _trace(source):
    ref = WiderReference(source)
    if ref.program.profile != "pure":
        return dict(
            outcome="unsupported",
            reason="wider trace requires explicit pure logical profile",
            violations=0,
        )
    outputs, outcome, reason = [], None, None
    try:
        for index in range(100):
            outputs.append(str(ref.compute(index)))
    except ExecutionFailure as failure:
        outcome, reason = failure.outcome, failure.reason
    return dict(
        outcome=outcome,
        reason=reason,
        outputs=outputs,
        steps=ref.steps,
        violations=ref.violations,
        maximum_intermediate_bits=ref.maximum_bits,
        final_outputs_in_range=len(outputs) == 100
        and all(-(1 << 255) <= int(v) < (1 << 255) for v in outputs),
        native_limb_wraps=None,
        native_limb_wraps_reason="logical integer profile; native word events unavailable",
    )


def _benchmark(source, expected):
    """Never return compiled bytes. Unchecked lowering stays inside this worker."""
    import wasmtime
    from oeis_learn.sandbox.lowering import lower_logical
    from oeis_learn.sandbox.logical_ir import parse_logical
    from oeis_learn.sandbox.pipeline import decode_limbs
    from oeis_learn.tracking.foundation_metrics import fuel_summary

    checked = lower_logical(parse_logical(source))
    # Suppress only the numeric guard branches in trusted generated helpers;
    # divide-by-zero, array bounds and other traps remain active.
    unchecked = re.sub(
        r"if\s+i32.const 1\s+global.set \$__oeis_numeric_limit\s+unreachable\s+end", "drop", checked
    )
    config = wasmtime.Config()
    config.consume_fuel = True
    config.parallel_compilation = False
    config.memory_reservation = 0
    config.memory_guard_size = 0
    config.memory_reservation_for_growth = 0
    engine = wasmtime.Engine(config)
    modules = {
        name: wasmtime.Module(engine, text)
        for name, text in [("checked", checked), ("unchecked", unchecked)]
    }
    times, fuels = {k: [] for k in modules}, {k: [] for k in modules}
    aggregate = 0
    try:
        for repeat in range(40):
            for variant in (
                ("checked", "unchecked") if repeat % 2 == 0 else ("unchecked", "checked")
            ):
                elapsed = 0
                for index in range(100):
                    if aggregate >= 50000000:
                        raise ExecutionFailure("unknown", "benchmark_aggregate_fuel")
                    with wasmtime.Store(engine) as store:
                        fuel = min(1000000, 50000000 - aggregate)
                        store.set_fuel(fuel)
                        instance = wasmtime.Instance(store, modules[variant], [])
                        start = time.monotonic_ns()
                        value = decode_limbs(instance.exports(store)["compute"](store, index))
                        elapsed += time.monotonic_ns() - start
                        used = fuel - store.get_fuel()
                        aggregate += used
                        if str(value) != expected[index]:
                            raise ExecutionFailure(
                                "runtime_failure", "checked_unchecked_disagreement"
                            )
                        if repeat >= 10:
                            fuels[variant].append(used)
                if repeat >= 10:
                    times[variant].append(elapsed)
    finally:
        for module in modules.values():
            module.close()
        engine.close()
    return dict(
        artifact_purpose="benchmark_only_unchecked",
        admission_eligible=False,
        warmups=10,
        repeats=30,
        outputs_equal=True,
        timings_ns=times,
        fuel={k: fuel_summary(v, limit=1000000) for k, v in fuels.items()},
        aggregate_fuel_including_warmups=aggregate,
        checked_sha256=digest(checked),
        unchecked_sha256=digest(unchecked),
    )


def diagnose_overflow(strata, output):
    if not isinstance(strata, dict) or set(strata) != set(STRATA):
        raise ValueError("three separate frozen strata required")
    from oeis_learn.sandbox.logical_ir import parse_logical
    from oeis_learn.sandbox.contained import run_contained
    from oeis_learn.evaluation.experiment_runner import implementation_identity

    for name, rows in strata.items():
        if not isinstance(rows, list) or len(rows) > 256:
            raise ValueError("stratum cap is 256")
        seen = set()
        for row in rows:
            if (
                set(row) != {"source", "origin", "provenance_sha256"}
                or row["origin"] != STRATA[name]
            ):
                raise ValueError("invalid stratum origin/fields")
            check_digest(row["provenance_sha256"], "provenance")
            identity = parse_logical(row["source"]).identity
            if identity in seen:
                raise ValueError("duplicate stratum source")
            seen.add(identity)
    root = Path(output)
    from oeis_learn.evaluation.foundation_cohort import write_json

    write_json(root, "strata.json", strata)
    report = dict(
        version="overflow-diagnostic/v1",
        qualified=False,
        admission_eligible=False,
        source_sha256=digest(strata),
        code_sha256=implementation_identity(),
        indices=list(range(100)),
        strata={},
        rescued_candidates=0,
        added_development_groups=None,
        prefix_matching_rescues=None,
        development_matching_rescues=None,
        missing_evidence=[
            "verified archive origin linkage",
            "sealed development scoring",
            "baseline body/native word tracing",
        ],
        proof_status="not_claimed",
    )
    for name, rows in strata.items():
        records = []
        for row in rows:
            status, trace = run_contained(_trace, (row["source"],), timeout_ms=2000)
            if status != "ok":
                trace = dict(outcome="unknown", reason=trace, violations=0, outputs=[])
            benchmark = None
            if (
                trace.get("outcome") is None
                and trace.get("violations") == 0
                and trace.get("final_outputs_in_range")
            ):
                state, result = run_contained(
                    _benchmark, (row["source"], trace["outputs"]), timeout_ms=10000, proof=True
                )
                benchmark = result if state == "ok" else dict(outcome="unknown", reason=result)
            if (
                trace.get("outcome") is None
                and trace.get("violations", 0) > 0
                and trace.get("final_outputs_in_range")
            ):
                report["rescued_candidates"] += 1
            records.append(
                dict(
                    source_sha256=digest(row["source"]),
                    provenance_sha256=row["provenance_sha256"],
                    wide=trace,
                    benchmark=benchmark,
                )
            )
        report["strata"][name] = dict(
            availability="available" if rows else "unavailable",
            population=len(rows),
            records=records,
            shortfall=256 - len(rows),
        )
    report["report_sha256"] = digest(report)
    write_json(root, "overflow.json", report)
    return report
