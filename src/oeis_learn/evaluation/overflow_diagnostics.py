"""Isolated wider logical-reference diagnostics and benchmark-only unchecked modules.

Only explicit logical pure programs are supported. Origin labels are descriptive,
not verified archive admission. No private truth, model access or admission output.
"""

from pathlib import Path
import re
import time
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from oeis_learn.experiments.models import check_digest
from oeis_learn.sandbox.reference import LogicalReference, Interpreter
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

    def expr(self, node):
        value = super().expr(node)
        self.maximum_bits = max(self.maximum_bits, abs(value).bit_length())
        return value


def _trace(source):
    if not source.lstrip().startswith("("):
        return _trace_body(source)
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

    if source.lstrip().startswith("("):
        checked = lower_logical(parse_logical(source))
    else:
        from oeis_learn.sandbox.lowering import lower_body
        from oeis_learn.sandbox.wat_ast import parse

        checked = lower_body(parse(source))
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


def diagnose_overflow(
    strata, output, *, archive=None, snapshot=None, development=None, provenance_packets=None
):
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
            if row["source"].lstrip().startswith("("):
                identity = parse_logical(row["source"]).identity
            else:
                from oeis_learn.sandbox.wat_ast import parse

                identity = digest(parse(row["source"]).canonical_source)
            if identity in seen:
                raise ValueError("duplicate stratum source")
            seen.add(identity)
    root = Path(output)
    from oeis_learn.evaluation.foundation_cohort import write_json

    provenance = (
        verify_origins(strata, archive, snapshot, provenance_packets=provenance_packets)
        if archive is not None or provenance_packets is not None
        else None
    )
    write_json(root, "strata.json", strata)
    if provenance_packets is not None:
        write_json(root, "provenance.json", provenance_packets)
    write_json(root, "seal.json", dict(strata_sha256=digest(strata), provenance=provenance))
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
    rescued_sources = set()
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
                rescued_sources.add(digest(row["source"]))
            records.append(
                dict(
                    source_sha256=digest(row["source"]),
                    provenance_sha256=row["provenance_sha256"],
                    source=row["source"],
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
    report["rescued_candidates"] = len(rescued_sources)
    if provenance is not None:
        report["provenance"] = provenance
        report["missing_evidence"].remove("verified archive origin linkage")
    report["missing_evidence"].remove("baseline body/native word tracing")
    if development is not None:
        report.update(score_development(report, development, root))
        report["missing_evidence"].remove("sealed development scoring")
    report["report_sha256"] = digest(report)
    write_json(root, "overflow.json", report)
    return report


class _WideLimb(int):
    """Shadow metadata survives only unmodified local moves, never native arithmetic."""

    def __new__(cls, word, value, limb, group):
        obj = int.__new__(cls, word)
        obj.wide_value, obj.limb, obj.group = value, limb, group
        return obj


class WiderBodyInterpreter(Interpreter):
    """Independent body AST execution with wide macro shadows and native word semantics."""

    def __init__(self, steps, deadline):
        super().__init__(steps, deadline)
        self.maximum_bits = self.violations = self.native_wraps = self.group = 0
        self.in_native = False

    def signed(self, value, bits):
        if self.in_native and not -(1 << (bits - 1)) <= value < (1 << (bits - 1)):
            self.native_wraps += 1
        return Interpreter.signed(value, bits)

    def native(self, op):
        self.in_native = op.split(".")[1] in ("add", "sub", "mul", "shl", "wrap_i64")
        try:
            super().native(op)
        finally:
            self.in_native = False

    def wide(self):
        words = self.stack[-4:]
        if (
            len(words) == 4
            and all(isinstance(w, _WideLimb) for w in words)
            and len({w.group for w in words}) == 1
            and [w.limb for w in words] == list(range(4))
        ):
            del self.stack[-4:]
            return words[0].wide_value
        return super().wide()

    def push_wide(self, value):
        self.maximum_bits = max(self.maximum_bits, abs(value).bit_length())
        if self.maximum_bits > 4096:
            raise ExecutionFailure("unknown", "wider_reference_4096_bit_cap")
        self.violations += int(not -(1 << 255) <= value < (1 << 255))
        self.group += 1
        for i in range(4):
            self.stack.append(
                _WideLimb(Interpreter.signed(value >> (64 * i), 64), value, i, self.group)
            )


def _trace_body(source):
    from oeis_learn.sandbox.wat_ast import parse

    program = parse(source)
    deadline = time.monotonic_ns() + 2_000_000_000
    outputs, steps, violations, wraps, maximum = [], 0, 0, 0, 0
    outcome = reason = None
    for index in range(100):
        ref = WiderBodyInterpreter(50000000 - steps, deadline)
        result = ref.run(program, index)
        steps += result.steps
        violations += ref.violations
        wraps += ref.native_wraps
        maximum = max(maximum, ref.maximum_bits)
        if result.outcome:
            outcome, reason = result.outcome, result.reason
            if outcome == "execution_limit":
                outcome = "unknown"
            break
        outputs.append(str(result.value))
    return dict(
        outcome=outcome,
        reason=reason,
        outputs=outputs,
        steps=steps,
        violations=violations,
        maximum_intermediate_bits=maximum,
        final_outputs_in_range=len(outputs) == 100
        and all(-(1 << 255) <= int(v) < (1 << 255) for v in outputs),
        native_limb_wraps=wraps,
        native_limb_wraps_reason=None,
        profile="wat_body_decimal_v1",
        shadow_policy="unaltered four-limb moves only",
    )


def archive_strata(archive, snapshot=None):
    """Freeze genuine admitted source strata without importing unavailable solutions."""
    snapshot = archive.freeze() if snapshot is None else snapshot
    archive.training_view(snapshot)  # snapshot identity and eligibility replay
    result = {name: [] for name in STRATA}
    for key in snapshot["entries"]:
        value = archive.get(key)
        name = "random_generic" if value["entry"]["origin"] == "generic_sample" else "model_samples"
        if len(result[name]) < 256:
            result[name].append(
                dict(source=value["source"], origin=STRATA[name], provenance_sha256=key)
            )
        # Already verified is a distinct stratum; do not promote a mere label.
        if len(result["verified_solutions"]) < 256:
            result["verified_solutions"].append(
                dict(
                    source=value["source"],
                    origin=STRATA["verified_solutions"],
                    provenance_sha256=key,
                )
            )
    # Multiple archive entries can have identical source bytes.
    for name, rows in result.items():
        result[name] = list({r["source"]: r for r in rows}.values())
    return result


def verify_origins(strata, archive=None, snapshot=None, *, provenance_packets=None):
    from oeis_learn.discovery.program_archive import ProgramArchive
    from oeis_learn.data.structural_splits import StructuralHoldouts

    snapshot = archive.freeze() if archive is not None and snapshot is None else snapshot
    if archive is not None:
        archive.training_view(snapshot)
    packets = provenance_packets or {}
    verifier = archive or object.__new__(ProgramArchive)
    if archive is None:
        verifier.holdouts = StructuralHoldouts()
    for name, rows in strata.items():
        for row in rows:
            key = row["provenance_sha256"]
            if snapshot is not None and key in snapshot["entries"]:
                value = archive.get(key)
                if value["source"] != row["source"]:
                    raise ValueError("archive/source mismatch")
                origin = value["entry"]["origin"]
                if (name == "random_generic" and origin != "generic_sample") or (
                    name == "model_samples"
                    and origin not in ("generic_model", "training_prefix_hypothesis")
                ):
                    raise ValueError("archive origin does not match stratum")
                continue
            if name == "verified_solutions" or key not in packets or digest(packets[key]) != key:
                raise ValueError("source absent from frozen archive/provenance")
            packet = packets[key]
            if name == "random_generic":
                verifier._provenance("generic_sample", packet, row["source"], [], None)
            else:
                if set(packet) not in ({"rollout", "conditioning"}, {"search", "conditioning"}):
                    raise ValueError("raw model provenance/visible conditioning required")
                if len(packet["conditioning"]) != 20:
                    raise ValueError("exact visible prompt required")
                if "rollout" in packet:
                    r = packet["rollout"]
                    model = digest([r["behavior_encoder_sha256"], r["behavior_decoder_sha256"]])
                    body = {"rollout": r}
                else:
                    r = packet["search"]
                    model = digest([r["encoder_sha256"], r["decoder_sha256"]])
                    body = {"search": r}
                verifier._provenance(
                    "generic_model", body, row["source"], packet["conditioning"], model
                )
    return dict(
        snapshot_sha256=snapshot["root_sha256"] if snapshot else None,
        entries=list(snapshot["entries"]) if snapshot else [],
        provenance_sha256=digest(packets),
    )


def sampled_strata(sampler, *, model_samples=(), archive=None, snapshot=None, count=256):
    """Raw generic/model draws remain traceable even if strict admission overflows."""
    if type(count) is not int or not 0 <= count <= 256:
        raise ValueError("stratum cap256")
    strata = (
        archive_strata(archive, snapshot) if archive is not None else {name: [] for name in STRATA}
    )
    packets = {}
    raw = []
    for counter in range(count):
        draw = sampler.sample(counter)
        packet = {"config": sampler.config, "sample": draw}
        key = digest(packet)
        packets[key] = packet
        raw.append(
            dict(
                source=draw["canonical_source"],
                origin=STRATA["random_generic"],
                provenance_sha256=key,
            )
        )
    strata["random_generic"] = list({r["source"]: r for r in raw}.values())
    raw = []
    for source, packet in model_samples:
        if len(raw) >= 256:
            break
        key = digest(packet)
        packets[key] = packet
        raw.append(dict(source=source, origin=STRATA["model_samples"], provenance_sha256=key))
    if raw:
        strata["model_samples"] = list({r["source"]: r for r in raw}.values())
    verify_origins(strata, archive, snapshot, provenance_packets=packets)
    return strata, packets


def score_development(report, development, output):
    """Evaluator-only after source/trace sealing; only development truth is decoded."""
    from oeis_learn.evaluation.foundation_cohort import load_cohort, read_ref, write_json
    from oeis_learn.experiments.artifacts import load_json

    root = Path(development)
    cohort = load_cohort(root)
    seal = load_json((Path(output) / "seal.json").read_bytes())
    if seal["strata_sha256"] != report["source_sha256"]:
        raise ValueError("diagnostic candidates were not sealed")
    prefix_sources, continuation_sources = set(), set()
    rescued_groups, checked_groups = set(), set()
    records = [r for s in report["strata"].values() for r in s["records"]]
    # Deduplicate correlated implementations across strata for these counts.
    records = list({r["source_sha256"]: r for r in records}.values())
    for group in cohort["groups"]:
        if group["partition"] != "development":
            continue
        truth = read_ref(root, group["truth"])["values"]
        for row in records:
            trace = row["wide"]
            if trace.get("outcome") is not None or not trace.get("final_outputs_in_range"):
                continue
            values = trace["outputs"]
            rescued = trace["violations"] > 0
            if values[:20] == truth[:20] and rescued:
                prefix_sources.add(row["source_sha256"])
            if values == truth:
                if rescued:
                    continuation_sources.add(row["source_sha256"])
                    rescued_groups.add(group["group_id"])
                else:
                    checked_groups.add(group["group_id"])
    score = dict(
        cohort_sha256=cohort["cohort_id"],
        selection_split="development",
        candidate_seal_sha256=digest(seal),
        prefix_matching_rescues=len(prefix_sources),
        development_matching_rescues=len(continuation_sources),
        added_development_groups=len(rescued_groups - checked_groups),
        rescued_development_groups=sorted(rescued_groups),
        checked_development_groups=sorted(checked_groups),
        feedback_to_candidates=False,
    )
    write_json(Path(output), "development-score.json", score)
    return score
