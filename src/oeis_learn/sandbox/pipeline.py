"""Shared strict prepare/execute/verify service for the foundation runtime.

These synchronous primitives run *inside* the externally supervised WorkerPool.
Calling them directly is useful for diagnostics, not process-isolated admission.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import resource
import time

import wasmtime

from oeis_learn.experiments.artifacts import compute_canonical_digest
from oeis_learn.experiments.models import validate_artifact, parse_integer_text, check_digest
from oeis_learn.experiments.profiles import I256_MIN, I256_MAX, profile_digests
from oeis_learn.sandbox import reference
from oeis_learn.sandbox.lowering import lower_body
from oeis_learn.sandbox.preamble import load_preamble_wat
from oeis_learn.sandbox.wat_ast import ExecutionFailure, Program, parse


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class Limits:
    fuel_per_term: int = 1_000_000
    fuel_aggregate: int = 50_000_000
    reference_per_term: int = 1_000_000
    reference_aggregate: int = 50_000_000
    deadline_ns: int = 2_000_000_000

    def __post_init__(self):
        caps = (1_000_000, 50_000_000, 1_000_000, 50_000_000, 2_000_000_000)
        for (key, value), cap in zip(asdict(self).items(), caps):
            if type(value) is not int or not 0 <= value <= cap:
                raise ValueError(f"{key}: exact integer in 0..{cap} required")


@dataclass(frozen=True)
class Failure:
    outcome: str
    reason: str
    stage: str = "prepare"


@dataclass(frozen=True)
class PreparedCandidate:
    program: Program
    module: wasmtime.Module
    source_sha256: str
    lowered_sha256: str
    wrapper_sha256: str
    helper_sha256: str
    runtime_sha256: str
    body_token_count: int
    module_token_count: int
    cache_hit: bool


@dataclass
class ExecutionEvidence:
    source_sha256: str
    indices: list[int]
    outputs: list[str] = field(default_factory=list)
    reference_outputs: list[str] = field(default_factory=list)
    reference_terms: list[dict] = field(default_factory=list)
    fuel_terms: list[int] = field(default_factory=list)
    fuel: int | None = 0
    reference_steps: int | None = 0
    elapsed_ns: int = 0
    peak_rss_bytes: int | None = None
    outcome: str | None = None
    reason: str | None = None
    first_failure_index: int | None = None
    failure_stage: str | None = None
    identities: dict = field(default_factory=dict)
    reference_sha256: str | None = None
    cache_hit: bool = False

    def to_dict(self):
        return asdict(self)


def runtime_identity() -> dict:
    root = Path(__file__).resolve().parent
    implementation = {
        name: digest_bytes((root / name).read_bytes())
        for name in (
            "pipeline.py",
            "wat_ast.py",
            "reference.py",
            "lowering.py",
            "worker_pool.py",
            "preamble.py",
            "../decoder/program_codec.py",
            "../experiments/profiles.py",
        )
    }
    return {
        "backend": "python_wasmtime",
        "wasmtime": version("wasmtime"),
        "python": platform.python_version(),
        "architecture": platform.machine(),
        "os": platform.system(),
        "consume_fuel": True,
        "parallel_compilation": False,
        "wasm_threads": False,
        "memory_reservation": 0,
        "memory_guard_size": 0,
        "helper_sha256": digest_bytes(load_preamble_wat().encode()),
        "profiles": profile_digests(),
        "implementation": implementation,
    }


def decode_limbs(values) -> int:
    if (
        not isinstance(values, (list, tuple))
        or len(values) != 4
        or any(type(v) is not int or not -(1 << 63) <= v < (1 << 63) for v in values)
    ):
        raise ExecutionFailure("invalid_type", "exactly four signed i64 limbs required")
    bits = sum((v & ((1 << 64) - 1)) << (64 * i) for i, v in enumerate(values))
    return bits - (1 << 256) if bits & (1 << 255) else bits


class Runtime:
    """One engine per worker; bounded LRU of self-produced compiled bytes.

    Serialized compiled modules avoid retaining unaccounted native Module
    objects in the cache. Only locally compiled bytes may be deserialized.
    The worker address-space limit also covers compilation and the active module.
    """

    def __init__(self, cache_bytes=128 << 20):
        if type(cache_bytes) is not int or not 0 <= cache_bytes <= 1 << 30:
            raise ValueError("invalid cache capacity")
        config = wasmtime.Config()
        config.consume_fuel = True
        config.parallel_compilation = False
        config.wasm_threads = False
        config.memory_reservation = 0
        config.memory_guard_size = 0
        self.engine = wasmtime.Engine(config)
        self.identity = runtime_identity()
        self.identity_sha256 = compute_canonical_digest(self.identity)
        self.cache, self.cache_size, self.cache_capacity = OrderedDict(), 0, cache_bytes

    def prepare(
        self, source: str, *, proposed_source: str | None = None
    ) -> PreparedCandidate | Failure:
        # Only an explicit test/caller proposal may replace source; no solvers,
        # optimization or proof labels are consulted. Validate the final bytes.
        final = proposed_source if proposed_source is not None else source
        try:
            program = parse(final)
            lowered = lower_body(program)
            source_sha = digest_bytes(program.canonical_source.encode())
            key = compute_canonical_digest(
                {
                    "source": source_sha,
                    "runtime": self.identity_sha256,
                    "helper": self.identity["helper_sha256"],
                    "profiles": profile_digests(),
                }
            )
            hit = key in self.cache
            if hit:
                compiled = self.cache.pop(key)
                self.cache[key] = compiled
                module = wasmtime.Module.deserialize(self.engine, compiled)
            else:
                module = wasmtime.Module(self.engine, lowered)
                compiled = bytes(module.serialize())
                size = len(compiled) + len(key) + 256
                if size <= self.cache_capacity:
                    while self.cache and self.cache_size + size > self.cache_capacity:
                        old_key, old = self.cache.popitem(last=False)
                        self.cache_size -= len(old) + len(old_key) + 256
                    self.cache[key] = compiled
                    self.cache_size += size
            exports = {item.name: item.type for item in module.exports}
            signature = exports.get("compute")
            if (
                module.imports
                or not isinstance(signature, wasmtime.FuncType)
                or [str(v) for v in signature.params] != ["i32"]
                or [str(v) for v in signature.results] != ["i64"] * 4
            ):
                module.close()
                return Failure("invalid_type", "invalid compute signature or imports")
            # Empty body wrapper is a byte-level inventory, not a valid program.
            wrapper = lower_body(Program((), "", ()))
            return PreparedCandidate(
                program,
                module,
                source_sha,
                digest_bytes(lowered.encode()),
                digest_bytes(wrapper.encode()),
                self.identity["helper_sha256"],
                self.identity_sha256,
                len(program.body_tokens),
                len(lowered.replace("(", " ( ").replace(")", " ) ").split()),
                hit,
            )
        except ExecutionFailure as exc:
            return Failure(exc.outcome, exc.reason)
        except wasmtime.WasmtimeError as exc:
            return Failure("invalid_type", f"lowered compilation failed: {exc}")

    def production_term(self, prepared, index, fuel):
        with wasmtime.Store(self.engine) as store:
            store.set_limits(memory_size=0, table_elements=0, instances=1, tables=0, memories=0)
            store.set_fuel(fuel)
            try:
                instance = wasmtime.Instance(store, prepared.module, [])
                exports = instance.exports(store)
                try:
                    value = decode_limbs(exports["compute"](store, index))
                    return value, fuel - store.get_fuel(), None
                except wasmtime.Trap as exc:
                    if exports["__oeis_numeric_limit"].value(store) == 1:
                        failure = Failure("numeric_limit", "checked_signed_256_overflow", "execute")
                    elif exc.trap_code == wasmtime.TrapCode.OUT_OF_FUEL:
                        failure = Failure("execution_limit", "wasmtime_fuel", "execute")
                    else:
                        failure = Failure(
                            "runtime_failure", f"wasmtime_trap:{exc.trap_code}", "execute"
                        )
                    return None, fuel - store.get_fuel(), failure
            except ExecutionFailure as exc:
                return None, fuel - store.get_fuel(), Failure(exc.outcome, exc.reason, "execute")
            except wasmtime.WasmtimeError as exc:
                return (
                    None,
                    fuel - store.get_fuel(),
                    Failure("runtime_failure", str(exc), "execute"),
                )

    def execute(self, prepared, indices, limits=Limits(), *, deadline_ns=None, progress=None):
        indices = list(indices)
        if (
            not indices
            or len(indices) > 100
            or any(type(n) is not int or not 0 <= n <= 99 for n in indices)
            or indices != sorted(set(indices))
        ):
            raise ValueError("indices must be nonempty, unique ascending exact integers in 0..99")
        start = time.monotonic_ns()
        deadline = min(deadline_ns or start + limits.deadline_ns, start + limits.deadline_ns)
        evidence = ExecutionEvidence(
            prepared.source_sha256,
            indices,
            identities={
                "runtime_sha256": prepared.runtime_sha256,
                "helper_sha256": prepared.helper_sha256,
                "lowered_sha256": prepared.lowered_sha256,
                "wrapper_sha256": prepared.wrapper_sha256,
                "language_profile": profile_digests()["language"],
                "resource_profile": profile_digests()["resource"],
                "codec_sha256": profile_digests()["codec"],
                "body_token_count": prepared.body_token_count,
                "module_token_count": prepared.module_token_count,
            },
            cache_hit=prepared.cache_hit,
        )
        for index in indices:
            fuel = min(limits.fuel_per_term, limits.fuel_aggregate - evidence.fuel)
            steps = min(
                limits.reference_per_term, limits.reference_aggregate - evidence.reference_steps
            )
            failure = None
            if time.monotonic_ns() >= deadline or fuel <= 0 or steps <= 0:
                reason = (
                    "candidate_deadline"
                    if time.monotonic_ns() >= deadline
                    else "aggregate_fuel"
                    if fuel <= 0
                    else "aggregate_reference_steps"
                )
                failure = Failure("execution_limit", reason, "execute")
            else:
                value, used, prod_failure = self.production_term(prepared, index, fuel)
                evidence.fuel += used
                evidence.fuel_terms.append(used)
                ref = reference.execute_term(
                    prepared.program, index, step_limit=steps, deadline_ns=deadline
                )
                evidence.reference_steps += ref.steps
                evidence.reference_terms.append(
                    {
                        "index": index,
                        **asdict(ref),
                        "value": str(ref.value) if ref.value is not None else None,
                    }
                )
                if value is not None:
                    evidence.outputs.append(str(value))
                if ref.value is not None:
                    evidence.reference_outputs.append(str(ref.value))
                if prod_failure and prod_failure.outcome == "execution_limit":
                    failure = prod_failure
                elif ref.outcome == "execution_limit":
                    failure = Failure(ref.outcome, ref.reason, "reference")
                elif (
                    prod_failure.outcome if prod_failure else None
                ) != ref.outcome or value != ref.value:
                    failure = Failure(
                        "runtime_failure", "internal_production_reference_disagreement", "reference"
                    )
                elif prod_failure:
                    failure = prod_failure
                if time.monotonic_ns() >= deadline:
                    failure = Failure("execution_limit", "candidate_deadline", "execute")
            if failure:
                evidence.outcome, evidence.reason = failure.outcome, failure.reason
                evidence.first_failure_index, evidence.failure_stage = index, failure.stage
            evidence.elapsed_ns = time.monotonic_ns() - start
            evidence.peak_rss_bytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                1 if platform.system() == "Darwin" else 1024
            )
            evidence.reference_sha256 = compute_canonical_digest(
                {"source_sha256": prepared.source_sha256, "records": evidence.reference_terms}
            )
            if progress:
                progress(evidence.to_dict())
            if failure:
                break
        return evidence

    def evaluate(
        self,
        source,
        indices,
        limits=Limits(),
        *,
        deadline_ns=None,
        progress=None,
        proposed_source=None,
    ):
        start = time.monotonic_ns()
        deadline = min(deadline_ns or start + limits.deadline_ns, start + limits.deadline_ns)
        prepared = self.prepare(source, proposed_source=proposed_source)
        if isinstance(prepared, Failure):
            final = proposed_source if proposed_source is not None else source
            return ExecutionEvidence(
                digest_bytes(final.encode()),
                list(indices),
                fuel=None,
                reference_steps=None,
                elapsed_ns=time.monotonic_ns() - start,
                outcome=prepared.outcome,
                reason=prepared.reason,
                failure_stage="prepare",
            )
        try:
            result = self.execute(
                prepared, indices, limits, deadline_ns=deadline, progress=progress
            )
            result.elapsed_ns = time.monotonic_ns() - start
            return result
        finally:
            prepared.module.close()


def verify(
    evidence: ExecutionEvidence,
    expected_terms: list[str],
    scope: str,
    *,
    run_id: str,
    attempt_index: int,
    prompt_sha256: str,
    checkpoint_sha256: str | None = None,
    purpose="conformance",
    selected=False,
) -> dict:
    """Finite evidence only. Exact truth is mandatory before any scoring."""
    if scope not in ("prefix", "full"):
        raise ValueError("scope must be prefix or full")
    horizon = 20 if scope == "prefix" else 100
    if len(expected_terms) != horizon:
        raise ValueError("missing or wrong-horizon benchmark truth")
    for term in expected_terms:
        parse_integer_text(term, I256_MIN, I256_MAX, "expected_terms")
    if evidence.indices != list(range(horizon)) or any(
        type(i) is not int for i in evidence.indices
    ):
        raise ValueError("evidence indices do not match requested scope")
    outcome, reason, failure = evidence.outcome, evidence.reason, evidence.first_failure_index
    if outcome not in (
        None,
        "invalid_syntax",
        "invalid_type",
        "unsupported",
        "numeric_limit",
        "execution_limit",
        "runtime_failure",
    ):
        raise ValueError("raw execution evidence cannot supply a verified outcome or proof label")
    matched = 0
    for a, b, expected in zip(evidence.outputs, evidence.reference_outputs, expected_terms):
        if a != b or a != expected:
            break
        matched += 1
    if outcome is None:
        if len(evidence.outputs) != horizon or len(evidence.reference_outputs) != horizon:
            outcome, reason = "incomplete_output", "missing execution outputs"
        elif evidence.outputs != evidence.reference_outputs:
            outcome, reason = "runtime_failure", "internal_production_reference_disagreement"
        elif matched != horizon:
            outcome, reason = "wrong_values", "exact truth mismatch"
        else:
            required = {
                "runtime_sha256",
                "helper_sha256",
                "lowered_sha256",
                "wrapper_sha256",
                "language_profile",
                "resource_profile",
                "codec_sha256",
            }
            if not required <= evidence.identities.keys() or evidence.reference_sha256 is None:
                raise ValueError("missing independent execution identity")
            for key in required:
                check_digest(evidence.identities[key], key)
            if (
                evidence.identities["language_profile"] != profile_digests()["language"]
                or evidence.identities["resource_profile"] != profile_digests()["resource"]
                or evidence.identities["codec_sha256"] != profile_digests()["codec"]
            ):
                raise ValueError("execution profile mismatch")
            if (
                evidence.fuel is None
                or evidence.reference_steps is None
                or evidence.fuel > 50_000_000
                or evidence.reference_steps > 50_000_000
                or evidence.elapsed_ns > 2_000_000_000
            ):
                raise ValueError("missing or exceeded resource evidence")
            identity = runtime_identity()
            if (
                evidence.identities["runtime_sha256"] != compute_canonical_digest(identity)
                or evidence.identities["helper_sha256"] != identity["helper_sha256"]
            ):
                raise ValueError("execution runtime identity mismatch")
            if (
                len(evidence.fuel_terms) != horizon
                or any(
                    type(fuel) is not int or not 0 <= fuel <= 1_000_000
                    for fuel in evidence.fuel_terms
                )
                or sum(evidence.fuel_terms) != evidence.fuel
            ):
                raise ValueError("invalid per-term fuel evidence")
            if len(evidence.reference_terms) != horizon:
                raise ValueError("missing per-term independent evidence")
            for index, record in enumerate(evidence.reference_terms):
                if (
                    type(record.get("index")) is not int
                    or record["index"] != index
                    or record.get("value") != evidence.reference_outputs[index]
                    or record.get("outcome") is not None
                    or record.get("reason") is not None
                    or type(record.get("steps")) is not int
                    or not 0 <= record["steps"] <= 1_000_000
                ):
                    raise ValueError("invalid per-term independent evidence")
            if sum(
                record["steps"] for record in evidence.reference_terms
            ) != evidence.reference_steps or evidence.reference_sha256 != compute_canonical_digest(
                {"source_sha256": evidence.source_sha256, "records": evidence.reference_terms}
            ):
                raise ValueError("independent evidence digest/count mismatch")
            outcome = "prefix_match" if scope == "prefix" else "full_horizon_match"
    if (
        outcome not in ("prefix_match", "full_horizon_match")
        and failure is None
        and evidence.failure_stage != "prepare"
    ):
        failure = min(matched, horizon - 1)
    if failure is not None:
        matched = min(matched, failure)

    def metric(value, unit, source):
        return {
            "state": "measured" if value is not None else "unavailable",
            "value": value,
            "unit": unit,
            "source": source if value is not None else None,
            "reason": None if value is not None else "stage did not report this measurement",
        }

    result = {
        "kind": "candidate_result",
        "schema_version": "foundation/v1",
        "purpose": purpose,
        "run_id": run_id,
        "attempt_index": attempt_index,
        "stage": "prepare" if evidence.failure_stage == "prepare" else scope,
        "language_profile": profile_digests()["language"],
        "resource_profile": profile_digests()["resource"],
        "source_sha256": evidence.source_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "prompt_sha256": prompt_sha256,
        "outcome": outcome,
        "reason": reason,
        "outputs": evidence.outputs,
        "verified_terms": matched,
        "first_failure_index": failure,
        "selected": selected,
        "reference_evidence_sha256": evidence.reference_sha256,
        "expected_terms_sha256": compute_canonical_digest({"terms": expected_terms}),
        "usage": {
            "fuel": metric(evidence.fuel, "wasmtime_fuel", "Store.get_fuel"),
            "reference_steps": metric(
                evidence.reference_steps, "ast_instructions", "independent interpreter"
            ),
            "elapsed_ns": metric(evidence.elapsed_ns, "ns", "monotonic clock"),
            "peak_rss_bytes": metric(evidence.peak_rss_bytes, "bytes", "worker getrusage"),
        },
        "proof_status": "not_claimed",
    }
    validate_artifact(result)
    return result
