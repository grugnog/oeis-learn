"""Explicit logical runtimes with fresh pure calls and committed stream blocks.

Checkpoint restore replays and verifies committed history from init. Replay is
charged to the same stream budget. Every failing next restores pre-next state.
"""

import time
from dataclasses import dataclass
from oeis_learn.experiments.artifacts import compute_canonical_digest as digest
from oeis_learn.sandbox.logical_ir import parse_logical, profile_identity
from oeis_learn.sandbox.lowering import lower_logical
from oeis_learn.sandbox.reference import LogicalReference
from oeis_learn.sandbox.wat_ast import ExecutionFailure


@dataclass(frozen=True)
class LogicalLimits:
    per_call_fuel: int = 1000000
    aggregate_fuel: int = 50000000
    per_call_steps: int = 1000000
    aggregate_steps: int = 50000000
    timeout_ms: int = 2000

    def validate(self):
        if (
            any(type(v) is not int or v <= 0 for v in vars(self).values())
            or self.per_call_fuel > 1000000
            or self.aggregate_fuel > 50000000
            or self.per_call_steps > 1000000
            or self.aggregate_steps > 50000000
            or self.timeout_ms > 2000
        ):
            raise ValueError("logical resource caps exceeded")


def identity(source, backend="python_wasmtime"):
    import importlib.metadata
    from pathlib import Path

    program = parse_logical(source)
    modules = ("logical_ir.py", "wide_helpers.py", "lowering.py", "reference.py", "streaming.py")
    implementation = digest({p: digest((Path(__file__).parent / p).read_text()) for p in modules})
    return {
        "program_sha256": program.identity,
        "profile_sha256": digest(profile_identity(program.profile)),
        "compiled_sha256": digest(lower_logical(program)),
        "implementation_sha256": implementation,
        "backend": backend,
        "runtime": importlib.metadata.version("wasmtime")
        if backend == "python_wasmtime"
        else __import__(
            "oeis_learn.sandbox.native_runtime", fromlist=["native_identity"]
        ).native_identity(),
    }


class LogicalRuntime:
    """Trusted worker-local runtime. Public callers use bounded process adapters."""

    def __init__(self, source, limits, backend="python_wasmtime"):
        import wasmtime

        limits.validate()
        self.limits = limits
        self.program = parse_logical(source)
        if backend not in ("python_wasmtime", "rust"):
            raise ValueError("explicit logical backend required")
        self.backend = backend
        self.native = None
        self.deadline = time.monotonic() + limits.timeout_ms / 1000
        config = wasmtime.Config()
        config.consume_fuel = True
        config.parallel_compilation = False
        config.wasm_threads = False
        config.memory_reservation = 0
        config.memory_guard_size = 0
        config.memory_reservation_for_growth = 0
        # Fuel checks every generated loop; trusted bounded one-page memory.
        self.engine = wasmtime.Engine(config)
        self.module = None
        if backend == "rust":
            from oeis_learn.sandbox.native_runtime import native_identity

            native_identity()
            from oeis_wasm_evaluator import LogicalModule

            self.native = LogicalModule(lower_logical(self.program), self.program.profile)
        else:
            self.module = wasmtime.Module(self.engine, lower_logical(self.program))
        self.ref = LogicalReference(self.program)
        self.fuel = 0
        self.steps = 0
        self.store = None

    def fresh(self):
        if self.backend == "rust":
            return
        import wasmtime

        self.store = wasmtime.Store(self.engine)
        self.instance = wasmtime.Instance(self.store, self.module, [])
        self.exports = self.instance.exports(self.store)

    def wasm_state(self):
        from oeis_learn.sandbox.pipeline import decode_limbs

        if self.backend == "rust":
            index, limbs, memory = self.native.snapshot()
            return {
                "index": index,
                "slots": [decode_limbs(limbs[s * 4 : s * 4 + 4]) for s in range(32)],
                "array": [
                    int.from_bytes(bytes(memory[i * 32 : (i + 1) * 32]), "little", signed=True)
                    for i in range(1024)
                ],
            }
        memory = bytes(self.exports["__memory"].read(self.store, 0, 32768))
        slots = [
            decode_limbs([self.exports[f"__slot{s}_{i}"].value(self.store) for i in range(4)])
            for s in range(32)
        ]
        array = [
            int.from_bytes(memory[i * 32 : (i + 1) * 32], "little", signed=True)
            for i in range(1024)
        ]
        return {"index": self.exports["__index"].value(self.store), "slots": slots, "array": array}

    def restore_wasm(self, state):
        if self.backend == "rust":
            limbs = []
            for value in state["slots"]:
                for i in range(4):
                    limb = (value >> (64 * i)) % (1 << 64)
                    limbs.append(limb - (1 << 64) if limb >= (1 << 63) else limb)
            self.native.restore(
                state["index"],
                limbs,
                b"".join(v.to_bytes(32, "little", signed=True) for v in state["array"])
                + bytes(32768),
            )
            return
        self.exports["__index"].set_value(self.store, state["index"])
        for s, value in enumerate(state["slots"]):
            for i in range(4):
                limb = ((value % (1 << 256)) >> (64 * i)) % (1 << 64)
                if limb >= 1 << 63:
                    limb -= 1 << 64
                self.exports[f"__slot{s}_{i}"].set_value(self.store, limb)
        self.exports["__memory"].write(
            self.store, b"".join(v.to_bytes(32, "little", signed=True) for v in state["array"])
        )
        self.exports["__oeis_numeric_limit"].set_value(self.store, 0)

    def call(self, name, *args):
        import wasmtime

        if time.monotonic() >= self.deadline:
            raise ExecutionFailure("execution_limit", "logical deadline")
        cap = min(self.limits.per_call_fuel, self.limits.aggregate_fuel - self.fuel)
        if cap <= 0:
            raise ExecutionFailure("execution_limit", "aggregate fuel")
        if self.backend == "rust":
            values, used, outcome, reason = self.native.call(name, args[0] if args else None, cap)
            self.fuel += used
            if outcome:
                raise ExecutionFailure(outcome, reason)
            return values
        self.store.set_fuel(cap)
        try:
            return self.exports[name](self.store, *args)
        except wasmtime.Trap as exc:
            if self.exports["__oeis_numeric_limit"].value(self.store):
                raise ExecutionFailure("numeric_limit", "checked logical arithmetic") from exc
            outcome = (
                "execution_limit"
                if exc.trap_code == wasmtime.TrapCode.OUT_OF_FUEL
                else "runtime_failure"
            )
            raise ExecutionFailure(
                outcome, "fuel" if outcome == "execution_limit" else "logical trap"
            ) from exc
        finally:
            self.fuel += cap - self.store.get_fuel()

    def reference_call(self, method, *args):
        self.ref.steps = 0
        self.ref.max_steps = min(
            self.limits.per_call_steps, self.limits.aggregate_steps - self.steps
        )
        self.ref.deadline = self.deadline
        try:
            return getattr(self.ref, method)(*args)
        finally:
            self.steps += self.ref.steps

    def pair(self, name, method, *args):
        from oeis_learn.sandbox.pipeline import decode_limbs

        production_failure = reference_failure = None
        try:
            raw = self.call(name, *args)
            value = None if name == "init" else decode_limbs(raw)
        except ExecutionFailure as exc:
            production_failure = exc
            value = None
        try:
            expected = self.reference_call(method, *args)
        except ExecutionFailure as exc:
            reference_failure = exc
            expected = None
        failures = [e for e in (production_failure, reference_failure) if e]
        if any(e.outcome == "execution_limit" for e in failures):
            raise ExecutionFailure("execution_limit", "production/reference limit")
        if failures:
            if len(failures) != 2 or failures[0].outcome != failures[1].outcome:
                raise ExecutionFailure(
                    "runtime_failure", "production/reference outcome disagreement"
                )
            raise production_failure
        if value != expected:
            raise ExecutionFailure("runtime_failure", "production/reference value disagreement")
        return value

    def initialize(self, index):
        self.fresh()
        self.pair("init", "init", index)
        if self.wasm_state() != self.ref.state():
            raise ExecutionFailure("runtime_failure", "init/reference disagreement")

    def next(self):

        before = self.wasm_state()
        ref_before = self.ref.state()
        try:
            value = self.pair("next", "next")
            if self.wasm_state() != self.ref.state():
                raise ExecutionFailure("runtime_failure", "stream/reference disagreement")
            return value
        except BaseException:
            self.restore_wasm(before)
            self.ref.restore(ref_before)
            raise


def _stream_block(source, count, start, checkpoint, limits, backend):
    runtime = LogicalRuntime(source, limits, backend)
    binding = identity(source, backend)
    if runtime.program.profile != "stream":
        raise ValueError("explicit stream profile required")
    previous = []
    prior_fuel = prior_steps = 0
    if checkpoint is not None:
        if (
            set(checkpoint)
            != {
                "version",
                "identity",
                "start_index",
                "outputs",
                "state",
                "spent_fuel",
                "spent_steps",
                "checkpoint_sha256",
            }
            or checkpoint["version"] != "logical-stream/v1"
            or checkpoint["identity"] != binding
            or checkpoint["start_index"] != start
            or checkpoint["checkpoint_sha256"] != digest(checkpoint, "checkpoint_sha256")
        ):
            raise ValueError("unbound/corrupt stream checkpoint")
        previous = checkpoint["outputs"]
        prior_fuel = checkpoint["spent_fuel"]
        prior_steps = checkpoint["spent_steps"]
        if (
            any(type(x) is not int for x in (prior_fuel, prior_steps))
            or prior_fuel < 0
            or prior_steps < 0
        ):
            raise ValueError("invalid stream accounting")
    if len(previous) + count > 120:
        raise ValueError("stream horizon cap120")
    runtime.fuel = prior_fuel
    runtime.steps = prior_steps
    new = []
    outcome = reason = None
    committed = checkpoint["state"] if checkpoint else None
    try:
        runtime.initialize(start)
        for old in previous:
            if str(runtime.next()) != old:
                raise ValueError("stream checkpoint history mutation")
        if checkpoint and runtime.wasm_state() != checkpoint["state"]:
            raise ValueError("unreachable checkpoint state")
        committed = runtime.wasm_state()
        for _ in range(count):
            value = runtime.next()
            new.append(str(value))
            committed = runtime.wasm_state()
    except ExecutionFailure as exc:
        outcome, reason = exc.outcome, exc.reason
    if committed is None:
        return {
            "outcome": outcome,
            "reason": reason,
            "outputs": [],
            "checkpoint": None,
            "spent_fuel": runtime.fuel,
            "spent_steps": runtime.steps,
            "failure_index": start,
        }
    record = {
        "version": "logical-stream/v1",
        "identity": binding,
        "start_index": start,
        "outputs": previous + new,
        "state": committed,
        "spent_fuel": runtime.fuel,
        "spent_steps": runtime.steps,
    }
    record["checkpoint_sha256"] = digest(record)
    return {
        "outcome": outcome,
        "reason": reason,
        "outputs": new,
        "checkpoint": record,
        "spent_fuel": runtime.fuel,
        "spent_steps": runtime.steps,
        "failure_index": start + len(previous) + len(new) if outcome else None,
    }


def next_block(
    source,
    count,
    *,
    start_index=0,
    checkpoint=None,
    limits=LogicalLimits(),
    backend="python_wasmtime",
):
    from oeis_learn.sandbox.contained import run_contained

    limits.validate()
    parse_logical(source)
    if (
        type(count) is not int
        or not 1 <= count <= 120
        or type(start_index) is not int
        or not -(1 << 31) <= start_index < (1 << 31)
    ):
        raise ValueError("bounded next_block and exact start index required")
    status, value = run_contained(
        _stream_block,
        (source, count, start_index, checkpoint, limits, backend),
        timeout_ms=limits.timeout_ms,
    )
    if status == "ok":
        return value
    # A killed process cannot deliver actual fuel. Charge the issued upper bound;
    # retry starts with this conservative spent amount instead of refunding work.
    charged = dict(checkpoint) if checkpoint else None
    if charged:
        charged["spent_fuel"] = limits.aggregate_fuel
        charged["spent_steps"] = limits.aggregate_steps
        charged["checkpoint_sha256"] = digest(charged, "checkpoint_sha256")
    return {
        "outcome": "execution_limit" if status == "timeout" else "runtime_failure",
        "reason": value,
        "outputs": [],
        "checkpoint": charged,
        "spent_fuel": limits.aggregate_fuel,
        "spent_steps": limits.aggregate_steps,
        "cost_kind": "charged_upper_bound",
    }


def _pure(source, indices, limits, backend):

    runtime = LogicalRuntime(source, limits, backend)
    if runtime.program.profile != "pure":
        raise ValueError("pure profile required")
    outputs = []
    outcome = reason = None
    failure = None
    for index in indices:
        try:
            runtime.fresh()
            value = runtime.pair("compute", "compute", index)
            outputs.append(str(value))
        except ExecutionFailure as exc:
            outcome, reason, failure = exc.outcome, exc.reason, index
            break
    return {
        "identity": identity(source, backend),
        "outputs": outputs,
        "outcome": outcome,
        "reason": reason,
        "failure_index": failure,
        "fuel": runtime.fuel,
        "reference_steps": runtime.steps,
    }


def execute_logical(source, indices, *, limits=LogicalLimits(), backend="python_wasmtime"):
    from oeis_learn.sandbox.contained import run_contained

    limits.validate()
    program = parse_logical(source)
    if (
        program.profile != "pure"
        or not indices
        or len(indices) > 120
        or any(type(i) is not int or not -(1 << 31) <= i < (1 << 31) for i in indices)
    ):
        raise ValueError("bounded explicit pure indices required")
    status, result = run_contained(
        _pure, (source, list(indices), limits, backend), timeout_ms=limits.timeout_ms
    )
    return (
        result
        if status == "ok"
        else {
            "outputs": [],
            "outcome": "execution_limit" if status == "timeout" else "runtime_failure",
            "reason": result,
        }
    )


def save_stream_checkpoint(path, record):
    """Atomic committed state/index publication using the existing artifact writer."""
    from pathlib import Path
    from oeis_learn.experiments.artifacts import canonical_bytes, atomic_write, ArtifactPath

    path = Path(path)
    if record["checkpoint_sha256"] != digest(record, "checkpoint_sha256"):
        raise ValueError("checkpoint digest")
    atomic_write(ArtifactPath(path.parent, path.name), canonical_bytes(record))


class StreamController:
    """Durable single-writer stream accounting; crash retries cannot refund work."""

    def __init__(
        self, root, source, *, start_index=0, limits=LogicalLimits(), backend="python_wasmtime"
    ):
        from pathlib import Path
        from oeis_learn.experiments.artifacts import (
            load_json,
            canonical_bytes,
            atomic_write,
            ArtifactPath,
        )
        from oeis_learn.tracking.foundation_metrics import EventLog

        limits.validate()
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.source, self.start, self.limits, self.backend = source, start_index, limits, backend
        self.contract = {
            "version": "stream-controller/v1",
            "identity": identity(source, backend),
            "start_index": start_index,
            "limits": vars(limits),
        }
        contract_path = self.root / "contract.json"
        if contract_path.exists():
            if load_json(contract_path.read_bytes()) != self.contract:
                raise ValueError("frozen stream contract mismatch")
        else:
            atomic_write(ArtifactPath(self.root, "contract.json"), canonical_bytes(self.contract))
        self.events = EventLog(self.root / "events")
        self.checkpoint = None
        self.orphaned = False
        self.sequence = 0
        self.exhausted = False
        for key, value in self.events.events.items():
            if key == f"issued-{self.sequence}":
                if value["prior_checkpoint"] != (
                    self.checkpoint["checkpoint_sha256"] if self.checkpoint else None
                ):
                    raise ValueError("stream journal lineage mismatch")
                self.orphaned = True
            elif key == f"completed-{self.sequence}" and self.orphaned:
                path = self.root / (value["result_sha256"][7:] + ".json")
                record = load_json(path.read_bytes())
                if digest(record) != value["result_sha256"]:
                    raise ValueError("stream result mutation")
                self.checkpoint = record["checkpoint"]
                self.orphaned = False
                self.sequence += 1
                self.exhausted = (
                    record.get("cost_kind") == "charged_upper_bound" or record["checkpoint"] is None
                )
            else:
                raise ValueError("invalid stream journal ordering")

    def next_block(self, count):
        from oeis_learn.experiments.artifacts import canonical_bytes, atomic_write, ArtifactPath

        if self.orphaned or self.exhausted:
            return {
                "outcome": "execution_limit",
                "reason": "orphaned issued work conservatively consumes remaining stream budget",
                "outputs": [],
                "checkpoint": self.checkpoint,
                "spent_fuel": self.limits.aggregate_fuel,
                "spent_steps": self.limits.aggregate_steps,
                "cost_kind": "charged_upper_bound",
            }
        if type(count) is not int or not 1 <= count <= 120:
            raise ValueError("bounded block count required")
        self.events.append(
            f"issued-{self.sequence}",
            {
                "count": count,
                "prior_checkpoint": self.checkpoint["checkpoint_sha256"]
                if self.checkpoint
                else None,
                "reserved_fuel": self.limits.aggregate_fuel,
                "reserved_steps": self.limits.aggregate_steps,
            },
        )
        self.orphaned = True
        result = next_block(
            self.source,
            count,
            start_index=self.start,
            checkpoint=self.checkpoint,
            limits=self.limits,
            backend=self.backend,
        )
        key = digest(result)
        atomic_write(ArtifactPath(self.root, key[7:] + ".json"), canonical_bytes(result))
        self.events.append(f"completed-{self.sequence}", {"result_sha256": key})
        self.sequence += 1
        self.orphaned = False
        self.checkpoint = result["checkpoint"]
        # A process killed before delivering any state consumes the budget;
        # prevent a new checkpoint=None attempt from starting with fresh fuel.
        if result.get("cost_kind") == "charged_upper_bound" or result["checkpoint"] is None:
            self.exhausted = True
        return result


def _random_access(source, indices, limits, backend):
    runtime = LogicalRuntime(source, limits, backend)
    if runtime.program.profile != "stream":
        raise ValueError("stream random-access adapter required")
    outputs = []
    outcome = reason = None
    try:
        for n in indices:
            runtime.initialize(0)
            value = None
            for _ in range(n + 1):
                value = runtime.next()
            outputs.append(str(value))
    except ExecutionFailure as exc:
        outcome, reason = exc.outcome, exc.reason
    return {
        "outputs": outputs,
        "outcome": outcome,
        "reason": reason,
        "fuel": runtime.fuel,
        "reference_steps": runtime.steps,
        "execution_mode": "stream_reset_replay_random_access",
        "identity": identity(source, backend),
    }


def random_access_stream(source, indices, *, limits=LogicalLimits(), backend="python_wasmtime"):
    """Explicit expensive reset/replay wrapper; never presented as pure O(1) access."""
    from oeis_learn.sandbox.contained import run_contained

    limits.validate()
    program = parse_logical(source)
    if (
        program.profile != "stream"
        or not indices
        or len(indices) > 120
        or any(type(n) is not int or not 0 <= n < 120 for n in indices)
    ):
        raise ValueError("bounded stream replay indices required")
    status, result = run_contained(
        _random_access, (source, list(indices), limits, backend), timeout_ms=limits.timeout_ms
    )
    return (
        result
        if status == "ok"
        else {
            "outputs": [],
            "outcome": "execution_limit" if status == "timeout" else "runtime_failure",
            "reason": result,
        }
    )
