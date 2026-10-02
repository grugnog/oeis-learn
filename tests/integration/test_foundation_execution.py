"""Strict runtime regressions including real external process supervision."""

import time
import uuid
import pytest
from oeis_learn.sandbox.pipeline import Runtime, Limits, Failure, verify, digest_bytes
from oeis_learn.sandbox.worker_pool import WorkerPool, _send, _receive


def test_real_worker_executes_and_persists_idempotent_results(tmp_path):
    with WorkerPool(tmp_path) as pool:
        result = pool.evaluate("i256.zero", range(20), request_id="one")
        assert result.outcome is None and result.outputs == ["0"] * 20
        same = pool.evaluate("i256.zero", range(20), request_id="one")
        assert same.to_dict() == result.to_dict()
        with pytest.raises(ValueError):
            pool.evaluate("i256.const 1", range(20), request_id="one")
    with WorkerPool(tmp_path) as pool:
        assert pool.evaluate("i256.zero", range(20), request_id="one").to_dict() == result.to_dict()


def _blocked_worker(channel, cache_bytes):
    _send(channel, {"type": "ready"})
    _receive(channel)
    time.sleep(20)


def test_blocked_worker_deadline_and_reaping(tmp_path):
    with WorkerPool(tmp_path, _worker_target=_blocked_worker) as pool:
        start = time.monotonic()
        result = pool.evaluate(
            "i256.zero", range(20), request_id="blocked", limits=Limits(deadline_ns=50_000_000)
        )
        assert result.outcome == "execution_limit"
        assert result.reason == "external_candidate_deadline"
        assert time.monotonic() - start < 2.1
        assert pool.all_slots[0].process is None
        assert result.fuel is None and result.reference_steps is None


def test_canonical_runtime_cache_fresh_state_and_aggregate_limits():
    runtime = Runtime(cache_bytes=1 << 20)
    source = "local.get $v0 i64.const 1 i64.add local.tee $v0 i64.const 0 i64.const 0 i64.const 0"
    one = runtime.evaluate(source, range(5))
    assert one.outputs == ["1"] * 5
    assert runtime.evaluate(source, [0]).cache_hit
    limited = runtime.evaluate(source, range(100), Limits(fuel_aggregate=one.fuel))
    assert limited.outcome == "execution_limit" and len(limited.outputs) == 5
    assert limited.fuel == one.fuel
    assert runtime.cache_size <= runtime.cache_capacity


def test_final_source_is_checked_and_never_trusts_a_solver_claim():
    runtime = Runtime()
    proposal = "i256.const 4 i64.const 4 i256.mul_scalar"
    evidence = runtime.evaluate("placeholder", range(20), proposed_source=proposal)
    result = verify(
        evidence,
        ["4"] * 20,
        "prefix",
        run_id=str(uuid.uuid4()),
        attempt_index=0,
        prompt_sha256=digest_bytes(b"visible"),
    )
    assert result["outcome"] == "wrong_values"
    assert result["source_sha256"] == digest_bytes(proposal.encode())
    invalid = runtime.evaluate(
        "i256.zero", [0], proposed_source="i64.const 1 local.tee $missing i256.zero"
    )
    assert invalid.outcome == "invalid_type"


@pytest.mark.parametrize(
    "source,outcome",
    [
        ("(module)", "unsupported"),
        ("global.get $__oeis_numeric_limit", "unsupported"),
        ("call $i256_add", "unsupported"),
        ("i64.const 1", "invalid_type"),
        ("i256.zero )", "invalid_syntax"),
        (f"i256.const {2**255}", "numeric_limit"),
        ("unreachable", "runtime_failure"),
        (f"i256.const {2**255 - 1} i256.const 1 i256.add", "numeric_limit"),
        ("( loop br 0 ) i256.zero", "execution_limit"),
    ],
)
def test_classified_rejections(source, outcome):
    result = Runtime().evaluate(source, [0], Limits(fuel_per_term=1000, reference_per_term=1000))
    assert result.outcome == outcome


def _blocked_compile(channel, cache_bytes):
    from oeis_learn.sandbox.worker_pool import _worker

    def block(*args, **kwargs):
        time.sleep(20)

    Runtime.prepare = block
    _worker(channel, cache_bytes)


def _blocked_reference(channel, cache_bytes):
    from oeis_learn.sandbox import reference
    from oeis_learn.sandbox.worker_pool import _worker

    def block(*args, **kwargs):
        time.sleep(20)

    reference.execute_term = block
    _worker(channel, cache_bytes)


@pytest.mark.parametrize("target", [_blocked_compile, _blocked_reference])
def test_external_watchdog_covers_both_engines_and_compile(tmp_path, target):
    with WorkerPool(tmp_path, _worker_target=target) as pool:
        result = pool.evaluate(
            "i256.zero", [0], request_id="blocked-stage", limits=Limits(deadline_ns=50_000_000)
        )
        assert result.outcome == "execution_limit"
        assert result.reason == "external_candidate_deadline"
        assert pool.all_slots[0].process is None


def test_killed_worker_replaced_and_duplicate_delivery(tmp_path):
    with WorkerPool(tmp_path) as pool:
        old_pid = pool.all_slots[0].process.pid
        pool.all_slots[0].process.kill()
        pool.all_slots[0].process.join(2)
        failed = pool.evaluate("i256.zero", [0], request_id="killed")
        assert failed.outcome == "runtime_failure"
        first = pool.submit("i256.zero", range(100), request_id="fresh")
        duplicate = pool.submit("i256.zero", range(100), request_id="fresh")
        assert first is duplicate
        assert first.result().outputs == ["0"] * 100
        assert pool.all_slots[0].process.pid != old_pid
        assert pool.evaluate("i256.zero", [0], request_id="killed").to_dict() == failed.to_dict()


def test_single_batch_parity_and_explicit_backend(tmp_path):
    from oeis_learn.sandbox.runner import FoundationRunner, WasmRunner, decode_i256_limbs

    for backend in ("auto", "rust", None):
        with pytest.raises(ValueError):
            FoundationRunner(tmp_path, backend=backend)
    with pytest.raises(ValueError):
        WasmRunner().run_single("i256.zero", result_profile="wat_i256_checked_v1")
    with FoundationRunner(tmp_path, backend="python_wasmtime", workers=2) as runner:
        source = "i256.const 1 i64.const -1 i256.mul_scalar"
        single = runner.run_single(source, range(20), request_id="single")
        batch = runner.run_batch(
            [source, "i256.zero"], range(20), request_ids=["batch-1", "batch-2"]
        )
        assert single.outputs == batch[0].outputs == ["-1"] * 20
        assert single.fuel == batch[0].fuel and single.reference_steps == batch[0].reference_steps
        assert batch[1].outputs == ["0"] * 20
    for wrong in ([], [0] * 3, [0] * 5, [False, 0, 0, 0], [2**63, 0, 0, 0], "0000"):
        with pytest.raises(ValueError):
            decode_i256_limbs(wrong)


def test_legacy_batch_uses_lowered_source_and_fresh_instances():
    from oeis_learn.sandbox.runner import WasmRunner

    runner = WasmRunner(use_fallback=True)
    source = '(module (func (export "compute") (param i32) (result i64 i64 i64 i64) i256.const 1 i64.const -1 i256.mul_scalar))'
    single = runner.run_single(source, terms_to_generate=3, result_profile="i256x4_v1")
    batch = runner.run_batch([source], terms_to_generate=3, result_profile="i256x4_v1")[0]
    assert single.output == batch.output == [-1] * 3
    assert single.total_fuel == batch.total_fuel == single.max_fuel * 3
    mutable = '(module (global $g (mut i64) (i64.const 0)) (func (export "compute") (param i32) (result i64) global.get $g i64.const 1 i64.add global.set $g global.get $g))'
    assert runner.run_single(mutable, terms_to_generate=3).output == [1, 1, 1]


def test_reference_aggregate_and_lru_are_bounded():
    runtime = Runtime(cache_bytes=45000)
    first = runtime.evaluate("i256.zero", [0])
    limited = runtime.evaluate(
        "i256.zero", range(3), Limits(reference_aggregate=first.reference_steps * 2)
    )
    assert limited.outcome == "execution_limit"
    assert limited.reason == "aggregate_reference_steps"
    assert limited.outputs == ["0", "0"]
    for i in range(20):
        assert runtime.evaluate(f"i256.const {i}", [0]).outcome is None
        assert runtime.cache_size <= 45000
    assert len(runtime.cache) < 20
    assert not Runtime(cache_bytes=0).evaluate("i256.zero", [0]).cache_hit


def test_queue_and_message_bounds(tmp_path):
    from queue import Full

    with WorkerPool(tmp_path, _worker_target=_blocked_worker) as pool:
        futures = [pool.submit("i256.zero", [0], request_id=str(i)) for i in range(32)]
        with pytest.raises(Full):
            pool.submit("i256.zero", [0], request_id="33")
        with pytest.raises(ValueError, match="1 MiB"):
            pool.submit(" " * 2**20, [0], request_id="huge")
        for future in futures[1:]:
            future.cancel()
        assert futures[0].result().outcome == "execution_limit"


def test_declared_type_scope_and_source_caps():
    sources = [
        "( local $v0 i64 ) i256.zero",
        "memory.size i256.zero",
        "i64.const 1 local.set $c0 i256.zero",
        "i32.const 1 br_if 2 i256.zero",
        "( block " * 9 + "nop" + ") " * 9 + "i256.zero",
        "nop " * 1025 + "i256.zero",
    ]
    for source in sources:
        assert isinstance(Runtime().prepare(source), Failure)
    assert Runtime().evaluate(" " * 65537 + "i256.zero", [0]).outcome == "execution_limit"


def test_inherited_deadline_does_not_change_idempotency_key(tmp_path):
    with WorkerPool(tmp_path) as pool:
        first = pool.evaluate(
            "i256.zero", [0], request_id="phase", deadline_ns=time.monotonic_ns() + 1_000_000_000
        )
        assert first.outcome is None
        # Completed evidence remains reusable when the enclosing phase expired.
        assert (
            pool.evaluate(
                "i256.zero", [0], request_id="phase", deadline_ns=time.monotonic_ns() - 1
            ).to_dict()
            == first.to_dict()
        )
    with WorkerPool(tmp_path / "blocked", _worker_target=_blocked_worker) as pool:
        begin = time.monotonic()
        result = pool.evaluate(
            "i256.zero", [0], request_id="short", deadline_ns=time.monotonic_ns() + 30_000_000
        )
        assert result.outcome == "execution_limit" and time.monotonic() - begin < 2.1
