"""Independent vectors and generated boundary checks for logical profiles."""

import random
import pytest
from oeis_learn.sandbox.logical_ir import parse_logical
from oeis_learn.sandbox.streaming import LogicalRuntime, LogicalLimits, next_block
from oeis_learn.sandbox.pipeline import decode_limbs
from oeis_learn.sandbox.wat_ast import ExecutionFailure
from oeis_learn.decoder.program_codec import LogicalCodec

MIN = -(1 << 255)
MAX = (1 << 255) - 1


def production(op, a, b):
    source = f"(pure (return ({op} (const {a}) (const {b}))))"
    runtime = LogicalRuntime(source, LogicalLimits())
    runtime.fresh()
    return decode_limbs(runtime.call("compute", 0))


@pytest.mark.parametrize(
    "a,b",
    [(1, -1), (MIN, 1), (MIN, -1), (MAX, 2), (1 << 200, 1 << 54), (-17, 5), (MIN, MIN), (0, 0)],
)
def test_full_product(a, b):
    expected = a * b
    if MIN <= expected <= MAX:
        assert production("mul", a, b) == expected
    else:
        with pytest.raises(ExecutionFailure, match="checked"):
            production("mul", a, b)


@pytest.mark.parametrize(
    "a,b", [(MIN, 1), (MIN, -1), (MAX, -2), (-17, 5), (17, -5), (-17, -5), (1, 0)]
)
@pytest.mark.parametrize("op", ["div", "rem"])
def test_division_hand_vectors(op, a, b):
    if b == 0 or (a == MIN and b == -1):
        with pytest.raises(ExecutionFailure):
            production(op, a, b)
    else:
        q = abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)
        assert production(op, a, b) == (q if op == "div" else a - q * b)


def test_generated_wide_arithmetic():
    rng = random.Random(9001)
    # Reuse an engine across independent source compilation; expectations never
    # use helpers/reference implementations being qualified.
    for _ in range(30):
        a = rng.randrange(MIN, MAX)
        b = rng.randrange(MIN, MAX)
        for op in ("div", "rem"):
            q = abs(a) // abs(b) * (-1 if (a < 0) != (b < 0) else 1)
            assert production(op, a, b) == (q if op == "div" else a - q * b)
        a >>= 130
        b >>= 130
        assert production("mul", a, b) == a * b


@pytest.mark.parametrize("shift", [0, 1, 63, 64, 127, 128, 255])
def test_shifts_and_comparisons(shift):
    for op, expected in [("shr_s", -1), ("shr_u", ((1 << 256) - 1) >> shift)]:
        if expected > MAX:
            expected -= 1 << 256
        r = LogicalRuntime(f"(pure (return ({op} (const -1) {shift})))", LogicalLimits())
        r.fresh()
        assert decode_limbs(r.call("compute", 0)) == expected
    r = LogicalRuntime(
        f"(pure (return (select (lt (const {MIN}) (const {MAX})) (const 7) (const 9))))",
        LogicalLimits(),
    )
    r.fresh()
    assert decode_limbs(r.call("compute", 0)) == 7


@pytest.mark.parametrize(
    "source",
    [
        "(pure (return (shl (const 1) 256)))",
        "(pure (return (shr_u (const 1) -1)))",
        "(pure (return (eq (const 1) (const 2))))",
        "(pure (range-proof 0 1) (return (const 0)))",
    ],
)
def test_invalid_logical_types_and_claims(source):
    with pytest.raises(ValueError):
        parse_logical(source)


def test_codec_and_array_bounds_zero_alias():
    codec = LogicalCodec()
    source = "(pure (store (const 1023) (const -17)) (return (add (load (const 1023)) (load (const 0)))))"
    assert codec.decode(codec.encode(source)) == parse_logical(source).canonical_source
    r = LogicalRuntime(source, LogicalLimits())
    r.fresh()
    assert decode_limbs(r.call("compute", 0)) == -17
    for idx in (-1, 1024, 1 << 64):
        r = LogicalRuntime(f"(pure (return (load (const {idx}))))", LogicalLimits())
        r.fresh()
        with pytest.raises(ExecutionFailure):
            r.call("compute", 0)


STREAM = "(stream (init (set 0 (const 0)) (set 1 (const 1))) (next (set 2 (get 0)) (set 0 (get 1)) (set 1 (add (get 2) (get 1))) (return (get 2))))"


@pytest.mark.parametrize("count", [20, 50, 100, 120])
def test_stream_reset_resume_and_independent_fibonacci(count):
    a, b = 0, 1
    expected = []
    for _ in range(count):
        expected.append(str(a))
        a, b = b, a + b
    first = next_block(STREAM, min(10, count))
    assert first["outcome"] is None
    rest = next_block(STREAM, count - 10, checkpoint=first["checkpoint"])
    assert first["outputs"] + rest["outputs"] == expected
    assert rest["spent_fuel"] > first["spent_fuel"]
    assert next_block(STREAM, count)["outputs"] == expected


PURE_FIB = "(pure (set 0 (const 0)) (set 1 (const 1)) (set 3 (index)) (while (gt (get 3) (const 0)) (set 2 (get 0)) (set 0 (get 1)) (set 1 (add (get 2) (get 1))) (set 3 (sub (get 3) (const 1)))) (return (get 0)))"


@pytest.mark.parametrize("count", [20, 50, 100, 120])
def test_pure_stream_horizon_parity_and_linear_work(count):
    from oeis_learn.sandbox.streaming import execute_logical

    stream = next_block(STREAM, count)
    pure = execute_logical(PURE_FIB, list(range(count)))
    assert pure["outcome"] is None, pure
    assert stream["outputs"] == pure["outputs"]
    assert stream["spent_steps"] <= count * 20 + 10
    assert pure["reference_steps"] > stream["spent_steps"]


def test_pre_next_rollback_keeps_committed_state_and_charges_failures():
    source = (
        "(stream (init) (next (set 0 (add (get 0) (const 1))) (return (div (const 1) (const 0)))))"
    )
    result = next_block(source, 1)
    assert result["outcome"] == "runtime_failure"
    assert result["checkpoint"]["state"]["slots"][0] == 0
    assert result["checkpoint"]["state"]["index"] == 0
    assert result["outputs"] == [] and result["spent_fuel"] > 0
    retry = next_block(source, 1, checkpoint=result["checkpoint"])
    assert retry["spent_fuel"] > result["spent_fuel"]
    assert retry["checkpoint"]["outputs"] == []


def test_resealed_checkpoint_mutation_is_unreachable():
    from copy import deepcopy
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    checkpoint = next_block(STREAM, 2)["checkpoint"]
    changed = deepcopy(checkpoint)
    changed["state"]["slots"][0] += 1
    changed["checkpoint_sha256"] = digest(changed, "checkpoint_sha256")
    result = next_block(STREAM, 1, checkpoint=changed)
    assert result["outcome"] == "runtime_failure" and not result["outputs"]


def test_durable_stream_controller_resume_and_orphan_charge(tmp_path):
    from oeis_learn.sandbox.streaming import StreamController

    c = StreamController(tmp_path / "a", STREAM)
    c.next_block(3)
    restored = StreamController(tmp_path / "a", STREAM)
    assert restored.next_block(3)["checkpoint"]["outputs"] == next_block(STREAM, 6)["outputs"]
    orphan = StreamController(tmp_path / "b", STREAM)
    orphan.events.append(
        "issued-0",
        {
            "count": 3,
            "prior_checkpoint": None,
            "reserved_fuel": 50000000,
            "reserved_steps": 50000000,
        },
    )
    recovered = StreamController(tmp_path / "b", STREAM)
    assert recovered.next_block(3)["outcome"] == "execution_limit"
    assert recovered.next_block(3)["spent_fuel"] == 50000000
