"""Executable language/codec regressions, including typed prefix masks."""

import pytest

from oeis_learn.decoder import program_codec as c
from oeis_learn.experiments.profiles import I32_LOCAL_NAMES, I64_LOCAL_NAMES, NUMERIC_RANGES


def assert_round_trip(source):
    ids = c.encode_body(source)
    c.validate_body(ids)
    canonical = c.decode_body(ids)
    assert c.encode_body(canonical) == ids
    state = c.BodyState()
    for token in ids:
        assert token in state.allowed(), (canonical, c.ID_TO_TOKEN[token])
        state.consume(token)
    assert state.finished and state.allowed() == []
    assert ids.count(c.EOS_ID) == 1
    assert c.BOS_ID not in ids and c.PAD_ID not in ids
    return ids


def test_special_tokens_and_full_vocabulary():
    assert c.FOUNDATION_VOCAB_SIZE > 128
    assert len({c.PAD_ID, c.BOS_ID, c.EOS_ID}) == 3
    assert "<unk>" not in c.FOUNDATION_VOCABULARY
    ids = assert_round_trip("i256.zero")
    assert any(i >= 128 for i in ids)


@pytest.mark.parametrize("typ", ["i32", "i64", "i256"])
@pytest.mark.parametrize("which", [0, 1])
def test_all_numeric_endpoints(typ, which):
    value = NUMERIC_RANGES[typ][which]
    tail = "" if typ == "i256" else " drop i256.zero"
    assert_round_trip(f"{typ}.const {value}{tail}")
    bad = value + (-1 if which == 0 else 1)
    with pytest.raises(c.CodecError):
        c.encode_body(f"{typ}.const {bad}{tail}")


@pytest.mark.parametrize("local", ["$n", *I64_LOCAL_NAMES, *I32_LOCAL_NAMES])
def test_all_locals_and_type_appropriate_writes(local):
    typ = c.LOCAL_TYPES[local]
    assert_round_trip(
        f"{typ}.const -1 local.tee {local} local.set {local} local.get {local} drop i256.zero"
    )
    wrong = "i64" if typ == "i32" else "i32"
    with pytest.raises(c.CodecError):
        c.encode_body(f"{wrong}.const 1 local.set {local} i256.zero")


@pytest.mark.parametrize(
    "source",
    [
        "( block ( loop i32.const 1 br_if 1 br 0 ) ) i256.zero",
        "i32.const 1 ( if ( then nop ) ( else nop ) ) i256.zero",
        "i32.const 1 ( if ( then nop ) ) i256.zero",
        "( block ( result i64 ) i64.const 4 ) drop i256.zero",
        "i32.const 1 ( if ( result i64 ) ( then i64.const 1 ) ( else i64.const 2 ) ) drop i256.zero",
        "i256.zero return",
        "i256.zero br 0",
        "i256.zero i32.const 1 br_if 0",
        "unreachable",
        "( block unreachable i64.add drop ) i256.zero",
        "i256.const -123456789012345678901234567890 i256.zero i256.add i64.const -4 i256.mul_scalar",
    ],
)
def test_structured_types_and_branch_scope(source):
    assert_round_trip(source)


@pytest.mark.parametrize(
    "source",
    [
        "",
        "nop",
        "i64.const 1",
        "i32.const 1 i64.const 1 i64.add",
        "i64.add i256.zero",
        "i256.zero )",
        "( block i256.zero )",
        "( block br 2 ) i256.zero",
        "br -1",
        "i32.const 1 br_if 0",
        "( block ( result i64 ) nop ) drop i256.zero",
        "i32.const 1 ( if ( result i64 ) ( then i64.const 1 ) ) drop i256.zero",
        "i32.const 1 ( if ( then i64.const 1 ) ( else nop ) ) i256.zero",
        "( local $v0 i64 ) i256.zero",
        "( i64.const 1 ) i256.zero",
        "( module )",
        "local.get $v32 i256.zero",
        "i64.const 12 </int> i256.zero",
        "i256.const +1",
        "i256.const 01",
        "i256.const -0",
        "i256.const 1.0",
        "i256.const 1e3",
        "i256.const ١",
        "i256.zero <eos>",
    ],
)
def test_invalid_bodies_rejected(source):
    with pytest.raises(c.CodecError):
        c.encode_body(source)


def test_masks_partial_literals_and_scope():
    ids = [c.TOKEN_TO_ID["i32.const"], c.INT_ID, c.NEG_ID]
    assert c.EOS_ID not in c.allowed_tokens_at(ids)
    assert c.TERM_ID not in c.allowed_tokens_at(ids)
    assert c.NEG_ID not in c.allowed_tokens_at(ids)
    ids += [c.TOKEN_TO_ID[x] for x in "2147483648"]
    assert c.allowed_tokens_at(ids) == [c.TERM_ID]
    zero = [c.TOKEN_TO_ID["i256.const"], c.INT_ID, c.TOKEN_TO_ID["0"]]
    assert c.allowed_tokens_at(zero) == [c.TERM_ID]
    assert c.allowed_tokens_at([c.TOKEN_TO_ID["i256.zero"], c.TOKEN_TO_ID["br"], c.INT_ID]) == [
        c.TOKEN_TO_ID["0"]
    ]


def test_exact_token_cap_and_reject_truncation():
    ids = c.encode_body("nop " * 1022 + "i256.zero")
    assert len(ids) == 1024
    assert c.encode_body(c.decode_body(ids)) == ids
    for bad in (
        ids[:-1],
        ids + [c.EOS_ID],
        [c.BOS_ID] + ids,
        [c.PAD_ID] + ids,
        [False],
        [1.0],
        [-1],
        [999999],
    ):
        with pytest.raises(c.CodecError):
            c.validate_body(bad)
    with pytest.raises(c.CodecError):
        c.encode_body("nop " * 1023 + "i256.zero")
    with pytest.raises(c.CodecError):
        c.encode_body(" " * 65536 + "i256.zero")


def test_nesting_caps_and_branch_depth_eight():
    assert_round_trip("( block " * 8 + "i256.zero br 8 " + ") " * 8 + "i256.zero")
    assert_round_trip("( loop " * 3 + "br 2 " + ") " * 3 + "i256.zero")
    for kind, count in [("block", 9), ("loop", 4)]:
        with pytest.raises(c.CodecError):
            c.encode_body(f"( {kind} " * count + ") " * count + "i256.zero")


def test_comment_whitespace_canonicalization():
    assert (
        c.decode_body(c.encode_body(";;comment\n (; nested (; x ;) ;) i256.const\n-12"))
        == "i256.const -12"
    )
    with pytest.raises(c.CodecError):
        c.encode_body("i256.zero (; unfinished")


def test_native_mask_acceptance_agrees_with_wasmtime():
    """Independent native type checker catches stack/control bugs in our mask.

    Compile only: generated loops/unreachable are never executed. The sole
    macro used is zero, expanded here to four literal zero instructions.
    """
    import random

    import wasmtime

    engine = wasmtime.Engine()
    rng = random.Random(7002)
    locals_text = " ".join(
        f"(local {name} {typ})" for name, typ in c.LOCAL_TYPES.items() if name != "$n"
    )
    accepted = 0
    for _ in range(100):
        state, ids = c.BodyState(), []
        for step in range(100):
            allowed = [
                i
                for i in state.allowed()
                if not c.ID_TO_TOKEN[i].startswith("i256.") or c.ID_TO_TOKEN[i] == "i256.zero"
            ]
            if not allowed:
                break
            token = c.EOS_ID if c.EOS_ID in allowed and step > 10 else rng.choice(allowed)
            ids.append(token)
            state.consume(token)
            if state.finished:
                source = c.decode_body(ids).replace("i256.zero", "i64.const 0 " * 4)
                module = f'(module (func (export "compute") (param $n i32) (result i64 i64 i64 i64) {locals_text} {source}))'
                wasmtime.Module(engine, module)
                accepted += 1
                break
    assert accepted >= 20, "corpus did not exercise enough complete typed bodies"
