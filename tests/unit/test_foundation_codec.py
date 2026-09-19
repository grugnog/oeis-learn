"""Regression tests for the foundation `wat_body_decimal_v1` codec (T006).

Tests target signed-256/i64/i32 bounds, negative and large constants, every
declared local/branch scope, partially emitted literals, valid EOS, no UNK,
no truncation and exact canonical token round trips. The vocabulary must
exceed 128 tokens so a fixed-width mask cannot silently drop tokens.
"""

import pytest

from oeis_learn.decoder.program_codec import (
    CodecError,
    FOUNDATION_VOCAB_SIZE,
    decode_body,
    encode_body,
    allowed_tokens_at,
    validate_body,
    FOUNDATION_VOCABULARY,
)
from oeis_learn.experiments.profiles import I256_MAX, I256_MIN

# ---------------------------------------------------------------------------
# Vocabulary invariants
# ---------------------------------------------------------------------------


def test_vocab_size_above_128():
    assert FOUNDATION_VOCAB_SIZE > 128, "fixed-width mask must not drop tokens"


def test_no_unk_pad_bos_only_eos():
    assert "<eos>" in FOUNDATION_VOCABULARY
    for banned in ("<unk>", "<pad>", "<bos>"):
        assert banned not in FOUNDATION_VOCABULARY


def test_all_declared_locals_present():
    for local in ("$n", "$v31", "$c7", "$v0", "$c0"):
        assert local in FOUNDATION_VOCABULARY


def test_all_allowed_operators_present():
    # Fixed opcode tokens: every allowed operator must be encodable.
    for op in ("i64.const", "i64.add", "i256.const", "i256.mul_scalar", "local.get", "br_if"):
        assert op in FOUNDATION_VOCABULARY


# ---------------------------------------------------------------------------
# Round trips and bounds
# ---------------------------------------------------------------------------


def test_i64_const_round_trip():
    src = "( local.get $n ) i64.const 9223372036854775807 </int> i64.add"
    ids = encode_body(src)
    assert ids[-1] == FOUNDATION_VOCABULARY.index("<eos>")
    assert decode_body(ids) == src


def test_i64_min_round_trip():
    src = "i64.const -9223372036854775808 </int>"
    assert decode_body(encode_body(src)) == src


def test_negative_constant():
    src = "i32.const -1 </int>"
    assert decode_body(encode_body(src)) == src


def test_signed_256_extremes_round_trip():
    lo = f"i256.const {I256_MIN} </int>"
    hi = f"i256.const {I256_MAX} </int>"
    assert decode_body(encode_body(lo)) == lo
    assert decode_body(encode_body(hi)) == hi


def test_large_multilimb_constant():
    src = "i256.const 12345678901234567890 </int> -98765432109876543210 </int>"
    assert decode_body(encode_body(src)) == src


def test_i32_bounds_round_trip():
    src = "i32.const -2147483648 </int> i32.const 2147483647 </int>"
    assert decode_body(encode_body(src)) == src


# ---------------------------------------------------------------------------
# Scopes
# ---------------------------------------------------------------------------


def test_local_scope_round_trip():
    src = "( local $v0 i64 ) ( local $c7 i32 ) local.get $v31 local.set $c0"
    assert decode_body(encode_body(src)) == src


def test_branch_scope_round_trip():
    src = "block end br 2 </int> br_if 1 </int> loop end"
    assert decode_body(encode_body(src)) == src


# ---------------------------------------------------------------------------
# Invalid token sequences
# ---------------------------------------------------------------------------


def test_partially_emitted_literal_rejected():
    # Digit sequence without the explicit terminator is an incomplete immediate.
    src = "i64.const 12 </int>"
    ids = encode_body(src)  # eos is appended after terminator, so indices: [i64,1,2,</int>,eos]
    # Remove both terminator (index -2) and eos (index -1) to leave digits unterminated
    ids = ids[:-2]  # now [i64,1,2]
    with pytest.raises(CodecError):
        decode_body(ids)


def test_no_unknown_token():
    with pytest.raises(CodecError):
        encode_body("i64.bogus_op")


def test_eos_exactly_once_at_end():
    ids = encode_body("i64.const 1 </int>")
    assert ids.count(FOUNDATION_VOCABULARY.index("<eos>")) == 1
    assert ids[-1] == FOUNDATION_VOCABULARY.index("<eos>")
    # EOS appearing mid-sequence is invalid.
    with pytest.raises(CodecError):
        validate_body([FOUNDATION_VOCABULARY.index("<eos>"), 0])


def test_empty_body_rejected():
    with pytest.raises(CodecError):
        validate_body([])


def test_no_truncation_over_1024_tokens():
    body = " ".join(["i64.const 1 </int>"] * 600)  # 3 tokens each = 1800 > 1024
    with pytest.raises(CodecError):
        encode_body(body)


def test_source_over_64kib_rejected():
    with pytest.raises(CodecError):
        encode_body("i64.const " + "9" * 70000 + " </int>")


def test_exact_canonical_round_trip():
    program = (
        "( local $v0 i64 ) ( local $v1 i64 ) "
        "local.get $n i64.const 0 </int> i64.add "
        "i256.const -57896044618658097711785492504343953926634992332820282019728792003956564819968 </int> "
        "i256.zero i256.add local.set $v31 end"
    )
    ids = encode_body(program)
    assert decode_body(ids) == program
    assert encode_body(decode_body(ids)) == ids


# ---------------------------------------------------------------------------
# Incremental mask must not silently drop high-id tokens
# ---------------------------------------------------------------------------


def test_mask_allows_high_id_tokens():
    # The mask at some position must admit a token id >= 128; otherwise a
    # fixed-width (128-wide) mask would silently drop vocabulary members.
    ids = encode_body("i64.const 1 </int>")
    allowed = set()
    for position in range(len(ids) + 1):
        allowed |= set(allowed_tokens_at(ids[:position]))
    assert any(tid >= 128 for tid in allowed), "mask truncates vocabulary at 128"
