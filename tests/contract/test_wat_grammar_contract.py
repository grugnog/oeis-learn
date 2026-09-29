"""Contract test for WAT EBNF grammar tokens and tokenization round-trip."""

import pytest
from oeis_learn.decoder.wat_grammar import (
    ID_TO_TOKEN,
    TOKEN_TO_ID,
    VOCAB_SIZE,
    WAT_VOCABULARY,
    decode_wat_tokens,
    encode_wat,
    tokenize_wat,
)


def test_grammar_vocabulary_completeness():
    assert len(WAT_VOCABULARY) == VOCAB_SIZE
    assert "( " in [f"{t} " for t in WAT_VOCABULARY]
    assert "i64.add" in WAT_VOCABULARY
    assert "local.get" in WAT_VOCABULARY
    assert "$n" in WAT_VOCABULARY
    assert "$n64" in WAT_VOCABULARY
    assert "i64.extend_i32_u" in WAT_VOCABULARY
    assert '"compute"' in WAT_VOCABULARY


def test_opcode_signatures_completeness():
    from oeis_learn.decoder.wat_grammar import OPCODE_SIGNATURES
    assert "i64.add" in OPCODE_SIGNATURES
    assert OPCODE_SIGNATURES["i64.add"] == (("i64", "i64"), ("i64",))
    assert OPCODE_SIGNATURES["i32.eqz"] == (("i32",), ("i32",))
    assert OPCODE_SIGNATURES["i64.extend_i32_u"] == (("i32",), ("i64",))


def test_tokenization_and_encoding_roundtrip():
    wat_code = '(module (func (export "compute") (param $n i32) (result i64) (i64.const 42)))'
    tokens = tokenize_wat(wat_code)

    assert tokens[0] == "("
    assert tokens[1] == "module"
    assert tokens[2] == "("
    assert tokens[3] == "func"
    assert tokens[4] == "("
    assert tokens[5] == "export"
    assert tokens[6] == '"compute"'

    encoded = encode_wat(wat_code)
    assert all(idx != TOKEN_TO_ID["<unk>"] for idx in encoded)

    decoded = decode_wat_tokens(encoded)
    assert "( module ( func" in decoded
