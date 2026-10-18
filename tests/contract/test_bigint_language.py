import pytest
from dataclasses import replace

from oeis_learn.synthesis.contracts import (
    ContractError,
    ResourceProfile,
    canonical_integer,
    signed_magnitude,
    from_signed_magnitude,
)


@pytest.mark.parametrize(
    "text", ["", "-0", "+1", "00", "01", "-01", "1.0", "1e3", " 1", "1\n", "١", 1, True]
)
def test_only_canonical_strings(text):
    with pytest.raises(ContractError):
        canonical_integer(text)


@pytest.mark.parametrize("bits", [256, 512, 1024, 4096])
def test_signed_neighbors_round_trip(bits):
    for value in [0, (1 << bits) - 1, -((1 << bits) - 1), 1 << (bits - 1), (1 << (bits - 1)) + 1]:
        assert canonical_integer(str(value)) == value
        sign, raw = signed_magnitude(value)
        assert from_signed_magnitude(sign, raw, bits=4096) == value


@pytest.mark.parametrize(
    "sign,raw",
    [(0, b""), (0, b"\0\0"), (-1, b"\0"), (1, b"\0"), (0, b"\1"), (2, b"\1"), (1, b"\1\0")],
)
def test_nonminimal_transfer_rejected(sign, raw):
    with pytest.raises(ContractError):
        from_signed_magnitude(sign, raw)


def test_lengths_checked_before_decimal_parse():
    with pytest.raises(ContractError):
        canonical_integer("9" * 100000)
    with pytest.raises(ContractError):
        canonical_integer(str(1 << 4096))


def test_profiles_validate_and_hash_every_resource():
    profile = ResourceProfile()
    assert replace(profile, term_fuel=1).identity != profile.identity
    assert replace(profile, aggregate_fuel=1).identity != profile.identity
    for fields in [
        {"term_fuel": 0},
        {"term_seconds": float("nan")},
        {"input_bits": 20000},
        {"source_bytes": True},
    ]:
        with pytest.raises(ContractError):
            replace(profile, **fields)


def test_codec_parameter_renaming_round_trip_and_complete_literals():
    from oeis_learn.decoder import bigint_codec as codec

    source = (
        "(program (return (add (param.get p7) (mul (param.get p7) (int -12345678901234567890)))))"
    )
    tokens = codec.encode_body(source)
    decoded = codec.decode_body(tokens)
    assert "p7" not in decoded and decoded.count("p0") == 2
    assert codec.encode_body(decoded) == tokens
    state = codec.BodyState()
    for token in tokens:
        assert token in state.allowed()
        state.consume(token)
    assert state.finished
    for bad in [tokens[:-1], tokens + [codec.PAD_ID], [codec.BOS_ID] + tokens]:
        with pytest.raises(codec.CodecError):
            codec.decode_body(bad)


def test_codec_grammar_distinguishes_handles_control_and_protected_counter():
    from oeis_learn.decoder import bigint_codec as codec

    for source in [
        "(program (return (control 1)))",
        "(program (return (int -0)))",
        "(program (local $c0 control) (loop $c0 (control 2) (set $c0 (control 1))) (return $index))",
    ]:
        with pytest.raises(codec.CodecError):
            codec.encode_body(source)
    prefix = [codec.TOKEN_TO_ID[t] for t in ["(", "program", "(", "return", "("]]
    allowed = {codec.ID_TO_TOKEN[t] for t in codec.allowed_tokens_at(prefix)}
    assert "int" in allowed and "control" not in allowed and "host_init" not in allowed


def test_control_flow_codec_matches_independent_source_parser():
    from oeis_learn.decoder import bigint_codec as codec
    from oeis_learn.synthesis.reference_parser import parse

    source = """(program (local $b0 bigint) (local $c0 control) (local $q0 bool)
      (set $b0 (int 1)) (set $q0 (lt $index (int 0)))
      (if (get $q0) (then (set $b0 (int -1)))
      (else (loop $c0 (to_control $index)
        (set $b0 (mul (get $b0) (to_bigint (increment (get $c0))))))))
      (return (get $b0)))"""
    decoded = codec.decode_body(codec.encode_body(source))
    parse(decoded)
