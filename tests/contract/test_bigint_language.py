import pytest
from dataclasses import replace

from oeis_learn.synthesis.contracts import (
    ContractError, ResourceProfile, canonical_integer, signed_magnitude, from_signed_magnitude,
)


@pytest.mark.parametrize("text", ["", "-0", "+1", "00", "01", "-01", "1.0", "1e3", " 1", "1\n", "١", 1, True])
def test_only_canonical_strings(text):
    with pytest.raises(ContractError):
        canonical_integer(text)


@pytest.mark.parametrize("bits", [256, 512, 1024, 4096])
def test_signed_neighbors_round_trip(bits):
    for value in [0, (1 << bits) - 1, -((1 << bits) - 1), 1 << (bits - 1), (1 << (bits - 1)) + 1]:
        assert canonical_integer(str(value)) == value
        sign, raw = signed_magnitude(value)
        assert from_signed_magnitude(sign, raw, bits=4096) == value


@pytest.mark.parametrize("sign,raw", [(0,b""),(0,b"\0\0"),(-1,b"\0"),(1,b"\0"),(0,b"\1"),(2,b"\1"),(1,b"\1\0")])
def test_nonminimal_transfer_rejected(sign, raw):
    with pytest.raises(ContractError):
        from_signed_magnitude(sign,raw)


def test_lengths_checked_before_decimal_parse():
    with pytest.raises(ContractError):
        canonical_integer("9" * 100000)
    with pytest.raises(ContractError):
        canonical_integer(str(1 << 4096))


def test_profiles_validate_and_hash_every_resource():
    profile = ResourceProfile()
    assert replace(profile, term_fuel=1).identity != profile.identity
    assert replace(profile, aggregate_fuel=1).identity != profile.identity
    for fields in [{"term_fuel":0},{"term_seconds":float("nan")},{"input_bits":20000},{"source_bytes":True}]:
        with pytest.raises(ContractError):
            replace(profile, **fields)
