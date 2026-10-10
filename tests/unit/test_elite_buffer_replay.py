"""Legacy inspection cannot silently seed teachers or admit unverified source."""

from copy import deepcopy
import pytest
from oeis_learn.rl.elite_buffer import EliteSeedDemonstrationBuffer


@pytest.mark.parametrize(
    "source,terms",
    [
        ("(wat 1)", [1] * 20),
        ('(module (func (export "compute") (param $n i32) (result i64) i64.const 2))', [1] * 20),
    ],
)
def test_invalid_legacy_admission_is_atomic(source, terms):
    buffer = EliteSeedDemonstrationBuffer()
    before = deepcopy(buffer.__dict__)
    with pytest.raises(ValueError):
        buffer.add_canonical_entry("A000012", source, terms)
    assert buffer.__dict__ == before


def test_no_implicit_canaries_or_fake_positive_reward():
    buffer = EliteSeedDemonstrationBuffer()
    assert not buffer.has_entry("A000045")
    assert buffer.sample_dormancy_vulnerable_batch(2, 100) == []
    assert buffer.inject_into_prompt_group("A000045", ["bad", "bad"], [-1.0, -1.0]) == (
        ["bad", "bad"],
        [-1.0, -1.0],
        False,
    )
    assert buffer.sample_demonstration() is None
