import pytest
from oeis_learn.pipeline_validation.progressive_outputs import slot
from tests.unit.test_progressive_continuation import campaign


def test_only_declared_ancestor_slots_can_supply_typed_outputs():
    c = campaign()
    plans = c["payload"]["tasks"]
    g = c["payload"]["graph"]
    ref = dict(task="first", operation="generate", output="candidates", kind="candidate")
    assert slot(ref, plans=plans, consumer=plans[1], graph=g) == plans[0]
    for change in [
        dict(task="second"),
        dict(kind="admitted_target"),
        dict(operation="assess"),
        dict(output="../truth"),
    ]:
        with pytest.raises(ValueError):
            slot(ref | change, plans=plans, consumer=plans[1], graph=g)
