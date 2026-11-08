import pytest

from oeis_learn.generalization.progressive.admission import admit


def member():
    return {
        "id": "a",
        "group": "g",
        "role": "training",
        "source_snapshot": "sha256:" + "a" * 64,
        "indices": list(map(str, range(100))),
        "values": list(map(str, range(100))),
    }


@pytest.mark.parametrize(
    "change", [{"role": "assessment"}, {"values": ["0"] * 20}, {"formula": "imported"}]
)
def test_wrong_membership_short_horizon_and_imported_formula_are_rejected(change):
    with pytest.raises(ValueError):
        admit(member() | change, "(program (return $index))", client=None)


def test_no_native_service_is_not_admission():
    with pytest.raises(ValueError):
        admit(member(), "(program (return $index))", client=None)


def test_native_reference_disagreement_cannot_publish():
    class Disagreement:
        def execute(self, *args, **kwargs):
            return {"outcome": "ok", "values": ["0"] * 100}

    with pytest.raises(ValueError):
        admit(member(), "(program (return $index))", client=Disagreement())


def test_real_native_reference_admission_and_finite_scope(fitting_client):
    value = admit(member(), "(program (return $index))", client=fitting_client)
    assert value["payload"]["status"] == "COMMITTED"
    assert value["payload"]["validity"] == "finite_100_terms_only"
    assert value["track"] == "oeis_derived"
    bad = member()
    bad["values"][-1] = "-1"
    with pytest.raises(ValueError):
        admit(bad, "(program (return $index))", client=fitting_client)
