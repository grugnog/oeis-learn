"""Closed recipes and durable identity/isolation precede production integration."""

from copy import deepcopy
import pytest
from oeis_learn.fresh.bigint_config import resolve, defaults


def test_recipe_defaults_are_closed_explicit_and_identity_bound():
    a = resolve(dict(version="bigint-bootstrap/v1"))
    assert a["learning"]["max_updates"] == 3000 and a["evaluation"]["final_access"] == "sealed"
    assert set(a) - {"identity"} == set(defaults())
    assert a == resolve(dict(version="bigint-bootstrap/v1"))
    changed = deepcopy(a)
    changed.pop("identity")
    changed["model"]["seed"] = 20261017
    assert resolve(changed)["identity"] != a["identity"]
    for path in (("oops",), ("model", "oops"), ("fitting", "service_limits", "oops")):
        bad = dict(version="bigint-bootstrap/v1")
        node = bad
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = True
        with pytest.raises(ValueError):
            resolve(bad)


def test_assisted_requires_review_and_no_final_access_in_training_recipe():
    with pytest.raises(ValueError, match="assisted"):
        resolve(dict(version="bigint-bootstrap/v1", track=dict(kind="assisted")))
    with pytest.raises(ValueError, match="sealed"):
        resolve(dict(version="bigint-bootstrap/v1", evaluation=dict(final_access="authorized")))
