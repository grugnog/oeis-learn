import pytest
from oeis_learn.fresh.provenance import root_lineage, descendant, validate_assistance, fresh_cohort

H = "sha256:" + "1" * 64
J = "sha256:" + "2" * 64


def manifest():
    return dict(
        version="assistance/v1",
        review=H,
        sources=(("source", "license", H),),
        demonstrations=((H, J),),
        recipients=("assisted",),
        excluded_groups=(H,),
        exposed_groups=(J,),
        historical_approval=None,
    )


def test_assistance_is_monotone_and_overlap_never_fresh_again():
    root = root_lineage(seed=17, cost=10, checkpoint_sha256=H, recipe_sha256=H)
    assisted = descendant(root, track="assisted", cost=2, manifest=manifest())
    assert assisted["track"] == "assisted" and assisted["foundation_cost"] == 10
    with pytest.raises(ValueError):
        descendant(assisted, track="generic_only", cost=3)
    further = descendant(assisted, track="assisted", cost=3)
    assert further["incremental_cost"] == 5
    with pytest.raises(ValueError):
        fresh_cohort((J,), further)
    with pytest.raises(ValueError):
        validate_assistance({**manifest(), "review": None})
    with pytest.raises(ValueError):
        validate_assistance({**manifest(), "recipients": ("generic_only",)})
    with pytest.raises(ValueError):
        descendant(root, track="generic_only", cost=2, manifest=manifest())


def test_named_demonstrations_and_historical_weights_require_review():
    root = root_lineage(seed=17, cost=10, checkpoint_sha256=H, recipe_sha256=H)
    for origin in ("named_demonstration", "historical_checkpoint", "legacy_replay"):
        with pytest.raises(ValueError):
            descendant(root, track="generic_only", cost=1, origin=origin)
    with pytest.raises(ValueError):
        descendant(
            root, track="assisted", cost=1, origin="historical_checkpoint", manifest=manifest()
        )
    assert (
        descendant(
            root,
            track="assisted",
            cost=1,
            origin="historical_checkpoint",
            manifest={**manifest(), "historical_approval": H},
        )["track"]
        == "assisted"
    )


def test_complete_ancestry_required_and_assistance_cannot_be_relabelled():
    from oeis_learn.fresh.provenance import validate_ancestry
    from oeis_learn.synthesis.contracts import identity

    root = root_lineage(seed=17, cost=10, checkpoint_sha256=H, recipe_sha256=H)
    child = descendant(root, track="assisted", cost=2, manifest=manifest())
    with pytest.raises(ValueError, match="incomplete"):
        validate_ancestry(child, {})
    assert len(validate_ancestry(child, {root["sha256"]: root})) == 2
    forged = {**child, "track": "generic_only", "assistance": (), "origin": "generic_training"}
    forged["sha256"] = identity({k: v for k, v in forged.items() if k != "sha256"})
    grandchild = descendant(child, track="assisted", cost=1)
    grandchild.update(track="generic_only", assistance=(), origin="generic_training")
    grandchild["sha256"] = identity({k: v for k, v in grandchild.items() if k != "sha256"})
    with pytest.raises(ValueError):
        validate_ancestry(grandchild, {child["sha256"]: child, root["sha256"]: root})
