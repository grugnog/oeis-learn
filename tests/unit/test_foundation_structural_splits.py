import pytest
from oeis_learn.data.structural_splits import (
    StructuralHoldouts,
    structural_signature,
    split_for_structure,
    structural_report,
)


def test_alpha_normalization_preserves_dependencies_and_literals_do_not_split():
    a = "i64.const 3 local.set $v0 local.get $v0 i64.const 0 i64.const 0 i64.const 0"
    b = "i64.const 7 local.set $v5 local.get $v5 i64.const 0 i64.const 0 i64.const 0"
    c = "i64.const 7 local.set $v5 local.get $v6 i64.const 0 i64.const 0 i64.const 0"
    assert structural_signature(a) == structural_signature(b)
    assert structural_signature(a) != structural_signature(c)
    report = structural_report([a, b, c])
    assert len(report["groups"]) == 2
    assert all(
        row["split"] in {"train", "development", "final"} for row in report["groups"].values()
    )


def test_expanded_guards_are_independent_and_exact():
    source = "i256.const 1"
    signature = structural_signature(source)
    seed = next(i for i in range(100) if split_for_structure(signature, i) == "train")
    base = StructuralHoldouts(seed=seed)
    assert base.decision(source, [1] * 20) == (True, "eligible")
    assert (
        StructuralHoldouts(seed=seed, reserved_structures=frozenset({signature})).decision(
            source, [1] * 20
        )[1]
        == "structural_holdout"
    )
    assert base.decision("i256.const " + str(1 << 128), [1] * 20)[1] == "parameter_holdout"
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    assert (
        StructuralHoldouts(seed=seed, reserved_prefixes=frozenset({digest(["1"] * 20)})).decision(
            source, [1] * 20
        )[1]
        == "oeis_prefix_holdout"
    )
    with pytest.raises(ValueError):
        base.decision(source, [1.0] * 20)


def test_qualified_draw_assigns_split_before_independent_parameter_variants():
    from oeis_learn.data.generic_programs import QualifiedGenericSampler, default_config

    sampler = QualifiedGenericSampler(default_config())
    accepted = []
    for counter in range(30):
        try:
            a = sampler.sample_variant(counter, 0)
            b = sampler.sample_variant(counter, 1)
        except ValueError:
            continue
        assert a["structure_sha256"] == b["structure_sha256"]
        assert a["split"] == b["split"]
        assert structural_signature(a["sample"]["canonical_source"]) == a["structure_sha256"]
        with pytest.raises(ValueError, match="partition"):
            sampler.sample_variant(counter, 2, split="final" if a["split"] != "final" else "train")
        accepted.append(a)
    assert accepted
