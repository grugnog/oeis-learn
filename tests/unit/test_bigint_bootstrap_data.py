import pytest
from oeis_learn.data.bigint_programs import BigintSampler, materialize, canonical_family
from oeis_learn.fresh.bigint_splits import (
    overlap_groups,
    choose_splits,
    diverse_tiny,
    paired_tokens,
)


def test_counter_families_shared_parameters_and_no_rewrite():
    sampler = BigintSampler(seed=20261016)
    rows = [sampler.sample(i) for i in range(64)]
    assert rows == [sampler.sample(i) for i in range(64)]
    assert {r["stage"] for r in rows} == set("ABCD")
    assert len({r["family"] for r in rows[:16]}) == 1
    for row in rows[::16]:
        exact = materialize(row)
        assert len(exact["values"]) == 100 and exact["parameter_source"] == row["parameter_source"]
    assert canonical_family(
        "(program (return (add (param.get p7) (param.get p7))))"
    ) == canonical_family("(program (return (add (param.get p0) (param.get p0))))")
    assert canonical_family(
        "(program (return (add (param.get p0) (param.get p1))))"
    ) != canonical_family("(program (return (add (param.get p0) (param.get p0))))")
    # A dead expression that traps must not disappear through target shortening.
    bad = dict(
        rows[0],
        source="(program (local $c0 control) (set $c0 (to_control (int -1))) (return (int 0)))",
    )
    with pytest.raises(ValueError, match="control_range"):
        materialize(bad)


def test_behavior_prefix_shift_overlap_and_shortfalls():
    def row(key, start):
        return dict(
            id=key,
            values=[str((n + start) ** 2) for n in range(100)],
            indices=list(map(str, range(100))),
            family=key,
        )

    rows = [row("a", 0), row("b", 1), row("c", 300)]
    groups = overlap_groups(rows)
    assert groups["a"] == groups["b"] and groups["a"] != groups["c"]
    with pytest.raises(ValueError, match="shortfall"):
        choose_splits(rows)
    with pytest.raises(ValueError, match="shortfall"):
        diverse_tiny(rows)


def test_paired_token_exclusions_never_simplify():
    row = BigintSampler().sample(0)
    assert paired_tokens(row)["concrete"] and paired_tokens(row)["parameterized"]
    with pytest.raises(ValueError, match="paired"):
        paired_tokens(dict(row, source="(program (return (int " + "1" * 1300 + ")))"))


def test_complete_matched_splits_and_tiny_composition(monkeypatch):
    # Isolate grouping/selection from the separately tested codec boundary.
    # Re-encoding 2,816 independent cubic fixtures adds no splitting coverage.
    monkeypatch.setattr(
        "oeis_learn.fresh.bigint_splits.paired_tokens",
        lambda row: {"concrete": [1], "parameterized": [1]},
    )
    # Independent synthetic rows with 176 distinct structures and 16 variants.
    # Large distinct cubic/quadratic coefficients avoid accidental shift grouping.
    rows = []
    for family in range(176):
        for variant in range(16):
            a = 10000 + family * 101
            b = family + 1
            c = variant * 37
            source = f"(program (return (add (mul (int {a}) (mul $index (mul $index $index))) (add (mul (int {b}) (mul $index $index)) (int {c})))))"
            parameter = source.replace(f"(int {c})", "(param.get p0)")
            rows.append(
                dict(
                    id=f"{family:03}:{variant:02}",
                    family=str(family),
                    variant=variant,
                    stage="ABCD"[family % 4],
                    source=source,
                    parameter_source=parameter,
                    indices=list(map(str, range(100))),
                    values=[str(a * n**3 + b * n * n + c) for n in range(100)],
                )
            )
    result = choose_splits(rows)
    assert [len(v) for v in result["splits"].values()] == [512, 128, 128]
    train, params, structure = result["splits"].values()
    assert {r["family"] for r in params} <= {r["family"] for r in train}
    assert not {r["family"] for r in structure} & {r["family"] for r in train + params}
    used = [{result["groups"][r["id"]] for r in rs} for rs in (train, params, structure)]
    assert not used[0] & used[1] and not used[0] & used[2] and not used[1] & used[2]
    assert len(diverse_tiny(rows)) == 32
