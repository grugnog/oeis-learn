"""T063 independent expansions and negative component boundaries."""

import copy
import pytest

from oeis_learn.generalization.progressive.contracts import digest


def origins(count=3, groups=2):
    return [
        dict(
            artifact=dict(path=f"origin-{i}.json", bytes=1, sha256=digest(i)),
            role="training",
            group=f"g{i % groups}",
            structure=digest(["structure", i]),
            source=digest(["source", i]),
            path=[],
            bindings=[],
        )
        for i in range(count)
    ]


def definition(id="subtract", body=None, args=None, **changes):
    return (
        dict(
            id=id,
            arguments=args or [dict(name="x", type="bigint"), dict(name="y", type="bigint")],
            returns="bigint",
            body=body or ["sub", ["arg", "x"], ["arg", "y"]],
            dependencies=[],
            origins=origins(),
            assumptions=["bigint-body/v1 checked arithmetic"],
        )
        | changes
    )


def library(*definitions):
    from oeis_learn.generalization.progressive.components import make_library

    return make_library(list(definitions), purpose="acceptance", track="synthetic_strict")


def test_alpha_renaming_preserves_operand_order_and_expanded_work():
    from oeis_learn.generalization.progressive.components import expand, normalize

    a = definition()
    b = definition(
        args=[dict(name="left", type="bigint"), dict(name="right", type="bigint")],
        body=["sub", ["arg", "left"], ["arg", "right"]],
    )
    assert normalize(a) == normalize(b)
    result = expand(library(a), ["call", "subtract", ["int", "3"], ["int", "8"]])
    assert result["source"] == "(program (return (sub (int 3) (int 8))))"
    assert result["expanded_nodes"] > result["calls"] and result["execution_units"] > 1
    from oeis_learn.synthesis.reference import evaluate

    assert evaluate(result["source"], ["0", "9"]).values == ("-5", "-5")


def test_nested_same_named_holes_do_not_capture_caller_operands():
    from oeis_learn.generalization.progressive.components import expand

    reverse = definition(
        id="reverse",
        body=["call", "subtract", ["arg", "y"], ["arg", "x"]],
        dependencies=["subtract"],
    )
    result = expand(
        library(definition(), reverse), ["call", "reverse", ["int", "10"], ["int", "3"]]
    )
    assert result["source"] == "(program (return (sub (int 3) (int 10))))"


@pytest.mark.parametrize(
    "change",
    [
        dict(body=["get", "$b0"]),
        dict(body=["arg", "free"]),
        dict(body=["set", "$b0", ["int", "1"]]),
        dict(body=["call", "subtract", ["arg", "x"], ["arg", "y"]], dependencies=["subtract"]),
        dict(origins=origins(2)),
        dict(origins=origins(3, 1)),
        dict(arguments=[dict(name="x", type="bigint")] * 9),
        dict(origins=[dict(o, role="confirmation") for o in origins()]),
    ],
)
def test_captures_recursion_support_and_training_only_origins_fail_closed(change):
    with pytest.raises(ValueError):
        library(definition(**change))


def test_dependency_cycle_type_mismatch_and_expansion_bomb():
    from oeis_learn.generalization.progressive.components import expand

    a = definition(id="a", body=["call", "b", ["arg", "x"], ["arg", "y"]], dependencies=["b"])
    b = definition(id="b", body=["call", "a", ["arg", "x"], ["arg", "y"]], dependencies=["a"])
    with pytest.raises(ValueError, match="cyclic"):
        library(a, b)
    with pytest.raises(ValueError, match="type"):
        expand(library(definition()), ["call", "subtract", ["true"], ["int", "1"]])
    tree = ["int", "1"]
    for _ in range(12):
        tree = ["add", copy.deepcopy(tree), copy.deepcopy(tree)]
    with pytest.raises(ValueError, match="bound|limit|budget"):
        expand(library(definition()), tree)


def test_dependency_depth_and_lower_runtime_limits_take_precedence():
    from oeis_learn.generalization.progressive.components import expand
    from oeis_learn.synthesis.contracts import ResourceProfile

    defs = [definition(id="c0")]
    for i in range(1, 5):
        defs.append(
            definition(
                id=f"c{i}",
                body=["call", f"c{i - 1}", ["arg", "x"], ["arg", "y"]],
                dependencies=[f"c{i - 1}"],
            )
        )
    with pytest.raises(ValueError, match="depth"):
        library(*defs)
    with pytest.raises(ValueError, match="bound|limit|budget"):
        expand(
            library(definition()),
            ["call", "subtract", ["int", "3"], ["int", "8"]],
            limits=ResourceProfile(source_bytes=20),
        )


def test_unverified_library_cannot_acquire_a_native_or_research_label():
    from oeis_learn.generalization.progressive.components import make_library, validate_library
    from oeis_learn.generalization.progressive.contracts import seal

    with pytest.raises(ValueError, match="native"):
        make_library([definition()], purpose="research", track="synthetic_strict")
    forged = library(definition())
    forged["payload"]["verification"]["status"] = "NATIVE_VERIFIED"
    with pytest.raises(ValueError, match="native"):
        validate_library(seal(forged))
