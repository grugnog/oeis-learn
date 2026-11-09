"""Independent algebra, adversarial source and resource cases for T070."""

import copy
from dataclasses import replace

import pytest

from oeis_learn.generalization.progressive.contracts import digest
from tests.unit.test_progressive_claims import SOURCE, certificate


def check(source=SOURCE, cert=None, **kwargs):
    from oeis_learn.generalization.progressive.polynomial_certificate import check as replay

    return replay(source, certificate() if cert is None else cert, **kwargs)


@pytest.mark.parametrize(
    "expression, coefficients",
    [
        ("$index", ["0", "1"]),
        ("(int -7)", ["-7"]),
        ("(sub $index $index)", ["0"]),
        ("(sub (mul $index $index) (int 1))", ["-1", "0", "1"]),
        ("(mul (sub $index (int 1)) (add $index (int 1)))", ["-1", "0", "1"]),
        ("(mul (sub (int 2) $index) (add $index (int 3)))", ["6", "-1", "-1"]),
        ("(mul (int 0) (mul $index $index))", ["0"]),
    ],
)
def test_exact_symbolic_identity_with_hand_computed_coefficients(expression, coefficients):
    source = f"(program (return {expression}))"
    cert = certificate(source=digest(source), coefficients=coefficients)
    before = copy.deepcopy(cert)
    result = check(source, cert)
    assert result["statement"] == "proved"
    assert result["scope"] == "program_equals_polynomial_only"
    assert result["domain"] == "all_integer_indices"
    assert result["arithmetic"] == "mathematical_integers"
    assert result["coefficients"] == coefficients
    assert result["certificate"] == digest(cert)
    assert result["source"] == digest(source)
    assert result["machine_safety"] == "unverified"
    assert result["resource_safety"] == "unverified"
    assert result["sequence_law"] == "unproved"
    assert result == check(source, cert) and cert == before


def test_original_true_polynomial_and_exact_source_binding():
    result = check()
    assert result["coefficients"] == ["1", "2", "1"]
    for change in (
        {"coefficients": ["1", "1", "1"]},
        {"source": digest(SOURCE + " ")},
        {"version": "legacy-ir-proof/v1"},
    ):
        with pytest.raises(ValueError):
            check(cert=certificate(**change))


@pytest.mark.parametrize(
    "change",
    [
        {"coefficients": []},
        {"coefficients": ["1", "2", "1", "0"]},
        {"coefficients": [True, "2", "1"]},
        {"coefficients": [1, "2", "1"]},
        {"coefficients": ["01", "2", "1"]},
        {"coefficients": ["-0", "2", "1"]},
        {"coefficients": ["1e0", "2", "1"]},
        {"coefficients": ["9" * 10000]},
        {"arithmetic": "checked_machine"},
        {"assumptions": ["the answer is true"]},
        {"assumptions": None},
        {"trusted": True},
        {"machine_domain": {"lower": "0", "upper": "10", "safe": True}},
        {"machine_domain": {"lower": "10", "upper": "0"}},
        {"machine_domain": {"lower": True, "upper": "10"}},
        {"machine_domain": {"lower": "0", "upper": "9" * 10000}},
    ],
)
def test_closed_canonical_certificate_rejects_forgery(change):
    with pytest.raises(ValueError):
        check(cert=certificate(**change))


@pytest.mark.parametrize(
    "source",
    [
        "(program (return (select (true) $index $index)))",
        "(program (return (param.get p0)))",
        "(program (return (div $index (int 1))))",
        "(program (return (pow $index (int 2))))",
        "(program (local $b0 bigint) (return (int 3)))",
        "(program (local $b0 bigint) (set $b0 (int 3)) (return (get $b0)))",
        "(program (local $b0 bigint) (local $c0 control) (loop $c0 (control 3) (set $b0 (add (get $b0) (int 1)))) (return (get $b0)))",
        "(program (return (int 3))) (program (return (int 3)))",
        "(program (return (add $index)))",
        "(program (return (int +3)))",
        "(program (return (int 03)))",
        "(program (return (int -0)))",
    ],
)
def test_unsupported_or_malformed_source_never_gets_proof(source):
    with pytest.raises(ValueError):
        check(source, certificate(source=digest(source), coefficients=["3"]))


def test_finite_range_is_checked_separately_and_covers_negative_indices():
    result = check(cert=certificate(machine_domain={"lower": "-10", "upper": "10"}))
    assert result["statement"] == "proved"
    assert result["machine_safety"] == "finite_numeric_range_verified"
    assert result["machine_domain"] == {"lower": "-10", "upper": "10"}
    assert result["resource_safety"] == "unverified"
    assert result["numeric_limits"]["intermediate_bits"] == 16384


def test_cancellation_does_not_hide_unsafe_intermediate_or_input():
    power = "$index"
    for _ in range(3):
        power = f"(mul {power} {power})"
    source = f"(program (return (sub {power} {power})))"
    cert = certificate(source=digest(source), coefficients=["0"])
    assert check(source, cert)["machine_safety"] == "unverified"
    # n**8 cancels to zero, but evaluates an intermediate above 16,384 bits.
    with pytest.raises(ValueError, match="range"):
        check(source, cert | {"machine_domain": {"lower": str(2**3000), "upper": str(2**3000)}})
    # Input bounds apply even to a constant polynomial.
    source = "(program (return (int 0)))"
    with pytest.raises(ValueError, match="range"):
        check(
            source,
            certificate(
                source=digest(source),
                coefficients=["0"],
                machine_domain={"lower": "0", "upper": str(2**4096)},
            ),
        )


@pytest.mark.parametrize(
    "bound, value",
    [
        ("source_bytes", 10),
        ("nodes", 4),
        ("nesting", 2),
        ("degree", 1),
        ("coefficient_bits", 1),
        ("work", 1),
    ],
)
def test_each_independent_bound_is_binding(bound, value):
    from oeis_learn.generalization.progressive.polynomial_certificate import Limits

    with pytest.raises(ValueError, match="limit|budget"):
        check(limits=replace(Limits(), **{bound: value}))


def test_coefficients_cannot_explode_even_when_final_polynomial_cancels():
    from oeis_learn.generalization.progressive.polynomial_certificate import Limits

    x = "(mul (int 255) (int 255))"
    source = f"(program (return (sub {x} {x})))"
    with pytest.raises(ValueError, match="coefficient"):
        check(
            source,
            certificate(source=digest(source), coefficients=["0"]),
            limits=replace(Limits(), coefficient_bits=8),
        )
    for change in ({"work": True}, {"nodes": 0}, {"work": Limits().work + 1}):
        with pytest.raises(ValueError):
            replace(Limits(), **change)


def test_checker_never_calls_execution_or_production_lowering(monkeypatch):
    import oeis_learn.synthesis.reference as reference
    import oeis_learn.decoder.bigint_codec as codec

    def forbidden(*args, **kwargs):
        raise AssertionError("proof delegated to execution or production lowering")

    monkeypatch.setattr(reference, "evaluate", forbidden)
    monkeypatch.setattr(codec, "encode_body", forbidden)
    assert check()["statement"] == "proved"


def test_degree_growth_is_bounded_before_dense_allocation():
    expression = "$index"
    for _ in range(7):
        expression = f"(mul {expression} {expression})"
    source = f"(program (return {expression}))"
    assert check(source, certificate(source=digest(source), coefficients=["0"] * 128 + ["1"]))[
        "coefficients"
    ] == ["0"] * 128 + ["1"]
    source = f"(program (return (mul {expression} {expression})))"
    with pytest.raises(ValueError, match="degree limit"):
        check(source, certificate(source=digest(source), coefficients=["0"]))


def test_many_signed_linear_factors_against_independent_algebra():
    for a in range(-4, 5):
        for b in range(-4, 5):
            source = f"(program (return (mul (add $index (int {a})) (add $index (int {b})))))"
            # (x+a)(x+b) = ab + (a+b)x + x^2, independent of the checker.
            coefficients = [str(a * b), str(a + b), "1"]
            assert (
                check(source, certificate(source=digest(source), coefficients=coefficients))[
                    "statement"
                ]
                == "proved"
            )


def test_proof_attempt_is_charged_before_invalid_work_and_honors_deadline():
    import time

    class Budget:
        deadline = time.monotonic() + 10
        charges = []

        def consume(self, **cost):
            self.charges.append(cost)

    budget = Budget()
    with pytest.raises(ValueError):
        check(cert=certificate(coefficients=["0"]), budget=budget)
    assert budget.charges == [{"attempts": 1, "rejections": 1}]
    budget.deadline = time.monotonic() - 1
    with pytest.raises(TimeoutError):
        check(budget=budget)
