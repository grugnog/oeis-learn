"""Native SAT is provisional; admission independently parses final source."""

from dataclasses import replace
import pytest
from oeis_learn.synthesis.admission import (
    VisiblePrompt,
    admit as _admit,
    source_identity,
    reference_identity,
)
from oeis_learn.synthesis.contracts import ResourceProfile


RUNTIME = "sha256:" + "1" * 64


def admit(result, prompt, **kwargs):
    return _admit(result, prompt, runtime_sha256=RUNTIME, **kwargs)


def provisional_fixture(source, prompt, values, limits=ResourceProfile()):
    return dict(
        outcome="assignment_found",
        source=source,
        source_sha256=source_identity(source),
        original_source_sha256=source_identity(source),
        transformed_source=source,
        occurrences=[],
        assignment={},
        indices=list(prompt.indices),
        values=list(values),
        prompt_sha256=prompt.sha256,
        resource_sha256=limits.identity,
        runtime_sha256=RUNTIME,
        reference_sha256=reference_identity(),
        language_profile="bigint-body/v1",
        coverage=len(values),
        coefficient_bits=4096,
        uniqueness="unchecked",
    )


def test_false_fit_and_incomplete_production_never_admitted():
    prompt = VisiblePrompt(("-7", "12"), ("4", "4"))
    source = "(program (return (int 16)))"
    result = provisional_fixture(source, prompt, ("4", "4"))
    assert admit(result, prompt).outcome == "verification_failure"
    result = provisional_fixture("(program (return (int 4)))", prompt, ("4",))
    assert admit(result, prompt).outcome == "verification_failure"


def test_admission_binds_all_identities_and_reference_budget():
    prompt = VisiblePrompt(("-7", "12"), ("4", "4"))
    result = provisional_fixture("(program (return (int 4)))", prompt, prompt.values)
    admitted = admit(result, prompt)
    assert admitted.outcome == "verified_solution" and admitted.verification["indices"] == list(
        prompt.indices
    )
    for field in ("source_sha256", "prompt_sha256", "resource_sha256", "reference_sha256"):
        assert (
            admit({**result, field: "sha256:" + "0" * 64}, prompt).outcome == "verification_failure"
        )
    assert (
        admit({**result, "outcome": "verified_solution"}, prompt).outcome == "verification_failure"
    )
    limits = replace(ResourceProfile(), reference_steps=1)
    result = provisional_fixture("(program (return (int 4)))", prompt, prompt.values, limits=limits)
    assert admit(result, prompt, limits=limits).outcome == "resource_exhausted"


def test_prompt_is_strict_and_has_no_hidden_field():
    for indices, values in [(("1", "0"), ("1", "2")), (("0", "0"), ("1", "1")), (("0",), ())]:
        with pytest.raises(ValueError):
            VisiblePrompt(indices, values)
    with pytest.raises(TypeError):
        VisiblePrompt(("0",), ("1",), hidden_values=("2",))
    # Missing indices stay explicit: there is no manufactured index 0 here.
    assert VisiblePrompt(("-3", "9"), ("1", "2")).indices == ("-3", "9")


@pytest.mark.parametrize(
    "source,indices,values",
    [
        (
            "(program (return (add (mul (param.get p0) $index) (param.get p1))))",
            ["-9", "0", "17"],
            [str(-(1 << 300) * n + 19) for n in [-9, 0, 17]],
        ),
        (
            "(program (return (add (mul (int 2) (param.get p0)) (mul (int 3) (param.get p1)))))",
            ["0"],
            ["1"],
        ),
        ("(program (return (mul (param.get p0) (param.get p0))))", ["0"], ["4"]),
    ],
)
def test_native_fit_independent_admission(synthesis_driver, source, indices, values):
    result = synthesis_driver("fit", source, indices, values)
    assert result["outcome"] == "assignment_found", result
    assert result["assignments"]
    timing = result["timing"]
    assert timing["policy"] == "native-fit-deadlines/v1"
    assert 0 < timing["search_allowance_seconds"] < timing["native_allowance_seconds"] <= 2
    assert 0 < timing["production_reserve_seconds"] <= 0.5
    assert 0 < timing["response_reserve_seconds"] <= 0.1
    assert len(result["searches"]) <= 5
    for search in result["searches"]:
        if search["assignments"] == 8:
            assert search["checks"] == 8 and search["uniqueness"] == "multiple"
    prompt = VisiblePrompt(tuple(indices), tuple(values))
    for item in result["assignments"]:
        assert (
            _admit(item, prompt, runtime_sha256=synthesis_driver.runtime_sha256).outcome
            == "verified_solution"
        ), item


def test_observed_candidate_returns_native_status_and_worker_stays_healthy(
    fitting_client, monkeypatch
):
    # Visible-only diagnostic fixture. This is not a new T096 score or a promise
    # that this seed discovers a fit; even an empty timeout must return its phase.
    import time

    source = (
        "( program ( return ( add ( mul ( add $index ( int 11 ) ) $index ) "
        "( add ( mul ( int -2 ) ( int 11 ) ) ( int 7 ) ) ) ) )"
    )
    prompt = VisiblePrompt(
        tuple(str(n) for n in range(20)), tuple(str(n * n + 11 * n - 15) for n in range(20))
    )
    responses = []
    call = fitting_client.worker.call

    def traced(request, **kwargs):
        response = call(request, **kwargs)
        responses.append(response)
        return response

    monkeypatch.setattr(fitting_client.worker, "call", traced)
    result = fitting_client.fit(
        source,
        prompt,
        policy="eligible-literals/v1",
        seed=75271729,
        deadline=time.monotonic() + 4,
    )
    assert len(responses) == 1 and responses[0]["complete"], responses
    native = result["native"]
    assert native["timing"]["policy"] == "native-fit-deadlines/v1", result
    assert native["outcome"] in {"assignment_found", "unknown", "resource_exhausted"}, result
    assert native["admitted"] is False
    if native["assignments"]:
        assert result["outcome"] == "verified_solution", result
        assert result["admitted"] and all(
            row["outcome"] == "verified_solution" for row in result["admission"]
        ), result
    else:
        assert result["admitted"] == [] and native["reason"], result
    generation = fitting_client.worker.generation
    assert generation == 0  # A delivered status does not retire/restart the worker.
    healthy = fitting_client.execute("(program (return $index))", ["-7", "19"])
    assert healthy["values"] == ["-7", "19"]
    assert responses[-1]["complete"] and fitting_client.worker.generation == generation


def test_native_unsat_replay_rederives_problem(synthesis_driver):
    source = "(program (return (mul (int 2) (param.get p0))))"
    result = synthesis_driver("fit", source, ["0"], ["1"])
    assert result["outcome"] == "unsat_in_scope", result
    replay = synthesis_driver("replay_unsat", source, ["0"], ["1"], evidence=result["evidence"])
    assert replay["outcome"] == "replayed_unsat", replay
    result["evidence"]["constraints_sha256"] = "sha256:" + "0" * 64
    assert (
        synthesis_driver("replay_unsat", source, ["0"], ["1"], evidence=result["evidence"])[
            "outcome"
        ]
        == "verification_failure"
    )


def test_native_control_taint_overflow_and_checked_legacy(synthesis_driver):
    source = "(program (return (select (lt (param.get p0) (int 0)) (int 1) (int 1))))"
    assert synthesis_driver("fit", source, ["0"], ["1"])["outcome"] == "unsupported"
    result = synthesis_driver(
        "fit",
        "i256.const 7 i256.const 7 i256.add",
        ["-7", "19"],
        ["4", "4"],
        language_profile="wat_i256_checked_v1",
        abstraction_policy="eligible-literals/v1",
    )
    assert result["outcome"] == "assignment_found", result
    prompt = VisiblePrompt(("-7", "19"), ("4", "4"))
    for item in result["assignments"]:
        assert (
            _admit(
                item, prompt, runtime_sha256=synthesis_driver.identity_for("wat_i256_checked_v1")
            ).outcome
            == "verified_solution"
        )
    # Unique algebraic output fit is still rejected when a cancelled intermediate overflows.
    source = "(program (return (sub (add (param.get p0) (int 128)) (int 128))))"
    result = synthesis_driver(
        "fit",
        source,
        ["0"],
        ["128"],
        limits=dict(input_bits=8, literal_bits=8, intermediate_bits=8),
    )
    assert result["outcome"] != "assignment_found" and result["outcome"] != "unsat_in_scope", result


def test_retained_reference_guards_cancelled_overflow():
    from oeis_learn.synthesis.reference import evaluate_retained

    hi = (1 << 255) - 1
    result = evaluate_retained(
        f"i256.const {hi} i256.const 1 i256.add i256.const 1 i256.sub", ["0"]
    )
    assert result.outcome == "failure" and result.reason == "numeric_limit"
    assert evaluate_retained("i256.const -1", ["-7"]).values == ("-1",)


def test_scoped_cache_never_promotes_unknown_or_small_bound():
    from oeis_learn.synthesis.evidence import classify_scope, reusable, cache_key

    assert (
        classify_scope(dict(outcome="unsat_in_scope", completed_bounds=[16])) == "incomplete_scope"
    )
    for outcome in ("unknown", "resource_exhausted", "assignment_found"):
        assert not reusable(dict(outcome=outcome), requested_bound=4096)
    assert not reusable(
        dict(outcome="unsat_in_scope", coefficient_bits=16, replay_status="replayed_unsat"),
        requested_bound=4096,
    )
    params = dict(
        source="x",
        indices=["0"],
        values=["1"],
        language="a",
        resources="b",
        solver="c",
        runtime="d",
        reference="e",
        policy="f",
    )
    assert cache_key(**params) != cache_key(**dict(params, reference="changed"))


def test_codec_roundtripped_affine_template_uses_explicit_parameters(synthesis_driver):
    from oeis_learn.decoder.bigint_codec import encode_body,decode_body,fitting_policy
    from oeis_learn.fresh.template_baseline import template_bank
    source=decode_body(encode_body(template_bank()['sources'][2]))
    assert fitting_policy(source)=='none'
    result=synthesis_driver('fit',source,['0','1','2'],['5','8','11'])
    assert result['outcome']=='assignment_found',result
    prompt=VisiblePrompt(('0','1','2'),('5','8','11'))
    assert any(_admit(item,prompt,runtime_sha256=synthesis_driver.runtime_sha256).outcome=='verified_solution'
        for item in result['assignments'])
