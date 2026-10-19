"""Native SAT is provisional; admission independently parses final source."""
from dataclasses import replace
import pytest
from oeis_learn.synthesis.admission import VisiblePrompt, admit as _admit, source_identity, reference_identity
from oeis_learn.synthesis.contracts import ResourceProfile


RUNTIME = "sha256:" + "1"*64

def admit(result, prompt, **kwargs):
    return _admit(result, prompt, runtime_sha256=RUNTIME, **kwargs)


def provisional_fixture(source, prompt, values, limits=ResourceProfile()):
    return dict(outcome="assignment_found", source=source, source_sha256=source_identity(source),
                original_source_sha256=source_identity(source), transformed_source=source,
                occurrences=[], assignment={}, indices=list(prompt.indices), values=list(values),
                prompt_sha256=prompt.sha256, resource_sha256=limits.identity, runtime_sha256=RUNTIME,
                reference_sha256=reference_identity(), language_profile="bigint-body/v1",
                coverage=len(values), coefficient_bits=4096, uniqueness="unchecked")


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
    assert admitted.outcome == "verified_solution" and admitted.verification["indices"] == list(prompt.indices)
    for field in ("source_sha256", "prompt_sha256", "resource_sha256", "reference_sha256"):
        assert admit({**result, field: "sha256:" + "0"*64}, prompt).outcome == "verification_failure"
    assert admit({**result, "outcome": "verified_solution"}, prompt).outcome == "verification_failure"
    limits = replace(ResourceProfile(), reference_steps=1)
    result = provisional_fixture("(program (return (int 4)))", prompt, prompt.values, limits=limits)
    assert admit(result, prompt, limits=limits).outcome == "resource_exhausted"


def test_prompt_is_strict_and_has_no_hidden_field():
    for indices, values in [(("1", "0"), ("1", "2")), (("0", "0"), ("1", "1")), (("0",), ())]:
        with pytest.raises(ValueError): VisiblePrompt(indices, values)
    with pytest.raises(TypeError): VisiblePrompt(("0",), ("1",), hidden_values=("2",))
    # Missing indices stay explicit: there is no manufactured index 0 here.
    assert VisiblePrompt(("-3", "9"), ("1", "2")).indices == ("-3", "9")


@pytest.mark.parametrize("source,indices,values", [
    ("(program (return (add (mul (param.get p0) $index) (param.get p1))))", ["-9","0","17"], [str(-(1<<300)*n+19) for n in [-9,0,17]]),
    ("(program (return (add (mul (int 2) (param.get p0)) (mul (int 3) (param.get p1)))))", ["0"], ["1"]),
    ("(program (return (mul (param.get p0) (param.get p0))))", ["0"], ["4"]),
])
def test_native_fit_independent_admission(synthesis_driver, source, indices, values):
    result = synthesis_driver("fit", source, indices, values)
    assert result["outcome"] == "assignment_found", result
    assert result["assignments"]
    prompt = VisiblePrompt(tuple(indices), tuple(values))
    for item in result["assignments"]:
        assert _admit(item, prompt, runtime_sha256=synthesis_driver.runtime_sha256).outcome == "verified_solution", item


def test_native_unsat_replay_rederives_problem(synthesis_driver):
    source = "(program (return (mul (int 2) (param.get p0))))"
    result = synthesis_driver("fit", source, ["0"], ["1"])
    assert result["outcome"] == "unsat_in_scope", result
    replay = synthesis_driver("replay_unsat", source, ["0"], ["1"], evidence=result["evidence"])
    assert replay["outcome"] == "replayed_unsat", replay
    result["evidence"]["constraints_sha256"] = "sha256:" + "0"*64
    assert synthesis_driver("replay_unsat", source, ["0"], ["1"], evidence=result["evidence"])["outcome"] == "verification_failure"
