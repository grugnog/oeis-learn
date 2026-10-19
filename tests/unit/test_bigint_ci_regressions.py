"""Keep new worker targets and fixed templates compatible with qualified gates."""

import pytest

from oeis_learn.evaluation.repair_gate import _rust_results
from oeis_learn.fresh.synthesis_policy import Portfolio
from oeis_learn.fresh.template_baseline import template_bank
from oeis_learn.decoder.bigint_codec import decode_body
from oeis_learn.synthesis.reference_parser import parse


def cargo_section(target, passed):
    return (
        target
        + "\n\nrunning "
        + str(passed)
        + " tests\n\n"
        + f"test result: ok. {passed} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s\n\n"
    )


LIB = cargo_section("     Running unittests src/lib.rs (/target/oeis_wasm_evaluator-abc)", 11)
BIN = cargo_section(
    "     Running unittests src/bin/synthesis_worker.rs (/target/synthesis_worker-abc)", 0
)
PHASE7 = cargo_section("     Running tests/phase7.rs (/target/phase7-abc)", 2)
DOC = cargo_section("   Doc-tests oeis_wasm_evaluator", 0)


@pytest.mark.parametrize(
    "log,count",
    [
        (LIB + PHASE7 + DOC, 13),
        (LIB + BIN + PHASE7 + DOC, 13),
        (LIB + BIN.replace("0 passed", "3 passed") + PHASE7 + DOC, 16),
    ],
)
def test_rust_gate_identifies_owning_suites_without_positional_assumptions(tmp_path, log, count):
    path = tmp_path / "rust-tests.txt"
    path.write_text(log)
    assert _rust_results(path) == count


@pytest.mark.parametrize(
    "log",
    [
        LIB + BIN + DOC,
        LIB + BIN + PHASE7,
        LIB + BIN + PHASE7 + PHASE7 + DOC,
        (LIB + BIN + PHASE7 + DOC).replace("2 passed", "1 passed"),
        (LIB + BIN + PHASE7 + DOC).replace("0 failed", "1 failed", 1),
        (LIB + BIN + PHASE7 + DOC).replace("0 ignored", "1 ignored", 1),
        (LIB + BIN + PHASE7 + DOC).replace("0 filtered out", "1 filtered out", 1),
        LIB
        + BIN
        + PHASE7
        + DOC
        + "test result: ok. 99 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out;\n",
        LIB.replace("11 passed", "0 passed") + BIN + PHASE7 + DOC,
    ],
)
def test_rust_gate_still_rejects_incomplete_failed_or_unowned_evidence(tmp_path, log):
    path = tmp_path / "rust-tests.txt"
    path.write_text(log)
    with pytest.raises(ValueError, match="Rust"):
        _rust_results(path)


def test_fixed_template_candidates_bind_the_exact_complete_codec_source():
    from oeis_learn.fresh.bigint_diagnostics import _template_candidate

    originals = template_bank()["sources"]
    policy = Portfolio(start=0, seed=20261016)
    for index, original in enumerate(originals):
        source, tokens = _template_candidate(index)
        assert source == decode_body(tokens)
        assert parse(source, allow_parameters=True) == parse(original, allow_parameters=True)
        attempt = policy.begin(now=index)
        policy.finish_generation(
            attempt["id"], source=source, tokens=tokens, complete=True, now=index + 0.1
        )
    assert len(policy.records) == 8
