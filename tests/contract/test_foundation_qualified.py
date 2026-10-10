"""Every qualified kind has positive and adversarial contract coverage."""

from oeis_learn.sandbox.wat_ast import parse

from copy import deepcopy
import pytest
from oeis_learn.experiments.qualified_models import (
    seal_record,
    validate_record,
    replay_record,
    MODES,
)
from oeis_learn.experiments.artifacts import compute_canonical_digest
from oeis_learn.experiments.models import FoundationValidationError

D = "sha256:" + "1" * 64
E = "sha256:" + "2" * 64


def test_record_boundary_rejects_oversize_and_cyclic_objects():
    with pytest.raises(ValueError, match="1 MiB"):
        validate_record({"kind": "grounding_result", "source": "x" * (1 << 20)})
    cyclic = {"kind": "grounding_result"}
    cyclic["execution"] = cyclic
    with pytest.raises(ValueError, match="acyclic"):
        validate_record(cyclic)


def fixtures():
    modes = {m: {"enabled": False, "profile_sha256": None} for m in MODES}
    return {
        "qualified_contract": dict(
            base_contract_sha256=D,
            profiles={
                k: D
                for k in (
                    "language",
                    "runtime",
                    "codec",
                    "model",
                    "objective",
                    "sampling",
                    "search",
                    "admission",
                )
            },
            gates={"G7": None, "G8": None, "G9": None},
            track="strict_generic",
            initialization="random",
            parent_chain=[],
            objective="sft",
            modes=modes,
        ),
        "grounding_result": dict(
            source="i256.const 7",
            parameter_values=[],
            fit_indices=[0],
            visible_prompt_sha256=D,
            solver_method="not_run",
            solver_version="not_run",
            coefficient_bound=1000,
            elapsed={"value": 1, "unit": "ns", "source": "clock"},
            outcome="unknown",
            reason="diagnostic",
            numeric_semantics="checked_i256",
            execution=None,
            unsat_evidence=None,
        ),
        "proof_evidence": dict(
            statement_sha256=D,
            expression_sha256=[D],
            program_sha256=[],
            domain={
                "integer_only": True,
                "lower_bound": 0,
                "upper_bound": None,
                "excluded_points": [],
                "excluded_poles": [],
            },
            assumptions=["external definition"],
            transforms=[{"scale": 1, "shift": 0}],
            numeric_semantics="mathematical_rational",
            evidence_kind="formula_identity",
            checker="fraction",
            checker_version="v1",
            budget_ns=1,
            result="unknown",
            certificate=None,
            witness=None,
            checker_report=None,
        ),
        "archive_entry": dict(
            source_sha256=D,
            ir_sha256=D,
            codec_sha256=D,
            runtime_sha256=D,
            origin="generic_sample",
            parent_sha256=None,
            model_sha256=None,
            sample_sha256=D,
            conditioning_values=["7"],
            outputs=["7"],
            fit_indices=[0],
            validation_indices=[1],
            execution_sha256=D,
            admission_sha256=D,
            training_eligible=True,
            eligibility_reason="checked",
            body_tokens=4,
            context_tokens=1024,
            elapsed_ns=1,
            fuel=1,
        ),
        "rollout_record": dict(
            behavior_encoder_sha256=D,
            behavior_decoder_sha256=D,
            visible_prompt_sha256=D,
            actions=[1],
            supports=[[1, 2]],
            temperatures=[1.0],
            log_probabilities=[-0.5],
            forced_mask=[False],
            pad_mask=[False],
            eos_mask=[True],
            rng_sha256=D,
            reward_components={"fit": 1, "validity": 1, "cost": 0},
            execution_sha256=D,
            top_p=1,
            top_k=None,
            rollout_dropout=False,
            scoring_dropout=False,
        ),
        "visit_observation": dict(
            task_sha256=D,
            prompt_sha256=D,
            successes=1,
            attempts=2,
            elapsed_ns=1,
            visit=1,
            step=1,
            selection_probability=0.5,
            sampling_rule="uniform",
        ),
        "qualified_checkpoint": dict(
            base_manifest_sha256=D,
            contract_sha256=D,
            modes={
                m: False
                for m in (
                    "reference",
                    "replay",
                    "self_training",
                    "curriculum",
                    "search",
                    "scaler",
                    "scheduler",
                )
            },
            reference_encoder_sha256=None,
            reference_decoder_sha256=None,
            replay_state=None,
            frozen_round=None,
            curriculum_state=None,
            search_state=None,
            sampler_sha256=D,
            prefetch_sha256=D,
            scaler_sha256=None,
            scheduler_sha256=None,
            stream_state=None,
            ledger_sequence=1,
            charged_budget_ns=1,
            previous_sha256=None,
        ),
        "relation_proposal": dict(
            source_sha256=[D, E],
            program_sha256=[D, E],
            transforms=[{"scale": 1, "shift": 0}] * 2,
            coefficients=["1", "-1"],
            structural_origin="generated dataflow",
            fit_indices=[0],
            validation_indices=[1],
            final_indices=[2],
            rank=1,
            nullity=1,
            residuals=["0"],
            attempts=1,
            elapsed_ns=1,
            seal_sha256=D,
        ),
        "macro_dictionary": dict(
            archive_root_sha256=D,
            entries=[
                dict(
                    internal_id="m0",
                    expanded_body="i256.const 7",
                    source_sha256=D,
                    dependencies=[],
                    side_conditions=[],
                    equivalence_sha256=D,
                    expansion_tokens=len(parse("i256.const 7").body_tokens),
                    description_length_benefit=1,
                )
            ],
        ),
        "experiment_decision": dict(
            question="throughput",
            hypothesis="gain",
            baseline_sha256=D,
            challenger_sha256=E,
            code_sha256=D,
            runtime_sha256=D,
            data_sha256=D,
            seeds=[7001],
            budget_ns=100,
            primary_metric="solved groups/hour",
            minimum_useful_effect=0.05,
            correctness_gates=[D],
            selection_rule="paired",
            stop_rule="hard cap",
            status="inconclusive",
            active_ns=1,
            charged_ns=1,
            paired_results=[],
            stop_reason="missing hardware",
            trigger={
                "diagnostic_sha256": None,
                "observed": None,
                "threshold": 0.05,
                "disposition": "inconclusive",
            },
            selection_split="development",
        ),
    }


def reseal(record):
    record["identity_sha256"] = compute_canonical_digest(record, "identity_sha256")
    return record


@pytest.mark.parametrize("kind", fixtures())
def test_kind_positive_unknown_fields_versions_and_identity(kind):
    record = seal_record(kind, **fixtures()[kind])
    validate_record(record)
    for change in ({"unknown": 1}, {"schema_version": "qualified/v9"}, {"identity_sha256": E}):
        bad = {**deepcopy(record), **change}
        with pytest.raises(FoundationValidationError):
            validate_record(bad)


@pytest.mark.parametrize(
    "kind,change",
    [
        ("qualified_contract", {"track": "teacher"}),
        ("grounding_result", {"outcome": "proved_unsat_in_scope"}),
        ("proof_evidence", {"result": "verified"}),
        ("archive_entry", {"origin": "imported_teacher"}),
        ("rollout_record", {"log_probabilities": [None]}),
        ("visit_observation", {"attempts": 0}),
        ("qualified_checkpoint", {"reference_encoder_sha256": D}),
        ("relation_proposal", {"final_indices": [1]}),
        (
            "macro_dictionary",
            {
                "entries": [
                    dict(
                        internal_id="m0",
                        expanded_body="i256.const 7",
                        source_sha256=D,
                        dependencies=["m0"],
                        side_conditions=[],
                        equivalence_sha256=D,
                        expansion_tokens=len(parse("i256.const 7").body_tokens),
                        description_length_benefit=1,
                    )
                ]
            },
        ),
        ("experiment_decision", {"status": "not_triggered"}),
    ],
)
def test_kind_specific_adversary(kind, change):
    with pytest.raises(ValueError):
        seal_record(kind, **{**fixtures()[kind], **change})


@pytest.mark.parametrize("value", [True, 1.0, "1"])
def test_exact_counter_types(value):
    with pytest.raises(ValueError):
        seal_record("visit_observation", **{**fixtures()["visit_observation"], "attempts": value})


def test_inactive_settings_and_parent_chain():
    fields = fixtures()["qualified_contract"]
    fields["modes"]["solver"]["profile_sha256"] = D
    with pytest.raises(ValueError):
        seal_record("qualified_contract", **fields)
    fields = fixtures()["qualified_contract"]
    fields["initialization"] = "strict_parent"
    fields["parent_chain"] = [
        dict(
            contract_sha256=D,
            parent_sha256=D,
            track="strict_generic",
            initialization="strict_parent",
        )
    ]
    with pytest.raises(ValueError):
        seal_record("qualified_contract", **fields)


def test_real_grounding_solution_unsat_and_tamper_replay():
    from oeis_learn.decoder.grounding import ground

    for source, terms in [
        ("i256.const_?", [7] * 20),
        ("i256.const 1", [2] * 20),
        ("i256.const_? i64.const 2 i256.mul_scalar", [1] * 20),
    ]:
        record = ground(source, terms).to_qualified()
        assert replay_record(record)["status"] == "verified", record
        bad = deepcopy(record)
        bad["visible_prompt_sha256"] = E
        assert replay_record(reseal(bad))["status"] == "unknown"


def test_real_scoped_formula_and_witness_replay():
    from oeis_learn.discovery.symbolic_prover import SymbolicProver
    from oeis_learn.discovery.formula_checker import qualified_proof

    for formulas in (["(n-1)/(n-1)", "1"], ["n", "n+1"]):
        _, evidence = SymbolicProver()._run(formulas, [1, -1])
        record = qualified_proof(evidence)
        assert replay_record(record)["status"] == "verified"
        bad = deepcopy(record)
        bad["domain"]["lower_bound"] += 1
        assert replay_record(reseal(bad))["status"] == "unknown"


def test_candidate_results_cannot_claim_proof():
    from oeis_learn.experiments.models import validate_artifact
    from pathlib import Path
    import json

    rows = json.loads(Path("tests/fixtures/foundation/artifacts.json").read_text())
    rows = rows if isinstance(rows, list) else list(rows.values())
    candidate = next(r for r in rows if isinstance(r, dict) and r.get("kind") == "candidate_result")
    candidate["proof_status"] = "PROVEN"
    with pytest.raises(ValueError):
        validate_artifact(candidate)
