"""Strict qualified/v1 records. Shape, evidence replay and activation are separate.

Canonical decimal strings represent mathematical values; JSON integers represent
bounded counters. Content identities exclude only the identity field itself.
Later-phase record formats are supported without enabling their algorithms.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from oeis_learn.experiments.artifacts import compute_canonical_digest
from oeis_learn.experiments.models import FoundationValidationError, validate_shape

VERSION = "qualified/v1"
D = {"type": "string", "pattern": "^sha256:[0-9a-f]{64}$"}
S = {"type": "string", "minLength": 1, "maxLength": 65536}
I = {"type": "integer", "minimum": 0, "maximum": 2**63 - 1}
B = {"type": "boolean"}
Z = {"type": "string", "pattern": "^(0|-?[1-9][0-9]*)$", "maxLength": 80}
N = {"type": "number"}


def enum(*values):
    return {"enum": list(values)}


def nullable(schema):
    return {"anyOf": [schema, {"type": "null"}]}


def array(schema, minimum=0, maximum=10000):
    return {"type": "array", "items": schema, "minItems": minimum, "maxItems": maximum}


def obj(**fields):
    return {
        "type": "object",
        "properties": fields,
        "required": list(fields),
        "additionalProperties": False,
    }


INDICES = array({"type": "integer", "minimum": 0, "maximum": 2**31 - 1}, 1, 100)
TRANSFORM = obj(
    scale={"type": "integer", "minimum": -1000000, "maximum": 1000000},
    shift={"type": "integer", "minimum": -1000000, "maximum": 1000000},
)
DOMAIN = obj(
    lower_bound={"type": "integer"},
    upper_bound=nullable({"type": "integer"}),
    excluded_points=array({"type": "integer"}),
    excluded_poles=array(obj(denominator=S, scale={"type": "integer"}, shift={"type": "integer"})),
    integer_only={"const": True},
)
METRIC = obj(value=I, unit=enum("ns"), source=S)
PROFILES = obj(
    **{
        name: D
        for name in (
            "language",
            "runtime",
            "codec",
            "model",
            "objective",
            "sampling",
            "search",
            "admission",
        )
    }
)
MODES = (
    "solver",
    "optimizer",
    "native",
    "formula_identity",
    "learning",
    "replay",
    "search",
    "program_proof",
    "extended_runtime",
)
MODE = obj(enabled=B, profile_sha256=nullable(D))
GATES = obj(G7=nullable(D), G8=nullable(D), G9=nullable(D))
# Evidence payloads are validated by their versioned independent checker, not a
# permissive generic "verified" flag. Schemas here only admit bounded objects.
PAYLOAD = {"type": "object", "maxProperties": 64}
FIELDS = {
    "qualified_contract": dict(
        base_contract_sha256=D,
        profiles=PROFILES,
        gates=GATES,
        track=enum("strict_generic"),
        initialization=enum("random", "strict_parent"),
        parent_chain=array(
            obj(
                contract_sha256=D,
                parent_sha256=nullable(D),
                track=enum("strict_generic"),
                initialization=enum("random", "strict_parent"),
            )
        ),
        objective=enum("sft", "reinforce", "grpo"),
        modes=obj(**{m: MODE for m in MODES}),
    ),
    "grounding_result": dict(
        source=S,
        parameter_values=array(Z, 0, 8),
        fit_indices=INDICES,
        visible_prompt_sha256=D,
        solver_method=S,
        solver_version=S,
        coefficient_bound={"type": "integer", "minimum": 0, "maximum": 1000},
        elapsed=METRIC,
        outcome=enum(
            "verified_solution", "proved_unsat_in_scope", "unknown", "timeout", "unsupported"
        ),
        reason=nullable(S),
        numeric_semantics=enum("checked_i256", "modular_i64"),
        execution=nullable(PAYLOAD),
        unsat_evidence=nullable(PAYLOAD),
    ),
    "proof_evidence": dict(
        statement_sha256=D,
        expression_sha256=array(D, 1, 16),
        program_sha256=array(D, 0, 16),
        domain=DOMAIN,
        assumptions=array(S),
        transforms=array(TRANSFORM, 1, 16),
        numeric_semantics=enum(
            "mathematical_rational", "mathematical_integer", "checked_i256", "modular_words"
        ),
        evidence_kind=enum(
            "formula_identity",
            "bounded_machine",
            "polynomial_program",
            "linear_recurrence",
            "finite_state",
            "loop_invariant",
        ),
        checker=S,
        checker_version=S,
        budget_ns=I,
        result=enum(
            "verified", "counterexample", "unknown", "unsupported", "timeout", "empty_domain"
        ),
        certificate=nullable(PAYLOAD),
        witness=nullable(PAYLOAD),
        checker_report=nullable(PAYLOAD),
    ),
    "archive_entry": dict(
        source_sha256=D,
        ir_sha256=D,
        codec_sha256=D,
        runtime_sha256=D,
        origin=enum("generic_sample", "generic_model", "training_prefix_hypothesis"),
        parent_sha256=nullable(D),
        model_sha256=nullable(D),
        sample_sha256=D,
        conditioning_values=array(Z, 1, 20),
        outputs=array(Z, 1, 100),
        fit_indices=INDICES,
        validation_indices=array({"type": "integer", "minimum": 0}, 0, 100),
        execution_sha256=D,
        admission_sha256=D,
        training_eligible=B,
        eligibility_reason=S,
        body_tokens=I,
        context_tokens=I,
        elapsed_ns=I,
        fuel=I,
    ),
    "rollout_record": dict(
        behavior_encoder_sha256=D,
        behavior_decoder_sha256=D,
        visible_prompt_sha256=D,
        actions=array(I, 1),
        supports=array(array(I, 1), 1),
        temperatures=array(N, 1),
        log_probabilities=array(nullable(N), 1),
        forced_mask=array(B, 1),
        pad_mask=array(B, 1),
        eos_mask=array(B, 1),
        rng_sha256=D,
        reward_components=obj(fit=N, validity=N, cost=N),
        execution_sha256=D,
        top_p={"const": 1},
        top_k={"type": "null"},
        rollout_dropout={"const": False},
        scoring_dropout={"const": False},
    ),
    "visit_observation": dict(
        task_sha256=D,
        prompt_sha256=D,
        successes=I,
        attempts=I,
        elapsed_ns=I,
        visit=I,
        step=I,
        selection_probability=N,
        sampling_rule=S,
    ),
    "qualified_checkpoint": dict(
        base_manifest_sha256=D,
        contract_sha256=D,
        modes=obj(
            **{
                m: B
                for m in (
                    "reference",
                    "replay",
                    "self_training",
                    "curriculum",
                    "search",
                    "scaler",
                    "scheduler",
                )
            }
        ),
        reference_encoder_sha256=nullable(D),
        reference_decoder_sha256=nullable(D),
        replay_state=nullable(obj(archive_root_sha256=D, cursor=I)),
        frozen_round=nullable(I),
        curriculum_state=nullable(obj(statistics_sha256=D, rng_sha256=D)),
        search_state=nullable(obj(frontier_sha256=D, cursor=I)),
        sampler_sha256=D,
        prefetch_sha256=D,
        scaler_sha256=nullable(D),
        scheduler_sha256=nullable(D),
        stream_state=nullable(obj(committed_outputs=array(Z), pre_call_state_sha256=D)),
        ledger_sequence=I,
        charged_budget_ns=I,
        previous_sha256=nullable(D),
    ),
    "relation_proposal": dict(
        source_sha256=array(D, 1, 8),
        program_sha256=array(D, 1, 8),
        transforms=array(TRANSFORM, 1, 8),
        coefficients=array(Z, 1, 8),
        structural_origin=S,
        fit_indices=INDICES,
        validation_indices=INDICES,
        final_indices=INDICES,
        rank=nullable(I),
        nullity=nullable(I),
        residuals=array(Z, 1, 100),
        attempts=I,
        elapsed_ns=I,
        seal_sha256=D,
    ),
    "macro_dictionary": dict(
        archive_root_sha256=D,
        entries=array(
            obj(
                internal_id=S,
                expanded_body=S,
                source_sha256=D,
                dependencies=array(S),
                side_conditions=array(S),
                equivalence_sha256=D,
                expansion_tokens=I,
                description_length_benefit={"type": "integer"},
            ),
            1,
            256,
        ),
    ),
    "experiment_decision": dict(
        question=S,
        hypothesis=S,
        baseline_sha256=D,
        challenger_sha256=D,
        code_sha256=D,
        runtime_sha256=D,
        data_sha256=D,
        seeds=array(I, 1, 3),
        budget_ns=I,
        primary_metric=S,
        minimum_useful_effect=N,
        correctness_gates=array(D, 1),
        selection_rule=S,
        stop_rule=S,
        status=enum("adopted", "baseline_retained", "not_triggered", "inconclusive", "failed_gate"),
        active_ns=I,
        charged_ns=I,
        paired_results=array(obj(seed=I, baseline=N, challenger=N), 0, 3),
        stop_reason=S,
        trigger=obj(
            diagnostic_sha256=nullable(D), observed=nullable(N), threshold=N, disposition=S
        ),
        selection_split=enum("development", "generic"),
    ),
}
SCHEMAS = {
    kind: obj(kind={"const": kind}, schema_version={"const": VERSION}, identity_sha256=D, **fields)
    for kind, fields in FIELDS.items()
}


def seal_record(kind: str, **fields: Any) -> dict:
    """Construct a content-addressed record; never authorize it implicitly."""
    record = dict(kind=kind, schema_version=VERSION, **deepcopy(fields))
    record["identity_sha256"] = compute_canonical_digest(record, "identity_sha256")
    validate_record(record)
    return record


def _require(condition, message):
    if not condition:
        raise FoundationValidationError(message)


def _indices(values):
    _require(values == sorted(set(values)), "indices must be unique ascending integers")


def _acyclic(edges):
    visiting, done = set(), set()

    def visit(key):
        _require(key not in visiting, "cyclic identities")
        if key in done:
            return
        visiting.add(key)
        for parent in edges[key]:
            _require(parent in edges, "unresolved dependency identity")
            visit(parent)
        visiting.remove(key)
        done.add(key)

    for key in edges:
        visit(key)


def validate_record(record: dict) -> None:
    """Validate structure and semantic consistency. Evidence replay is explicit."""
    _require(isinstance(record, dict), "qualified record must be an object")
    from oeis_learn.experiments.artifacts import canonical_bytes

    try:
        encoded = canonical_bytes(record)
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise ValueError("qualified record must be finite, acyclic JSON") from exc
    _require(len(encoded) <= 1 << 20, "qualified record exceeds 1 MiB")
    kind = record.get("kind")
    _require(kind in SCHEMAS, "unknown qualified record kind")
    validate_shape(record, SCHEMAS[kind])
    _require(
        record["identity_sha256"] == compute_canonical_digest(record, "identity_sha256"),
        "qualified content identity mismatch",
    )

    # Reject non-finite JSON numbers and cycles before crossing any boundary.
    def finite(value):
        if isinstance(value, float):
            _require(math.isfinite(value), "nonfinite number")
        elif isinstance(value, dict):
            for v in value.values():
                finite(v)
        elif isinstance(value, list):
            for v in value:
                finite(v)

    finite(record)
    for key in ("parameter_values", "conditioning_values", "outputs", "coefficients", "residuals"):
        for value in record.get(key, []):
            _require(-(1 << 255) <= int(value) < (1 << 255), f"{key}: outside signed i256")
    for key in ("fit_indices", "validation_indices", "final_indices"):
        if key in record:
            _indices(record[key])
    identity = record["identity_sha256"]
    for key in ("parent_sha256", "previous_sha256"):
        _require(record.get(key) != identity, "self-referential identity")
    if kind == "qualified_contract":
        chain = record["parent_chain"]
        _require(
            bool(chain) == (record["initialization"] == "strict_parent"),
            "initialization parent chain mismatch",
        )
        seen = set()
        for i, item in enumerate(chain):
            _require(item["contract_sha256"] not in seen | {identity}, "cyclic parent chain")
            seen.add(item["contract_sha256"])
            if i == len(chain) - 1:
                _require(
                    item["initialization"] == "random" and item["parent_sha256"] is None,
                    "parent root must be random",
                )
            else:
                _require(
                    item["initialization"] == "strict_parent"
                    and item["parent_sha256"] == chain[i + 1]["contract_sha256"],
                    "incomplete parent chain",
                )
        for name, mode in record["modes"].items():
            _require(
                mode["enabled"] == (mode["profile_sha256"] is not None),
                "inactive settings or missing profile",
            )
            gate = "G7" if name in MODES[:4] else "G8" if name in MODES[4:7] else "G9"
            _require(
                not mode["enabled"] or record["gates"][gate] is not None, f"{name} requires {gate}"
            )
        _require(
            record["objective"] == "sft" or record["modes"]["learning"]["enabled"],
            "policy gradient requires G8 learning mode",
        )
    elif kind == "grounding_result":
        solution = record["outcome"] == "verified_solution"
        unsat = record["outcome"] == "proved_unsat_in_scope"
        _require(
            solution == (record["execution"] is not None),
            "solution requires exact final execution evidence",
        )
        _require(
            unsat == (record["unsat_evidence"] is not None), "scoped UNSAT requires replay evidence"
        )
        _require(
            all(abs(int(v)) <= record["coefficient_bound"] for v in record["parameter_values"]),
            "parameter outside bound",
        )
        _require(
            solution or not record["parameter_values"],
            "unresolved result cannot provide accepted parameters",
        )
        if solution:
            from dataclasses import fields
            from oeis_learn.sandbox.pipeline import ExecutionEvidence, digest_bytes

            evidence = record["execution"]
            core = {"production", "reference", "fit_terms", "final_source_sha256"}
            expected = (
                core | {f.name for f in fields(ExecutionEvidence)}
                if "outputs" in evidence
                else core | {"scope"}
            )
            _require(set(evidence) == expected, "unknown/incomplete grounding execution fields")
            _require(
                evidence["final_source_sha256"] == digest_bytes(record["source"].encode()),
                "grounding final source mismatch",
            )
            for name in ("production", "reference", "fit_terms"):
                values = evidence[name]
                _require(
                    isinstance(values, list) and len(values) == len(record["fit_indices"]),
                    "grounding execution arity mismatch",
                )
                for value in values:
                    validate_shape(value, Z)
                    _require(-(1 << 255) <= int(value) < (1 << 255), "grounding output range")
            _require(
                evidence["production"] == evidence["reference"] == evidence["fit_terms"],
                "grounding execution does not match fit terms",
            )
            if "outputs" in evidence:
                _require(
                    evidence["indices"] == record["fit_indices"]
                    and evidence["source_sha256"] == evidence["final_source_sha256"]
                    and evidence["outputs"]
                    == evidence["reference_outputs"]
                    == evidence["fit_terms"]
                    and evidence["outcome"] is None,
                    "grounding execution binding mismatch",
                )
            else:
                _require(evidence["scope"] == "visible_prefix", "unsupported grounding scope")
    elif kind == "proof_evidence":
        _require(
            len(record["transforms"]) == len(record["expression_sha256"]), "proof operand arity"
        )
        domain = record["domain"]
        _indices(domain["excluded_points"])
        if record["result"] == "verified":
            _require(
                record["certificate"] is not None and record["checker_report"] is not None,
                "verified requires certificate and checker report",
            )
            _require(
                domain["upper_bound"] is None or domain["lower_bound"] <= domain["upper_bound"],
                "verified empty domain",
            )
        if record["result"] == "counterexample":
            _require(record["witness"] is not None, "counterexample requires witness")
        _require(
            not record["program_sha256"] or record["evidence_kind"] != "formula_identity",
            "formula identity cannot claim program proof",
        )
    elif kind == "archive_entry":
        _require(
            not set(record["fit_indices"]) & set(record["validation_indices"]),
            "archive fit/validation overlap",
        )
        _require(
            record["origin"] != "generic_model" or record["model_sha256"] is not None,
            "model origin lacks model identity",
        )
        _require(
            not record["training_eligible"] or record["body_tokens"] <= record["context_tokens"],
            "over-context program cannot train",
        )
        _require(
            not record["training_eligible"]
            or record["origin"] != "training_prefix_hypothesis"
            or bool(record["validation_indices"]),
            "prefix hypothesis needs independent continuation",
        )
    elif kind == "rollout_record":
        count = len(record["actions"])
        for key in (
            "supports",
            "temperatures",
            "log_probabilities",
            "forced_mask",
            "pad_mask",
            "eos_mask",
        ):
            _require(len(record[key]) == count, "rollout action arity")
        for i, action in enumerate(record["actions"]):
            support = record["supports"][i]
            _require(
                len(set(support)) == len(support) and action in support,
                "action outside unique support",
            )
            _require(record["temperatures"][i] > 0, "positive temperature required")
            p = record["log_probabilities"][i]
            sampled = not (record["forced_mask"][i] or record["pad_mask"][i])
            _require(
                (p is not None and p <= 0) if sampled else p is None,
                "one finite log probability per sampled action",
            )
    elif kind == "visit_observation":
        _require(
            0 <= record["successes"] <= record["attempts"] and record["attempts"] > 0,
            "invalid visit counts",
        )
        _require(0 < record["selection_probability"] <= 1, "invalid selection probability")
    elif kind == "qualified_checkpoint":
        for mode, keys in {
            "reference": ("reference_encoder_sha256", "reference_decoder_sha256"),
            "replay": ("replay_state",),
            "self_training": ("frozen_round",),
            "curriculum": ("curriculum_state",),
            "search": ("search_state",),
            "scaler": ("scaler_sha256",),
            "scheduler": ("scheduler_sha256",),
        }.items():
            _require(
                all((record[k] is not None) == record["modes"][mode] for k in keys),
                "active checkpoint state mismatch",
            )
    elif kind == "relation_proposal":
        count = len(record["coefficients"])
        _require(
            all(len(record[k]) == count for k in ("source_sha256", "program_sha256", "transforms")),
            "relation arity",
        )
        _require(any(int(c) for c in record["coefficients"]), "zero relation")
        identities = [
            (s, t["scale"], t["shift"])
            for s, t in zip(record["source_sha256"], record["transforms"])
        ]
        _require(
            len(set(identities)) == count and all(int(c) != 0 for c in record["coefficients"]),
            "duplicate/zero operands must be canonicalized",
        )
        a, b, c = (set(record[k]) for k in ("fit_indices", "validation_indices", "final_indices"))
        _require(not (a & b or a & c or b & c), "relation partitions overlap")
        _require((record["rank"] is None) == (record["nullity"] is None), "partial rank evidence")
        _require(
            record["rank"] is None or record["rank"] + record["nullity"] == count,
            "rank/nullity mismatch",
        )
    elif kind == "macro_dictionary":
        entries = record["entries"]
        _require(len({e["internal_id"] for e in entries}) == len(entries), "duplicate macro ID")
        _acyclic({e["internal_id"]: e["dependencies"] for e in entries})
        from oeis_learn.sandbox.wat_ast import parse

        for entry in entries:
            program = parse(entry["expanded_body"])
            _require(
                entry["expansion_tokens"] == len(program.body_tokens),
                "macro expansion cost mismatch",
            )
    elif kind == "experiment_decision":
        _require(
            record["active_ns"] <= record["charged_ns"] <= record["budget_ns"],
            "experiment budget mismatch",
        )
        _require(len(set(record["seeds"])) == len(record["seeds"]), "duplicate paired seeds")
        results = record["paired_results"]
        _require(
            len({r["seed"] for r in results}) == len(results)
            and all(r["seed"] in record["seeds"] for r in results),
            "unpaired results",
        )
        if record["status"] == "not_triggered":
            _require(
                record["trigger"]["diagnostic_sha256"] is not None
                and record["trigger"]["observed"] is not None,
                "not_triggered requires measured diagnostic",
            )
        if record["status"] == "adopted":
            _require(bool(results), "adoption requires paired evidence")


def replay_record(record: dict) -> dict:
    """Replay supported evidence in a bounded process, without model weights."""
    validate_record(record)
    from oeis_learn.sandbox.contained import run_contained

    status, value = run_contained(_replay, (record,))
    return {
        "record_sha256": record["identity_sha256"],
        "status": "verified"
        if status == "ok" and value
        else "timeout"
        if status == "timeout"
        else "unknown",
    }


def _replay(record):
    if record["kind"] == "grounding_result":
        from oeis_learn.decoder.grounding_evidence import check_grounding_record

        return check_grounding_record(record)
    if record["kind"] == "proof_evidence" and record["evidence_kind"] == "formula_identity":
        from oeis_learn.discovery.formula_checker import check_scoped_evidence

        return check_scoped_evidence(record)
    # Shape validation of later-phase records is not proof/admission authority.
    return False
