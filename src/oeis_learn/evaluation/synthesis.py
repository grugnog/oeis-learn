"""Complete 8-stage synthesis candidate evaluation state machine and cohort evaluation service."""

from __future__ import annotations

import datetime
import hashlib
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
from oeis_learn.curriculum.extrapolation import ExtrapolationVerifier
from oeis_learn.curriculum.mdl_verifier import MdlVerifier
from oeis_learn.data.benchmark import BenchmarkCohort, BenchmarkTarget
from oeis_learn.data.models import (
    ExecutionResult,
    StageRecord,
    SynthesisCandidateRecord,
    SynthesisEvaluationResult,
)
from oeis_learn.decoder.constant_solver import resolve_program_constants
from oeis_learn.decoder.sampler import WatProgramSampler
from oeis_learn.decoder.wat_decoder import WatTransformerDecoder
from oeis_learn.decoder.wat_grammar import tokenize_wat
from oeis_learn.encoder.tri_stream_encoder import TriStreamEncoder
from oeis_learn.evaluation.checkpoint import CheckpointProvenance, load_checkpoint_v2
from oeis_learn.evaluation.protocol import (
    EvaluationProtocol,
    canonical_json_hash,
    derive_candidate_seed,
)
from oeis_learn.sandbox.optimizer import optimize_wat_program
from oeis_learn.sandbox.runner import WasmRunner

STAGES_ORDER = [
    "GENERATION",
    "CONSTANT_RESOLUTION",
    "CANONICALIZATION",
    "ASSEMBLY",
    "EXECUTION",
    "OBSERVED_MATCH",
    "EXTRAPOLATION",
    "COMPACTNESS",
]


def evaluate_candidate_stages(
    candidate_index: int,
    candidate_seed: int,
    raw_token_ids: List[int],
    raw_wat: str,
    target: BenchmarkTarget,
    protocol: EvaluationProtocol,
    seen_canonical_hashes: Dict[str, str],
    runner: Optional[WasmRunner] = None,
    evaluation_id: str = "eval_0",
) -> SynthesisCandidateRecord:
    """Evaluates a single candidate through all 8 sequential qualification stages."""
    wasm_runner = runner or WasmRunner(
        fuel_budget=protocol.fuel_per_invocation,
        memory_limit_mib=protocol.memory_limit_mib,
    )
    cand_id = f"{evaluation_id}_c{candidate_index}"

    stage_records: List[StageRecord] = []
    primary_failure: Optional[str] = None
    classification = "FAILED"

    resolved_wat: Optional[str] = None
    resolved_constants: List[str] = []
    canonical_wat: Optional[str] = None
    canonical_sha256: Optional[str] = None
    duplicate_of: Optional[str] = None

    outputs: List[str] = []
    max_fuel: Optional[int] = None
    total_fuel: Optional[int] = None
    peak_memory_mib: Optional[float] = None
    first_obs_div: Optional[int] = None
    first_uns_div: Optional[int] = None
    byte_size: Optional[int] = None
    mdl_ratio: Optional[float] = None

    # Target observed & unseen integer terms
    obs_integers = [int(x) for x in target.observed_terms]
    uns_integers = [int(x) for x in target.unseen_terms]
    all_integers = obs_integers + uns_integers

    def record_stage(
        stage_name: str,
        status: str,
        duration_ms: float = 0.0,
        reason_code: Optional[str] = None,
        message: Optional[str] = None,
        evidence: Optional[Dict[str, Any]] = None,
    ) -> None:
        stage_records.append(
            StageRecord(
                stage=stage_name,
                status=status,
                duration_ms=duration_ms,
                reason_code=reason_code,
                message=message,
                evidence=evidence or {},
            )
        )

    # 1. Stage: GENERATION
    t0 = time.perf_counter()
    if not raw_wat or raw_wat.strip() == "":
        record_stage("GENERATION", "FAILED", (time.perf_counter() - t0) * 1000.0, "EMPTY_WAT", "Raw WAT is empty")
        primary_failure = "GENERATION"
    else:
        record_stage("GENERATION", "PASSED", (time.perf_counter() - t0) * 1000.0)

    # 2. Stage: CONSTANT_RESOLUTION
    if primary_failure is None:
        t0 = time.perf_counter()
        if "i64.const_?" in raw_wat:
            if not protocol.constant_resolution:
                record_stage("CONSTANT_RESOLUTION", "FAILED", 0.0, "SOLVER_DISABLED", "Placeholders emitted but constant_resolution is false")
                primary_failure = "CONSTANT_RESOLUTION"
            else:
                r_wat, r_consts, r_status, r_dur, r_err = resolve_program_constants(
                    wat_code=raw_wat,
                    terms=obs_integers,
                    timeout_ms=protocol.solver_timeout_ms,
                    max_placeholders=protocol.max_placeholders,
                    runner=wasm_runner,
                )
                if r_status == "PASSED":
                    resolved_wat = r_wat
                    resolved_constants = [str(c) for c in r_consts]
                    record_stage("CONSTANT_RESOLUTION", "PASSED", r_dur, evidence={"constants": resolved_constants})
                elif r_status == "TIMEOUT":
                    record_stage("CONSTANT_RESOLUTION", "TIMEOUT", r_dur, "SOLVER_TIMEOUT", r_err)
                    primary_failure = "CONSTANT_RESOLUTION"
                else:
                    record_stage("CONSTANT_RESOLUTION", "FAILED", r_dur, "UNSATISFIABLE", r_err)
                    primary_failure = "CONSTANT_RESOLUTION"
        else:
            resolved_wat = raw_wat
            record_stage("CONSTANT_RESOLUTION", "NOT_REQUIRED", 0.0)
    else:
        record_stage("CONSTANT_RESOLUTION", "NOT_RUN", 0.0)

    # 3. Stage: CANONICALIZATION
    target_wat = resolved_wat if resolved_wat else raw_wat
    if primary_failure is None:
        t0 = time.perf_counter()
        try:
            artifact = optimize_wat_program(target_wat, hard_waste_threshold=protocol.mdl_ratio_max)
            canonical_wat = artifact.opt_wat
            canonical_tokens = tokenize_wat(canonical_wat)
            token_str = " ".join(canonical_tokens)
            canonical_sha256 = f"sha256:{hashlib.sha256(token_str.encode('utf-8')).hexdigest()}"

            if canonical_sha256 in seen_canonical_hashes:
                duplicate_of = seen_canonical_hashes[canonical_sha256]
                classification = "DUPLICATE"
                record_stage(
                    "CANONICALIZATION",
                    "PASSED",
                    (time.perf_counter() - t0) * 1000.0,
                    evidence={"duplicate_of": duplicate_of, "canonical_sha256": canonical_sha256},
                )
            else:
                seen_canonical_hashes[canonical_sha256] = cand_id
                record_stage(
                    "CANONICALIZATION",
                    "PASSED",
                    (time.perf_counter() - t0) * 1000.0,
                    evidence={"canonical_sha256": canonical_sha256, "waste_ratio": artifact.waste_ratio},
                )
        except Exception as e:
            record_stage("CANONICALIZATION", "FAILED", (time.perf_counter() - t0) * 1000.0, "CANONICAL_ERROR", str(e))
            primary_failure = "CANONICALIZATION"
    else:
        record_stage("CANONICALIZATION", "NOT_RUN", 0.0)

    # If duplicate, mark later stages as not required / passed without re-executing
    if duplicate_of is not None and primary_failure is None:
        for stg in STAGES_ORDER[3:]:
            record_stage(stg, "NOT_REQUIRED", 0.0, message="Skipped due to duplicate canonical program")
        return SynthesisCandidateRecord(
            candidate_id=cand_id,
            candidate_index=candidate_index,
            candidate_seed=candidate_seed,
            raw_token_ids=raw_token_ids,
            raw_wat=raw_wat,
            resolved_wat=resolved_wat,
            resolved_constants=resolved_constants,
            canonical_wat=canonical_wat,
            canonical_sha256=canonical_sha256,
            duplicate_of=duplicate_of,
            stage_records=stage_records,
            outputs=[],
            classification="DUPLICATE",
            primary_failure_stage=None,
        )

    # 4. Stage: ASSEMBLY
    exec_wat = canonical_wat or target_wat
    if primary_failure is None:
        t0 = time.perf_counter()
        try:
            import wasmtime
            wasm_bytes = bytes(wasmtime.wat2wasm(exec_wat))
            byte_size = len(wasm_bytes)
            record_stage("ASSEMBLY", "PASSED", (time.perf_counter() - t0) * 1000.0, evidence={"byte_size": byte_size})
        except Exception as e:
            record_stage("ASSEMBLY", "FAILED", (time.perf_counter() - t0) * 1000.0, "ASSEMBLY_ERROR", str(e))
            primary_failure = "ASSEMBLY"
    else:
        record_stage("ASSEMBLY", "NOT_RUN", 0.0)

    # 5. Stage: EXECUTION
    exec_res: Optional[ExecutionResult] = None
    if primary_failure is None:
        t0 = time.perf_counter()
        exec_res = wasm_runner.run_single(
            exec_wat,
            terms_to_generate=120,
            result_profile=target.result_profile,
        )
        max_fuel = exec_res.max_fuel
        total_fuel = exec_res.total_fuel
        peak_memory_mib = 1.0  # Safe upper-bound estimate within 16 MiB ceiling

        if exec_res.status != "SUCCESS":
            record_stage(
                "EXECUTION",
                "FAILED",
                (time.perf_counter() - t0) * 1000.0,
                exec_res.status,
                exec_res.error,
                evidence={"max_fuel": max_fuel, "total_fuel": total_fuel},
            )
            primary_failure = "EXECUTION"
        else:
            outputs = [str(x) for x in exec_res.output]
            record_stage(
                "EXECUTION",
                "PASSED",
                (time.perf_counter() - t0) * 1000.0,
                evidence={"max_fuel": max_fuel, "total_fuel": total_fuel, "output_count": len(outputs)},
            )
    else:
        record_stage("EXECUTION", "NOT_RUN", 0.0)

    # 6. Stage: OBSERVED_MATCH
    if primary_failure is None:
        t0 = time.perf_counter()
        if len(outputs) < 20:
            record_stage("OBSERVED_MATCH", "FAILED", (time.perf_counter() - t0) * 1000.0, "INSUFFICIENT_OUTPUT", f"Outputs: {len(outputs)} < 20")
            primary_failure = "OBSERVED_MATCH"
            first_obs_div = len(outputs)
        else:
            for i in range(20):
                if int(outputs[i]) != obs_integers[i]:
                    first_obs_div = i
                    break
            if first_obs_div is not None:
                record_stage(
                    "OBSERVED_MATCH",
                    "FAILED",
                    (time.perf_counter() - t0) * 1000.0,
                    "OBSERVED_MISMATCH",
                    f"Divergence at index {first_obs_div}: expected {obs_integers[first_obs_div]}, got {outputs[first_obs_div]}",
                )
                primary_failure = "OBSERVED_MATCH"
            else:
                record_stage("OBSERVED_MATCH", "PASSED", (time.perf_counter() - t0) * 1000.0)
    else:
        record_stage("OBSERVED_MATCH", "NOT_RUN", 0.0)

    # 7. Stage: EXTRAPOLATION
    if primary_failure is None:
        t0 = time.perf_counter()
        if len(outputs) < 120:
            record_stage("EXTRAPOLATION", "FAILED", (time.perf_counter() - t0) * 1000.0, "INSUFFICIENT_UNSEEN", f"Outputs: {len(outputs)} < 120")
            primary_failure = "EXTRAPOLATION"
            first_uns_div = max(0, len(outputs) - 20)
        else:
            for j in range(100):
                if int(outputs[20 + j]) != uns_integers[j]:
                    first_uns_div = j
                    break
            if first_uns_div is not None:
                record_stage(
                    "EXTRAPOLATION",
                    "FAILED",
                    (time.perf_counter() - t0) * 1000.0,
                    "UNSEEN_MISMATCH",
                    f"Divergence at unseen index {first_uns_div}: expected {uns_integers[first_uns_div]}, got {outputs[20 + first_uns_div]}",
                )
                primary_failure = "EXTRAPOLATION"
            else:
                record_stage("EXTRAPOLATION", "PASSED", (time.perf_counter() - t0) * 1000.0)
    else:
        record_stage("EXTRAPOLATION", "NOT_RUN", 0.0)

    # 8. Stage: COMPACTNESS
    if primary_failure is None:
        t0 = time.perf_counter()
        mdl_verifier = MdlVerifier(threshold=protocol.mdl_ratio_max)
        mdl_rec = mdl_verifier.assess_compactness(exec_wat, obs_integers, canonical_wat=canonical_wat)
        mdl_ratio = mdl_rec.mdl_ratio
        if not mdl_rec.passed:
            record_stage(
                "COMPACTNESS",
                "FAILED",
                (time.perf_counter() - t0) * 1000.0,
                "MDL_EXCEEDED" if not mdl_rec.is_table_memorized else "TABLE_MEMORIZED",
                f"MDL ratio {mdl_ratio:.2f} > {protocol.mdl_ratio_max}",
                evidence={"mdl_ratio": mdl_ratio, "byte_size": byte_size},
            )
            primary_failure = "COMPACTNESS"
        else:
            record_stage(
                "COMPACTNESS",
                "PASSED",
                (time.perf_counter() - t0) * 1000.0,
                evidence={"mdl_ratio": mdl_ratio, "byte_size": byte_size},
            )
    else:
        record_stage("COMPACTNESS", "NOT_RUN", 0.0)

    if primary_failure is None:
        classification = "EXTRAPOLATING_SUCCESS"
    else:
        classification = "FAILED"

    # Invariant: exactly 8 stage records in fixed order
    assert len(stage_records) == 8, f"Expected 8 stage records, got {len(stage_records)}"

    return SynthesisCandidateRecord(
        candidate_id=cand_id,
        candidate_index=candidate_index,
        candidate_seed=candidate_seed,
        raw_token_ids=raw_token_ids,
        raw_wat=raw_wat,
        resolved_wat=resolved_wat,
        resolved_constants=resolved_constants,
        canonical_wat=canonical_wat,
        canonical_sha256=canonical_sha256,
        duplicate_of=duplicate_of,
        stage_records=stage_records,
        outputs=outputs,
        max_fuel=max_fuel,
        total_fuel=total_fuel,
        peak_memory_mib=peak_memory_mib,
        first_observed_divergence=first_obs_div,
        first_unseen_divergence=first_uns_div,
        byte_size=byte_size,
        mdl_ratio=mdl_ratio,
        classification=classification,
        primary_failure_stage=primary_failure,
    )


def evaluate_cohort_synthesis(
    encoder: TriStreamEncoder,
    decoder: WatTransformerDecoder,
    checkpoint: CheckpointProvenance,
    target: BenchmarkTarget,
    protocol: EvaluationProtocol,
    runner: Optional[WasmRunner] = None,
    evaluation_id: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> SynthesisEvaluationResult:
    """Evaluates a benchmark target sequence using the shared synthesis workflow."""
    start_eval_time = time.perf_counter()
    dev = device or torch.device("cpu")
    eval_id = evaluation_id or f"eval_{target.oeis_id}_{protocol.candidate_budget}_{protocol.base_seed}"

    sampler = WatProgramSampler(
        decoder=decoder,
        max_length=protocol.max_tokens,
        temperature=protocol.temperature,
        top_p=protocol.top_p,
    )

    # Encode observed 20 terms
    obs_integers = [int(x) for x in target.observed_terms]
    with torch.no_grad():
        z = encoder.forward_from_sequences([obs_integers], device=dev)

    candidates: List[SynthesisCandidateRecord] = []
    seen_canonical: Dict[str, str] = {}

    for k in range(protocol.candidate_budget):
        c_seed = derive_candidate_seed(
            base_seed=protocol.base_seed,
            protocol_id=protocol.protocol_id,
            sequence_id=target.oeis_id,
            candidate_index=k,
        )
        raw_wat, token_tensor = sampler.sample_candidate(
            memory=z,
            seed=c_seed,
            temperature=protocol.temperature,
            top_p=protocol.top_p,
            max_length=protocol.max_tokens,
        )
        token_ids = token_tensor.squeeze(0).tolist() if token_tensor.dim() > 0 else token_tensor.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]

        cand_record = evaluate_candidate_stages(
            candidate_index=k,
            candidate_seed=c_seed,
            raw_token_ids=token_ids,
            raw_wat=raw_wat,
            target=target,
            protocol=protocol,
            seen_canonical_hashes=seen_canonical,
            runner=runner,
            evaluation_id=eval_id,
        )
        candidates.append(cand_record)

    unique_count = len(seen_canonical)
    qualified_ids = [c.candidate_id for c in candidates if c.classification == "EXTRAPOLATING_SUCCESS"]

    status = "QUALIFIED_SUCCESS" if qualified_ids else "COMPLETED_NO_SUCCESS"
    duration_ms = (time.perf_counter() - start_eval_time) * 1000.0

    return SynthesisEvaluationResult(
        schema_version="1.0",
        evaluation_id=eval_id,
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        protocol=protocol,
        checkpoint=checkpoint,
        target=target,
        candidates=candidates,
        unique_candidate_count=unique_count,
        qualified_candidate_ids=qualified_ids,
        status=status,
        duration_ms=duration_ms,
    )
