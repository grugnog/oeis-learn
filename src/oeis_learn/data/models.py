"""Domain entity models and OEIS data parsers."""

from __future__ import annotations

import datetime
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

OEIS_ID_PATTERN = re.compile(r"^A\d{6}$")


@dataclass
class SequenceRecord:
    """Represents an OEIS integer sequence entry."""

    oeis_id: str
    name: str
    terms: List[int]
    tags: List[str] = field(default_factory=list)
    curriculum_stage: int = 1
    joeis_class: Optional[str] = None
    generating_formula: Optional[str] = None
    lz_complexity: float = 0.0

    def __post_init__(self) -> None:
        if not OEIS_ID_PATTERN.match(self.oeis_id):
            raise ValueError(f"Invalid OEIS ID: {self.oeis_id}")
        if not (1 <= self.curriculum_stage <= 5):
            raise ValueError(f"Curriculum stage must be between 1 and 5, got {self.curriculum_stage}")
        self.tags = [t.strip().lower() for t in self.tags if t.strip()]

    @property
    def term_count(self) -> int:
        return len(self.terms)

    @property
    def terms_json(self) -> str:
        return json.dumps(self.terms)

    @property
    def tags_str(self) -> str:
        return ",".join(self.tags)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["term_count"] = self.term_count
        d["terms_json"] = self.terms_json
        d["tags"] = self.tags_str
        return d


@dataclass
class ExecutionResult:
    """Output returned from sandboxed WASM evaluation."""

    status: str
    consumed_fuel: int
    output: List[int]
    error: Optional[str] = None
    divergence_step: Optional[int] = None
    max_fuel: Optional[int] = None
    total_fuel: Optional[int] = None
    wide_output: List[List[int]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.max_fuel is None:
            self.max_fuel = self.consumed_fuel
        if self.total_fuel is None:
            self.total_fuel = self.consumed_fuel

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateProgram:
    """Generated algorithmic candidate in WebAssembly Text format."""

    program_id: str
    prompt_oeis_id: str
    wat_code: str
    byte_size: int = 0
    mdl_ratio: float = 0.0
    extrapolation_passed: bool = False


@dataclass
class CurriculumProgress:
    """Maintains state for the 5-stage automated curriculum scheduler."""

    active_stage: int = 1
    rolling_pass_rates: Dict[str, float] = field(default_factory=dict)
    stage_competence: float = 0.0
    coverage_min: float = 0.0
    epoch_variance: float = 0.0
    graduated_stages: List[int] = field(default_factory=list)


@dataclass
class LatentDiscoveryCandidate:
    """Represents an algebraic conjecture discovered in latent representation space."""

    candidate_id: str
    relation_type: str
    sequences: Tuple[str, ...]
    vector_distance: float
    pslq_vector: Optional[List[int]] = None
    pslq_confidence_ratio: Optional[float] = None
    symbolic_proof: Optional[str] = None
    status: str = "CONJECTURED"


@dataclass
class SyntheticDemonstrationPair:
    """Represents a forward-generated synthetic training instance mapping an integer sequence to a valid WebAssembly program."""

    sample_id: str
    family: str
    terms: List[int]
    wat_code: str
    byte_size: int = 0
    lz_complexity: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SyntheticDemonstrationPair:
        return cls(
            sample_id=data["sample_id"],
            family=data["family"],
            terms=data["terms"],
            wat_code=data["wat_code"],
            byte_size=data.get("byte_size", 0),
            lz_complexity=data.get("lz_complexity", 0.0),
            metadata=data.get("metadata", {}),
        )


@dataclass
class EliteReplayBufferEntry:
    """Represents a verified canonical reference solution stored in the elite demonstration buffer."""

    oeis_id: str
    terms: List[int]
    wat_code: str
    byte_size: int = 0
    extrapolation_passed: bool = True
    mdl_ratio: float = 1.0
    source: str = "SYNTHETIC_GENERATOR"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DiagnosticTelemetryRecord:
    """Captures point-in-time RL optimization dynamics and early warning indicators."""

    epoch: int
    step: int
    policy_entropy: float
    reward_variance: float
    advantage_collapse_rate: float
    compiler_trap_rate: float
    avg_prefix_length: float
    oracle_ppl: Optional[float] = None
    active_stage: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProgressiveTierResult:
    """Records the execution outcome and gate verification status for the 5-tier testing hierarchy."""

    tier: int
    tier_name: str
    latency_seconds: float
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProgressiveValidationReport:
    """Full pre-flight progressive validation report across all executed tiers."""

    harness_version: str = "2.0.0"
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    overall_passed: bool = True
    max_authorized_tier: int = 0
    tier_results: List[ProgressiveTierResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "harness_version": self.harness_version,
            "timestamp": self.timestamp,
            "overall_passed": self.overall_passed,
            "max_authorized_tier": self.max_authorized_tier,
            "tier_results": [r.to_dict() for r in self.tier_results],
        }


@dataclass
class CompositeRewardBreakdown:
    """Decomposes intermediate and verifiable reward signals during policy gradient rollouts."""

    r_comp: float
    r_prefix: float
    r_dist: float
    r_exact: float
    r_total: float
    divergence_step: Optional[int] = None
    divergence_token_idx: Optional[int] = None
    is_non_trivial: bool = True
    output_variance: float = 0.0
    input_sensitivity: float = 0.0
    mi_score: float = 0.0
    potential_shaping: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NonTrivialityEvaluation:
    """Captures empirical output dynamics, input sensitivity, and non-triviality gating status."""

    output_variance: float
    target_variance: float
    input_sensitivity: float
    has_param_binding: bool = True
    mutual_information_score: float = 0.0
    is_non_trivial: bool = True
    penalty: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CoTrainingBatch:
    """Encapsulates an online training batch pairing RL exploratory rollouts with teacher-forced SFT demonstrations."""

    prompt_records: List[SequenceRecord]
    rollout_tokens: Any
    sft_demonstrations: List[SyntheticDemonstrationPair]
    ref_log_probs: Optional[Any] = None
    beta_sft: float = 0.20
    beta_kl: float = 0.05


@dataclass
class FineGrainedAttributionSpan:
    """Stores localized credit assignment spans, mapping failure mode, divergence step, and token masks."""

    failure_mode: str
    divergence_step: Optional[int] = None
    causal_token_start: int = 0
    causal_token_end: int = 0
    token_advantage_mask: List[float] = field(default_factory=list)
    executed_token_mask: List[bool] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PotentialState:
    """Tracks potential-based shaping variables across incremental AST decoding states."""

    step: int = 0
    structural_phase: str = "BODY"
    phi_comp: float = 0.0
    phi_bind: float = 0.0
    total_potential: float = 0.0
    shaping_difference: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LexicaseSelectionBatch:
    """Tracks per-test-case rollout evaluations across randomized sequence indices for down-sampled lexicase filtering."""

    prompt_id: str
    test_case_indices: List[int] = field(default_factory=list)
    candidate_errors: Dict[int, List[float]] = field(default_factory=dict)
    surviving_candidates: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =========================================================================
# Phase 4 Decoupled Grounding & SYMPLE Multi-Task Engine Domain Entities
# =========================================================================


@dataclass
class ASTSkeleton:
    """Represents an ungrounded WebAssembly program structure containing placeholder tokens."""

    raw_wat: str
    placeholder_count: int
    is_linear: bool = True
    placeholder_indices: List[int] = field(default_factory=list)
    basis_signatures: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConstantSolverResult:
    """Captures the output of Diophantine or SMT constant solving."""

    solver_type: str
    constants: Optional[List[int]] = None
    solve_duration_ms: float = 0.0
    is_sat: bool = False
    grounded_wat: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CanonicalProgramArtifact:
    """Stores the output of the optimizing compiler pass (wasm-opt)."""

    raw_wat: str
    opt_wat: str
    raw_token_count: int
    opt_token_count: int
    waste_ratio: float = 0.0
    passes_applied: List[str] = field(default_factory=lambda: ["--vacuum", "--dce", "--remove-unused-locals"])
    is_waste_exceeded: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ParsimonyRewardRecord:
    """Encapsulates parsimony-adjusted rewards, continuous log-distance return, and lexicographical group ranking."""

    dense_return: float
    covariance_coef: float = 0.0
    parsimony_penalty: float = 0.0
    waste_penalty: float = 0.0
    cpp_reward: float = 0.0
    lexicographic_rank: int = 1
    ordinal_advantage: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SYMPLETaskState:
    """Tracks the curriculum state for a sequence in the 524-benchmark pool under the EXP3.S scheduler."""

    oeis_id: str
    pass_history: List[int] = field(default_factory=list)
    competence: float = 0.0
    competence_slope: float = 0.0
    bandit_weight: float = 1.0
    selection_prob: float = 0.0
    last_visited_step: int = 0
    dormancy: int = 0
    has_elite_solution: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EliteDemonstrationEntry:
    """Encapsulates a verified canonical program in the Elite Demonstration Buffer (EDB)."""

    oeis_id: str
    canonical_wat: str
    token_length: int
    fuel_consumed: int = 0
    ast_hash: str = ""
    discovery_step: int = 0
    mdl_score: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NormalizedLatentRecord:
    """Stores L2-normalized continuous embeddings and manifold discovery representations."""

    oeis_id: str
    raw_embedding: List[float] = field(default_factory=list)
    normalized_embedding: List[float] = field(default_factory=list)
    cluster_id: int = -1
    affine_slope_pred: float = 0.0
    geom_ratio_pred: float = 0.0
    discovered_relations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def parse_stripped_line(line: str) -> Optional[Tuple[str, List[int]]]:
    """Parses a line from the OEIS 'stripped' file (format: A000045 ,0,1,1,2,3,5,8,13,...)."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(",")
    if not parts:
        return None
    oeis_id = parts[0].strip()
    if not OEIS_ID_PATTERN.match(oeis_id):
        return None
    terms = []
    for p in parts[1:]:
        p = p.strip()
        if p:
            try:
                terms.append(int(p))
            except ValueError:
                break
    return oeis_id, terms


def parse_names_line(line: str) -> Optional[Tuple[str, str]]:
    """Parses a line from the OEIS 'names' file (format: A000045 Fibonacci numbers: F(n) = F(n-1) + F(n-2)...)."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(" ", 1)
    if len(parts) < 2:
        return None
    oeis_id, name = parts[0].strip(), parts[1].strip()
    if not OEIS_ID_PATTERN.match(oeis_id):
        return None
    return oeis_id, name


# =========================================================================
# Phase 5 Trustworthy Synthesis Readiness & Evidence Models
# =========================================================================


@dataclass(frozen=True)
class CheckpointIdentity:
    """Identifies an inference-compatible model checkpoint and constructor metadata."""

    format_version: str
    checkpoint_sha256: str
    producer_version: str
    epoch: int
    precision: str
    encoder_config: Dict[str, Any]
    decoder_config: Dict[str, Any]
    vocabulary_sha256: str
    source_checkpoint_sha256: Optional[str] = None
    runtime_environment: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Alias
CheckpointProvenance = CheckpointIdentity


@dataclass(frozen=True)
class BenchmarkSource:
    """Describes origin and revision of authoritative sequence truth."""

    name: str
    revision: str
    retrieved_at: str
    content_sha256: str
    license_notice: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkTarget:
    """Frozen sequence target with 20 observed and 100 disjoint unseen terms."""

    oeis_id: str
    name: str
    offset: int
    family: str
    curriculum_stage: int
    observed_terms: List[str]
    unseen_terms: List[str]
    result_profile: str
    terms_sha256: str
    term_fingerprint: str
    formula_definition_id: Optional[str] = None
    program_fingerprints: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


BenchmarkSequence = BenchmarkTarget


@dataclass(frozen=True)
class BenchmarkCohort:
    """Versioned collection of benchmark targets and manifest exclusions."""

    schema_version: str
    cohort_id: str
    manifest_sha256: str
    source: Dict[str, Any]
    observed_horizon: int
    unseen_horizon: int
    targets: List[BenchmarkTarget]
    exclusions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cohort_id": self.cohort_id,
            "manifest_sha256": self.manifest_sha256,
            "source": self.source if isinstance(self.source, dict) else self.source.to_dict(),
            "observed_horizon": self.observed_horizon,
            "unseen_horizon": self.unseen_horizon,
            "targets": [t.to_dict() for t in self.targets],
            "exclusions": self.exclusions,
        }


BenchmarkManifest = BenchmarkCohort


@dataclass(frozen=True)
class StageRecord:
    """Evidence for one candidate processing stage."""

    stage: str
    status: str
    duration_ms: float = 0.0
    reason_code: Optional[str] = None
    message: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


StageEvidence = StageRecord


@dataclass
class SynthesisCandidateRecord:
    """Complete lineage for one requested candidate index."""

    candidate_id: str
    candidate_index: int
    candidate_seed: int
    raw_token_ids: List[int]
    raw_wat: str
    resolved_wat: Optional[str] = None
    resolved_constants: List[str] = field(default_factory=list)
    canonical_wat: Optional[str] = None
    canonical_sha256: Optional[str] = None
    duplicate_of: Optional[str] = None
    stage_records: List[StageRecord] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    max_fuel: Optional[int] = None
    total_fuel: Optional[int] = None
    peak_memory_mib: Optional[float] = None
    first_observed_divergence: Optional[int] = None
    first_unseen_divergence: Optional[int] = None
    byte_size: Optional[int] = None
    mdl_ratio: Optional[float] = None
    classification: str = "FAILED"
    primary_failure_stage: Optional[str] = None
    secondary_diagnostics: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_index": self.candidate_index,
            "candidate_seed": self.candidate_seed,
            "raw_token_ids": self.raw_token_ids,
            "raw_wat": self.raw_wat,
            "resolved_wat": self.resolved_wat,
            "resolved_constants": self.resolved_constants,
            "canonical_wat": self.canonical_wat,
            "canonical_sha256": self.canonical_sha256,
            "duplicate_of": self.duplicate_of,
            "stage_records": [s.to_dict() for s in self.stage_records],
            "outputs": self.outputs,
            "max_fuel": self.max_fuel,
            "total_fuel": self.total_fuel,
            "peak_memory_mib": self.peak_memory_mib,
            "first_observed_divergence": self.first_observed_divergence,
            "first_unseen_divergence": self.first_unseen_divergence,
            "byte_size": self.byte_size,
            "mdl_ratio": self.mdl_ratio,
            "classification": self.classification,
            "primary_failure_stage": self.primary_failure_stage,
            "secondary_diagnostics": self.secondary_diagnostics,
        }


@dataclass
class SynthesisEvaluationResult:
    """One target's candidate set and aggregate synthesis outcome."""

    schema_version: str
    evaluation_id: str
    created_at: str
    protocol: Any
    checkpoint: Any
    target: Any
    candidates: List[SynthesisCandidateRecord]
    unique_candidate_count: int
    qualified_candidate_ids: List[str]
    status: str
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evaluation_id": self.evaluation_id,
            "created_at": self.created_at,
            "protocol": self.protocol.to_dict() if hasattr(self.protocol, "to_dict") else self.protocol,
            "checkpoint": self.checkpoint.to_dict() if hasattr(self.checkpoint, "to_dict") else self.checkpoint,
            "target": {
                "cohort_id": getattr(self.target, "cohort_id", "trustworthy_synthesis_v1"),
                "oeis_id": getattr(self.target, "oeis_id", ""),
                "offset": getattr(self.target, "offset", 0),
                "terms_sha256": getattr(self.target, "terms_sha256", ""),
                "result_profile": getattr(self.target, "result_profile", "i64_scalar_v1"),
            } if hasattr(self.target, "oeis_id") else self.target,
            "candidates": [c.to_dict() for c in self.candidates],
            "unique_candidate_count": self.unique_candidate_count,
            "qualified_candidate_ids": self.qualified_candidate_ids,
            "status": self.status,
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class ReadinessThreshold:
    """Threshold specification for a readiness gate."""

    gate_id: str
    metric: str
    comparator: str
    threshold: float
    unit: str
    source: str
    non_relaxable: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReadinessGateResult:
    """Outcome and diagnostics for one readiness gate."""

    gate_id: str
    measured_value: float
    threshold: ReadinessThreshold
    passed: bool
    evaluated_at: str
    evidence: List[Dict[str, str]]
    diagnostics: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_id": self.gate_id,
            "measured_value": self.measured_value,
            "threshold": self.threshold.to_dict(),
            "passed": self.passed,
            "evaluated_at": self.evaluated_at,
            "evidence": self.evidence,
            "diagnostics": self.diagnostics,
        }


@dataclass(frozen=True)
class OverrideRecord:
    """Audit record for a diagnostic override."""

    override_id: str
    operator: str
    created_at: str
    reason: str
    diagnostic_intent: str
    failed_gate_ids: List[str]
    policy_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReadinessPolicy:
    """Governs criteria for qualification and authorization."""

    schema_version: str
    policy_id: str
    name: str
    thresholds: List[ReadinessThreshold]
    required_experiment_ids: List[str]
    required_artifacts: List[str]
    native_evaluator_required: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "policy_id": self.policy_id,
            "name": self.name,
            "thresholds": [t.to_dict() for t in self.thresholds],
            "required_experiment_ids": self.required_experiment_ids,
            "required_artifacts": self.required_artifacts,
            "native_evaluator_required": self.native_evaluator_required,
        }


@dataclass(frozen=True)
class ReadinessReport:
    """Complete readiness evaluation report."""

    schema_version: str
    report_id: str
    run_id: str
    created_at: str
    policy: ReadinessPolicy
    gate_results: List[ReadinessGateResult]
    overall_passed: bool
    override: Optional[OverrideRecord]
    qualification_state: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "report_id": self.report_id,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "policy": self.policy.to_dict(),
            "gate_results": [g.to_dict() for g in self.gate_results],
            "overall_passed": self.overall_passed,
            "override": self.override.to_dict() if self.override else None,
            "qualification_state": self.qualification_state,
        }


@dataclass(frozen=True)
class ExperimentVariant:
    """One tested variation in a paired experiment."""

    variant_id: str
    changed_factor: str
    factor_value: Any
    candidate_cache_id: Optional[str]
    active_rollout_budget: int
    replay_budget: int
    protocol_overrides: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentOutcome:
    """Execution outcome for one seed/variant cell."""

    variant_id: str
    seed: int
    status: str
    started_at: str
    finished_at: Optional[str]
    wall_hours: float
    evaluation_count: int
    metrics: Dict[str, Any]
    artifact_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentManifest:
    """Controls reproducible paired ablations."""

    schema_version: str
    experiment_id: str
    experiment_type: str
    status: str
    created_at: str
    checkpoint_sha256: str
    benchmark_manifest_sha256: str
    seeds: List[int]
    variants: List[ExperimentVariant]
    invariants: Dict[str, Any]
    decision_schedule: List[int]
    max_trial_hours: float
    max_total_hours: float
    outcomes: List[ExperimentOutcome] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "experiment_type": self.experiment_type,
            "status": self.status,
            "created_at": self.created_at,
            "checkpoint_sha256": self.checkpoint_sha256,
            "benchmark_manifest_sha256": self.benchmark_manifest_sha256,
            "seeds": self.seeds,
            "variants": [v.to_dict() for v in self.variants],
            "invariants": self.invariants,
            "decision_schedule": self.decision_schedule,
            "max_trial_hours": self.max_trial_hours,
            "max_total_hours": self.max_total_hours,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


@dataclass
class TaskTrainingState:
    """Tracks per-task learning progress in the training pool."""

    oeis_id: str
    pass_history: List[int]
    competence: float
    competence_slope: float
    bandit_weight: float
    selection_probability: float
    allocated_rollouts: int
    last_active_step: int
    last_replay_step: int
    has_verified_elite: bool
    retention_status: str = "UNKNOWN"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RecurrenceFrame:
    """Tracks state and rotation invariants for recurrence loops."""

    state_locals: List[str]
    next_locals: List[str]
    progress_local: str
    required_commits: Set[str]
    completed_commits: Set[str] = field(default_factory=set)
    progress_advanced: bool = False
    phase: str = "GUARD"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_locals": self.state_locals,
            "next_locals": self.next_locals,
            "progress_local": self.progress_local,
            "required_commits": list(self.required_commits),
            "completed_commits": list(self.completed_commits),
            "progress_advanced": self.progress_advanced,
            "phase": self.phase,
        }


RecurrenceTransition = RecurrenceFrame


@dataclass(frozen=True)
class SequenceRef:
    """Identifies a sequence with optional affine index scaling and shifting."""

    oeis_id: str
    index_scale: int = 1
    index_shift: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalRelation:
    """Primitive canonical linear relation across sequence references."""

    relation_type: str
    operands: List[SequenceRef]
    coefficients: List[str]
    canonical_expression: str
    claim_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "relation_type": self.relation_type,
            "operands": [op.to_dict() for op in self.operands],
            "coefficients": self.coefficients,
            "canonical_expression": self.canonical_expression,
            "claim_id": self.claim_id,
        }


@dataclass(frozen=True)
class DiscoveryClaim:
    """Lifecycle record for a mathematical relation claim."""

    schema_version: str
    relation: CanonicalRelation
    status: str
    latent_evidence: List[Dict[str, Any]]
    numerical_evidence: Optional[Dict[str, Any]] = None
    symbolic_evidence: Optional[Dict[str, Any]] = None
    rejection: Optional[Dict[str, Any]] = None
    status_history: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "relation": self.relation.to_dict(),
            "status": self.status,
            "latent_evidence": self.latent_evidence,
            "numerical_evidence": self.numerical_evidence,
            "symbolic_evidence": self.symbolic_evidence,
            "rejection": self.rejection,
            "status_history": self.status_history,
        }

