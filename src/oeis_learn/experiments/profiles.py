"""Frozen foundation/v1 profiles and exhaustive inventories.

Normative sources: ``specs/007-experiment-foundation/contracts/execution.md``,
``evaluation.md`` and ``data-model.md``. Profile IDs here are frozen defaults
for 007; changing any of them requires a new profile identity and a new run.

The allowed-operator inventory is exactly the baseline
``INSTRUCTION_TOKENS`` at commit ``aa2c8c4579dd74fdd8114ec9456960a98fcf500e``,
excluding ``i64.const_?`` and the pseudo-token ``result_i64_x4``. It is stored
explicitly here (not re-derived from the legacy grammar) so the hashed profile
never changes when a dependency is upgraded; an upgrade may not admit an
opcode implicitly.
"""

from __future__ import annotations

from typing import Dict, Tuple

# ---------------------------------------------------------------------------
# Frozen profile identities
# ---------------------------------------------------------------------------

LANGUAGE_PROFILE_ID = "wat_i256_checked_v1"
CODEC_PROFILE_ID = "wat_body_decimal_v1"
RESOURCE_PROFILE_ID = "ryzen_foundation_v1"
EVALUATION_PROFILE_ID = "prefix20_total100_v1"

# Learning track / objective / initialization (exact, frozen)
TRACK = "strict_generic"
OBJECTIVE = "sft"
INITIALIZATION = "random"

# Horizon: (visible prefix, full evaluation)
VISIBLE_HORIZON = 20
FULL_HORIZON = 100
POLICY = "prefix_rebased_zero"

# Baseline inventory commit the allowed operator set is pinned to.
BASELINE_COMMIT = "aa2c8c4579dd74fdd8114ec9456960a98fcf500e"

# Opcodes explicitly excluded from the baseline inventory.
EXCLUDED_OPERATORS = ("i64.const_?", "result_i64_x4")


# ---------------------------------------------------------------------------
# Exhaustive allowed-operator inventory (frozen)
# ---------------------------------------------------------------------------

ALLOWED_OPERATORS: Tuple[str, ...] = (
    "local.get",
    "local.set",
    "local.tee",
    "i32.const",
    "i64.const",
    "i64.add",
    "i64.sub",
    "i64.mul",
    "i64.div_s",
    "i64.div_u",
    "i64.rem_s",
    "i64.rem_u",
    "i64.and",
    "i64.or",
    "i64.xor",
    "i64.shl",
    "i64.shr_s",
    "i64.shr_u",
    "i64.eqz",
    "i64.eq",
    "i64.ne",
    "i64.lt_s",
    "i64.gt_s",
    "i64.le_s",
    "i64.ge_s",
    "i32.add",
    "i32.sub",
    "i32.mul",
    "i32.div_s",
    "i32.div_u",
    "i32.rem_s",
    "i32.rem_u",
    "i32.and",
    "i32.or",
    "i32.xor",
    "i32.shl",
    "i32.shr_s",
    "i32.shr_u",
    "i32.ge_s",
    "i32.lt_s",
    "i32.gt_s",
    "i32.le_s",
    "i32.eq",
    "i32.ne",
    "i32.eqz",
    "i32.wrap_i64",
    "i64.extend_i32_s",
    "i64.extend_i32_u",
    "drop",
    "nop",
    "unreachable",
    "return",
    "br",
    "br_if",
    "block",
    "loop",
    "if",
    "then",
    "else",
    "end",
    "i256.add",
    "i256.sub",
    "i256.mul_scalar",
    "i256.const",
    "i256.zero",
)

# The five checked wide macros with their exact stack effects.
MACROS: Dict[str, Tuple[int, int]] = {
    "i256.const": (0, 4),  # pushes four low-to-high limbs
    "i256.zero": (0, 4),
    "i256.add": (8, 4),
    "i256.sub": (8, 4),
    "i256.mul_scalar": (5, 4),
}


def _validate_inventory_against_baseline() -> None:
    """Guard against accidental drift from the pinned baseline inventory."""
    try:
        from oeis_learn.decoder.wat_grammar import INSTRUCTION_TOKENS  # legacy
    except Exception:  # pragma: no cover - import path may be unavailable
        return
    baseline = [t for t in INSTRUCTION_TOKENS if t not in EXCLUDED_OPERATORS]
    if baseline != list(ALLOWED_OPERATORS):
        raise RuntimeError(
            "Frozen foundation opcode inventory drifted from the baseline "
            "INSTRUCTION_TOKENS (commit %s) minus %s. Refusing to admit an "
            "implicit opcode." % (BASELINE_COMMIT, EXCLUDED_OPERATORS)
        )


_validate_inventory_against_baseline()


# ---------------------------------------------------------------------------
# Fixed wrapper / numeric inventory
# ---------------------------------------------------------------------------

# The task-independent wrapper declares 32 i64 locals $v0..$v31 and 8 i32
# locals $c0..$c7, all zero-initialized; $n is the only input.
I64_LOCAL_NAMES: Tuple[str, ...] = tuple(f"$v{i}" for i in range(32))
I32_LOCAL_NAMES: Tuple[str, ...] = tuple(f"$c{i}" for i in range(8))
INPUT_LOCAL = "$n"

# Signed value ranges enforced by the language profile.
I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
I256_MIN, I256_MAX = -(1 << 255), (1 << 255) - 1

NUMERIC_RANGES: Dict[str, Tuple[int, int]] = {
    "i32": (I32_MIN, I32_MAX),
    "i64": (I64_MIN, I64_MAX),
    "i256": (I256_MIN, I256_MAX),
}


# ---------------------------------------------------------------------------
# Resource profile: ryzen_foundation_v1 (frozen)
# ---------------------------------------------------------------------------

RESOURCE_LIMITS: Dict[str, int] = {
    "fuel_per_term": 1_000_000,
    "fuel_aggregate": 50_000_000,
    "reference_steps_per_term": 1_000_000,
    "reference_steps_aggregate": 50_000_000,
    "candidate_deadline_s": 2,
    "worker_reclaim_s": 2,
    "worker_rss_limit_bytes": 1 << 30,
    "max_execution_workers": 8,
    "learner_cpu_threads": 4,
    "pending_requests": 32,
    "message_bytes": 1 << 20,
    "compiled_cache_bytes": 1 << 30,
    "learner_memory_bytes": 88 << 30,
    "worker_memory_bytes": 1 << 30,
    "gpu_alloc_bytes": 64 << 30,
    "gpu_device_fraction": 80,
    "host_reserve_bytes": 16 << 30,
    "run_artifacts_bytes": 200 << 30,
    "checkpoint_bytes": 40 << 30,
    "log_bytes": 2 << 30,
    "log_segment_bytes": 64 << 20,
    "pid_per_worker": 64,
    "pid_learner": 512,
    "attempts_default": 16,
    "prefix_phase_s": 20,
    "hidden_phase_s": 10,
}

# Evaluation grouping / split defaults (prefix20_total100_v1).
EVALUATION_LIMITS: Dict[str, int] = {
    "dev_count_default": 128,
    "final_count_default": 512,
    "shift_max": 20,
    "seed_default": 20260913,
    "split_seed": 20260913,
}
