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

from types import MappingProxyType

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

ALLOWED_OPERATORS: tuple[str, ...] = (
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
MACROS: dict[str, tuple[int, int]] = {
    "i256.const": (0, 4),  # pushes four low-to-high limbs
    "i256.zero": (0, 4),
    "i256.add": (8, 4),
    "i256.sub": (8, 4),
    "i256.mul_scalar": (5, 4),
}


# ---------------------------------------------------------------------------
# Fixed wrapper / numeric inventory
# ---------------------------------------------------------------------------

# The task-independent wrapper declares 32 i64 locals $v0..$v31 and 8 i32
# locals $c0..$c7, all zero-initialized; $n is the only input.
I64_LOCAL_NAMES: tuple[str, ...] = tuple(f"$v{i}" for i in range(32))
I32_LOCAL_NAMES: tuple[str, ...] = tuple(f"$c{i}" for i in range(8))
INPUT_LOCAL = "$n"

# Signed value ranges enforced by the language profile.
I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1
I64_MIN, I64_MAX = -(1 << 63), (1 << 63) - 1
I256_MIN, I256_MAX = -(1 << 255), (1 << 255) - 1

NUMERIC_RANGES: dict[str, tuple[int, int]] = {
    "i32": (I32_MIN, I32_MAX),
    "i64": (I64_MIN, I64_MAX),
    "i256": (I256_MIN, I256_MAX),
}


# ---------------------------------------------------------------------------
# Resource profile: ryzen_foundation_v1 (frozen)
# ---------------------------------------------------------------------------

RESOURCE_LIMITS: dict[str, int] = {
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
    "host_free_disk_bytes": 100 << 30,
    "learner_swap_limit_bytes": 88 << 30,
    "aggregate_container_cap_bytes": 96 << 30,
    "learners": 1,
    "native_threads_per_worker": 1,
    "host_reserve_sample_seconds": 1,
    "host_reserve_sample_count": 2,
    "checkpoint_retained_latest": 2,
    "checkpoint_retained_final": 1,
    "checkpoint_write_reserve_multiple": 2,
}

# Evaluation grouping / split defaults (prefix20_total100_v1).
EVALUATION_LIMITS: dict[str, int] = {
    "dev_count_default": 128,
    "final_count_default": 512,
    "shift_max": 20,
    "seed_default": 20260913,
    "split_seed": 20260913,
}


# Full, serializable inventories are the source of profile content identities.
# Returning fresh containers prevents accidental changes through a caller.
def profile_inventory() -> dict:
    from oeis_learn.decoder.program_codec import codec_identity

    return {
        "schema_version": "foundation/v1",
        "learning": {
            "track": TRACK,
            "objective": OBJECTIVE,
            "initialization": INITIALIZATION,
            "horizon": [VISIBLE_HORIZON, FULL_HORIZON],
            "policy": POLICY,
        },
        "language": {
            "id": LANGUAGE_PROFILE_ID,
            "baseline_commit": BASELINE_COMMIT,
            "operators": list(ALLOWED_OPERATORS),
            "excluded_operators": list(EXCLUDED_OPERATORS),
            "wrapper": {
                "entrypoint": "compute",
                "input": ["$n", "i32"],
                "results": ["i64"] * 4,
                "i64_locals": list(I64_LOCAL_NAMES),
                "i32_locals": list(I32_LOCAL_NAMES),
                "initial_value": "0",
                "limb_order": "little_endian_twos_complement",
            },
            "numeric_ranges": {k: [str(lo), str(hi)] for k, (lo, hi) in NUMERIC_RANGES.items()},
            "native_semantics": "wasm_bit_vector",
            "wide_semantics": "checked_signed_256",
            "macros": {k: list(v) for k, v in MACROS.items()},
            "max_block_depth": 8,
            "max_nested_loops": 3,
            "forbidden": [
                "imports",
                "calls",
                "memories",
                "globals",
                "tables",
                "start",
                "recursion",
                "floats",
                "host_io",
            ],
        },
        "codec": codec_identity(),
        "resource": {
            "id": RESOURCE_PROFILE_ID,
            "limits": dict(RESOURCE_LIMITS),
            "worker_network": "disabled",
            "worker_gpu": "disabled",
            "runtime_source": "read_only",
            "output_mount": "bounded",
            "seccomp": "docker_default",
            "privileged": False,
            "host_ipc": False,
        },
        "evaluation": {
            "id": EVALUATION_PROFILE_ID,
            "limits": dict(EVALUATION_LIMITS),
            "visible_terms": VISIBLE_HORIZON,
            "total_terms": FULL_HORIZON,
            "policy": POLICY,
        },
    }


def profile_digests() -> dict[str, str]:
    import hashlib
    import json

    inventory = profile_inventory()
    return {
        key: "sha256:"
        + hashlib.sha256(
            json.dumps(inventory[key], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        for key in ("language", "codec", "resource", "evaluation")
    }


def validate_profile_file(path) -> None:
    import yaml

    from oeis_learn.experiments.config import ConfigError, _UniqueLoader

    with open(path, encoding="utf-8") as stream:
        try:
            data = yaml.load(stream, Loader=_UniqueLoader)
        except yaml.YAMLError as exc:
            raise ConfigError("invalid profile YAML") from exc
    expected = profile_inventory()
    expected["digests"] = profile_digests()
    if data != expected:
        raise ConfigError("profile inventory or content identity differs from frozen foundation/v1")


MACROS = MappingProxyType(MACROS)
NUMERIC_RANGES = MappingProxyType(NUMERIC_RANGES)
RESOURCE_LIMITS = MappingProxyType(RESOURCE_LIMITS)
EVALUATION_LIMITS = MappingProxyType(EVALUATION_LIMITS)
