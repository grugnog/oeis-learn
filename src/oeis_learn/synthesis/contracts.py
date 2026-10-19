"""Disabled, versioned shape stubs for the Phase 2 handoff.

Record envelopes carry untrusted content references. Shape validation is never
semantic validation, verification, profile qualification or permission to run.
"""

import hashlib
import json
from importlib.resources import files

MAX_FRAME_BYTES = 4 * 1024 * 1024
SCHEMA = json.loads(files(__package__).joinpath("records.schema.json").read_text())
RECORD_KINDS = tuple(kind for kind in SCHEMA["$defs"] if kind != "Request")
PRODUCTION_ENABLED = False


class ContractError(ValueError):
    """A malformed or unsupported shape cannot enter the disabled service."""


def _validate(kind, value):
    # Shape validation is image-side; host identity/state operations stay stdlib-only.
    from jsonschema import Draft202012Validator, validators

    # JSON numbers such as 1.0 and booleans are not bounded integer metadata.
    StrictValidator = validators.extend(
        Draft202012Validator,
        type_checker=Draft202012Validator.TYPE_CHECKER.redefine(
            "integer", lambda checker, value: type(value) is int
        ),
    )

    if kind not in SCHEMA["$defs"]:
        raise ContractError("unknown record kind")
    try:
        raw = json.dumps(value, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError) as exc:
        raise ContractError("invalid JSON") from exc
    if len(raw) > MAX_FRAME_BYTES:
        raise ContractError("frame exceeds 4 MiB")
    schema = {"$ref": f"#/$defs/{kind}", "$defs": SCHEMA["$defs"]}
    error = next(StrictValidator(schema).iter_errors(value), None)
    if error:
        raise ContractError(f"{kind}: {error.message}")


def validate_request(value):
    _validate("Request", value)
    indices = value["visible_indices"]
    if len(indices) != len(value["visible_values"]):
        raise ContractError("visible indices/values length mismatch")
    if any(int(a) >= int(b) for a, b in zip(indices, indices[1:])):
        raise ContractError("visible indices must strictly increase")
    seen = set()
    for candidate in value["candidates"]:
        candidate_id = candidate["candidate_id"]
        if candidate_id in seen:
            raise ContractError("duplicate candidate ID")
        seen.add(candidate_id)
        source = candidate["original_source"].encode("utf-8")
        if len(source) > 65536:
            raise ContractError("source exceeds 64 KiB")
        if "sha256:" + hashlib.sha256(source).hexdigest() != candidate["original_source_sha256"]:
            raise ContractError("original source hash mismatch")
    return value


def validate_record(kind, value):
    if kind not in RECORD_KINDS:
        raise ContractError("unknown record kind")
    _validate(kind, value)
    return value


def handle_request(value):
    validate_request(value)
    return {
        "protocol": "synthesis-worker/v1",
        "request_id": value["request_id"],
        "worker_generation": value["worker_generation"],
        "outcome": "unsupported",
        "reason": "phase_2_not_implemented",
        "admitted": False,
        "assignments": [],
    }


# Semantic records are separate from the retained disabled Phase 1 envelopes.
from dataclasses import asdict, dataclass
import math
import re


def identity(value):
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
    )


def canonical_integer(text, *, bits=4096):
    """Bound before parsing; the accepted decimal is always canonical."""
    if type(bits) is not int or not 1 <= bits <= 131072:
        raise ContractError("invalid integer bit bound")
    if type(text) is not str or len(text) > (bits * 30103 // 100000 + 2):
        raise ContractError("integer length limit")
    if re.fullmatch(r"0|-?[1-9][0-9]*", text, flags=re.ASCII) is None:
        raise ContractError("noncanonical integer")
    # Parse bounded chunks, without changing Python's process-wide digit guard.
    digits = text.removeprefix("-")
    result = 0
    for pos in range(0, len(digits), 9):
        part = digits[pos : pos + 9]
        result = result * 10 ** len(part) + int(part)
    if result.bit_length() > bits:
        raise ContractError("integer bit limit")
    return -result if text.startswith("-") else result


def decimal_integer(value):
    """Bounded formatting also works above Python's decimal conversion guard."""
    if type(value) is not int or abs(value).bit_length() > 131072:
        raise ContractError("invalid exact integer")
    n, chunks = abs(value), []
    while n >= 1_000_000_000:
        n, part = divmod(n, 1_000_000_000)
        chunks.append(f"{part:09d}")
    return ("-" if value < 0 else "") + str(n) + "".join(reversed(chunks))


def signed_magnitude(value):
    if type(value) is not int or abs(value).bit_length() > 131072:
        raise ContractError("invalid exact integer")
    return (
        0 if value == 0 else (1 if value > 0 else -1),
        abs(value).to_bytes(max(1, (abs(value).bit_length() + 7) // 8), "little"),
    )


def from_signed_magnitude(sign, raw, *, bits=4096):
    if type(sign) is not int or sign not in (-1, 0, 1) or type(raw) is not bytes:
        raise ContractError("invalid sign/magnitude")
    if not raw or len(raw) > (bits + 7) // 8 or (len(raw) > 1 and raw[-1] == 0):
        raise ContractError("nonminimal or overlong magnitude")
    value = int.from_bytes(raw, "little")
    if (value == 0) != (sign == 0) or value.bit_length() > bits:
        raise ContractError("noncanonical sign/magnitude")
    return sign * value


@dataclass(frozen=True)
class ResourceProfile:
    input_bits: int = 4096
    literal_bits: int = 4096
    intermediate_bits: int = 16384
    output_bytes: int = 2048
    arena_entries: int = 65536
    arena_bytes: int = 32 * 1024 * 1024
    memory_bytes: int = 64 * 1024 * 1024
    term_fuel: int = 10_000_000
    aggregate_fuel: int = 100_000_000
    reference_steps: int = 1_000_000
    aggregate_reference_steps: int = 50_000_000
    control_max: int = 65536
    loop_iterations: int = 1024
    aggregate_loop_iterations: int = 65536
    ast_nodes: int = 4096
    source_bytes: int = 65536
    nesting: int = 32
    bigint_locals: int = 16
    control_locals: int = 8
    bool_locals: int = 8
    term_seconds: float = 2.0
    evaluation_seconds: float = 30.0
    cleanup_seconds: float = 5.0

    def __post_init__(self):
        for key, value in asdict(self).items():
            if key.endswith("seconds"):
                if (
                    type(value) not in (float, int)
                    or not math.isfinite(value)
                    or not 0 < value <= 3600
                ):
                    raise ContractError(f"invalid {key}")
                object.__setattr__(self, key, float(value))
            elif type(value) is not int or not 0 < value <= 2**53 - 1:
                raise ContractError(f"invalid {key}")
        if (
            max(self.input_bits, self.literal_bits) > self.intermediate_bits
            or self.intermediate_bits > 131072
        ):
            raise ContractError("invalid numeric nesting")
        if self.arena_bytes > self.memory_bytes or self.memory_bytes > 1024**3:
            raise ContractError("invalid memory nesting")
        for name, cap in {
            "source_bytes": 65536,
            "ast_nodes": 4096,
            "nesting": 32,
            "bigint_locals": 16,
            "control_locals": 8,
            "bool_locals": 8,
            "control_max": 2**31 - 1,
            "aggregate_loop_iterations": 2**31 - 1,
            "loop_iterations": 1024,
            "arena_entries": 2**31 - 1,
            "output_bytes": 16384,
        }.items():
            if getattr(self, name) > cap:
                raise ContractError(f"unsupported {name}")

    @property
    def identity(self):
        return identity({"version": "bigint-resources/v1", **asdict(self)})


@dataclass(frozen=True)
class ExecutionProfile:
    runtime_sha256: str
    native_sha256: str
    source_sha256: str
    toolchain_sha256: str
    locks_sha256: str
    resources: ResourceProfile = ResourceProfile()
    language: str = "bigint-body/v1"
    codec: str = "bigint-codec/v1"
    runtime: str = "bigint-wasm/v1"
    operations: tuple = (
        "int",
        "param.get",
        "get",
        "add",
        "sub",
        "mul",
        "eq",
        "lt",
        "le",
        "to_control",
        "to_bigint",
        "control",
        "increment",
        "control.lt",
        "true",
        "false",
        "select",
        "set",
        "if",
        "loop",
        "return",
    )

    def __post_init__(self):
        for name in (
            "runtime_sha256",
            "native_sha256",
            "source_sha256",
            "toolchain_sha256",
            "locks_sha256",
        ):
            if re.fullmatch(r"sha256:[0-9a-f]{64}", getattr(self, name)) is None:
                raise ContractError(f"invalid {name}")
        if (self.language, self.codec, self.runtime) != (
            "bigint-body/v1",
            "bigint-codec/v1",
            "bigint-wasm/v1",
        ):
            raise ContractError("unsupported profile; retained records are immutable")
        if self.operations != type(self).__dataclass_fields__["operations"].default:
            raise ContractError("unknown operations")

    @property
    def identity(self):
        return identity(asdict(self))
