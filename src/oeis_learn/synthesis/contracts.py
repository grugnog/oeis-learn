"""Disabled, versioned shape stubs for the Phase 2 handoff.

Record envelopes carry untrusted content references. Shape validation is never
semantic validation, verification, profile qualification or permission to run.
"""
import hashlib
import json
from importlib.resources import files

from jsonschema import Draft202012Validator, validators

MAX_FRAME_BYTES = 4 * 1024 * 1024
SCHEMA = json.loads(files(__package__).joinpath("records.schema.json").read_text())
RECORD_KINDS = tuple(kind for kind in SCHEMA["$defs"] if kind != "Request")
PRODUCTION_ENABLED = False


class ContractError(ValueError):
    """A malformed or unsupported shape cannot enter the disabled service."""


# JSON numbers such as 1.0 and booleans are not bounded integer metadata.
StrictValidator = validators.extend(
    Draft202012Validator,
    type_checker=Draft202012Validator.TYPE_CHECKER.redefine(
        "integer", lambda checker, value: type(value) is int
    ),
)


def _validate(kind, value):
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
