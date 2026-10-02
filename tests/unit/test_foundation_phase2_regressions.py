"""Public-boundary regressions found in the phase-2 implementation review."""

import copy
import json
from pathlib import Path

import pytest

from oeis_learn.decoder import program_codec as codec
from oeis_learn.experiments.models import FoundationValidationError, validate_artifact

ROOT = Path(__file__).resolve().parents[2]


def examples():
    return json.loads(
        (ROOT / "specs/007-experiment-foundation/contracts/schema-examples.json").read_text()
    )


def test_valid_body_is_accepted():
    codec.validate_body(codec.encode_body("i256.zero"))


def test_return_signature_is_required():
    with pytest.raises(codec.CodecError):
        codec.encode_body("nop")


def test_underflow_is_masked():
    assert codec.TOKEN_TO_ID["i64.add"] not in codec.allowed_tokens_at([])


@pytest.mark.parametrize("index", range(3))
def test_public_validator_rejects_every_missing_and_extra_field(index):
    original = examples()[index]
    for key in original:
        value = copy.deepcopy(original)
        del value[key]
        with pytest.raises(FoundationValidationError):
            validate_artifact(value)
    value = copy.deepcopy(original)
    value["unexpected"] = True
    with pytest.raises(FoundationValidationError):
        validate_artifact(value)


def test_public_validator_rejects_floating_counts():
    value = examples()[1]
    value["attempt_index"] = 0.0
    with pytest.raises(FoundationValidationError):
        validate_artifact(value)


@pytest.mark.parametrize("index", range(3))
def test_nested_artifact_objects_reject_missing_and_unknown_members(index):
    original = examples()[index]

    def mappings(value, path=()):
        if isinstance(value, dict):
            yield path, value
            for key, child in value.items():
                yield from mappings(child, path + (key,))

    for path, mapping in mappings(original):
        for key in [*mapping, "__unknown__"]:
            value = copy.deepcopy(original)
            target = value
            for part in path:
                target = target[part]
            if key in target:
                del target[key]
            else:
                target[key] = 1
            with pytest.raises(FoundationValidationError):
                validate_artifact(value)
