"""Reviewable payload/mount contracts and disabled routing, not sandbox qualification."""
import copy
import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from oeis_learn.generalization.interfaces import (
    ArtifactRef, Command, CommandRequest, EXIT_CODES, FullStateCheckpoint,
    ScientificStatus, SoftwareStatus, WeightsOnlyCheckpoint,
)
from oeis_learn.generalization.role_contracts import ROLE_INPUTS, ROLE_OUTPUTS, Role
from oeis_learn.pipeline_validation.generalization_cli import HANDLERS, dispatch
from test_generalization_authority import validator

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures/generalization"


def test_visible_prompt_has_no_truth_or_protected_labels():
    record = json.loads((FIXTURES / "payloads.json").read_text())["visible-prompt"]
    validator("visible-prompt").validate(record)
    for field in ["source", "family", "stratum", "hidden_values", "target_program", "assignment"]:
        changed = copy.deepcopy(record)
        changed[field] = "forbidden"
        assert list(validator("visible-prompt").iter_errors(changed))
    for value in ["01", "-0", "+1", "1\n", 1, True]:
        changed = copy.deepcopy(record)
        changed["values"][0] = value
        assert list(validator("visible-prompt").iter_errors(changed))


def test_training_and_scorer_payloads_are_not_generator_inputs():
    records = json.loads((FIXTURES / "payloads.json").read_text())
    for name in ["training-target", "scorer-target"]:
        validator(name).validate(records[name])
        assert list(validator("visible-prompt").iter_errors(records[name]))
    changed = copy.deepcopy(records["training-target"])
    changed["role"] = "assessment"
    assert list(validator("training-target").iter_errors(changed))


def test_mounts_match_independently_declared_permissions():
    expected = json.loads((FIXTURES / "role-expectations.json").read_text())
    assert {r.value: [m.channel for m in mounts] for r, mounts in ROLE_INPUTS.items()} == expected
    assert all(m.read_only for mounts in ROLE_INPUTS.values() for m in mounts)
    assert all(not m.read_only for m in ROLE_OUTPUTS.values())
    assert len({m.destination for m in ROLE_OUTPUTS.values()}) == len(Role)
    with pytest.raises(TypeError):
        ROLE_INPUTS[Role.LEARNER] = ()


def test_immutable_types_and_distinct_checkpoint_kinds():
    ref = ArtifactRef("weights.bin", "sha256:" + "0" * 64, 1)
    weights = WeightsOnlyCheckpoint(ref, "random")
    full = FullStateCheckpoint(ref, ref, ref.sha256, 0, 0, ("latest",))
    assert not isinstance(weights, FullStateCheckpoint)
    for record, field in [(ref, "path"), (full, "tokens"), (weights, "initialization")]:
        with pytest.raises(FrozenInstanceError):
            setattr(record, field, None)


@pytest.mark.parametrize("operation", list(Command))
def test_all_prospective_commands_block_without_io(operation, tmp_path):
    output = tmp_path / "must-not-exist"
    request = CommandRequest(operation, str(output))
    result = dispatch(request)
    assert result.software_status == SoftwareStatus.BLOCKED
    assert result.scientific_status == ScientificStatus.UNMEASURED
    assert result.reason == "generalization_phase_c_not_implemented"
    assert result.artifacts == ()
    assert not output.exists()
    assert set(HANDLERS) == set(Command)
    assert EXIT_CODES[result.software_status] == 3


def test_unknown_command_rejected_and_exit_codes_preserved():
    with pytest.raises(ValueError):
        dispatch(CommandRequest("skip-tests", "unused"))
    assert dict(EXIT_CODES) == {
        "PASS": 0, "FAIL": 1, "BLOCKED": 3, "PARTIAL": 3,
        "INCONCLUSIVE": 3, "INTERRUPTED": 130,
    }
