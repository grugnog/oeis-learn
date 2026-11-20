"""Mixed policy inventories retain byte/hash and study authority checks."""
import pytest
from oeis_learn.generalization.progressive.contracts import atomic_write, reference, seal
from oeis_learn.pipeline_validation.progressive_study import policy


def inventory(root, *, study_bytes=0, other_bytes=0):
    study = seal(dict(version="progressive-study-policy/v1", token_cap=10,
        target_tokens=5, elapsed_cap_us=100, screening=dict(startup_us=1,
        evaluation_us=[1, 1, 1], training_us=1, finalization_us=1),
        panel=[], evaluations={}, confirmation={}))
    atomic_write(root / "study.json", study)
    with (root / "study.json").open("ab") as stream:
        stream.write(b" " * study_bytes)
    atomic_write(root / "calibration.json", {"version": "unrelated-policy/v1"})
    with (root / "calibration.json").open("ab") as stream:
        stream.write(b" " * other_bytes)
    refs = [dict(kind="policy", artifact=reference(root, name))
            for name in ("calibration.json", "study.json")]
    return dict(purpose="acceptance", payload=dict(references=refs)), study


def test_large_calibration_policy_does_not_hide_small_study(tmp_path):
    campaign, study = inventory(tmp_path, other_bytes=49 << 20)
    assert policy(tmp_path, campaign, study["identity"]) == study
    # The original large document is still authenticated, not skipped by size.
    with (tmp_path / "calibration.json").open("r+b") as stream:
        stream.write(b"X")
    with pytest.raises(ValueError):
        policy(tmp_path, campaign, study["identity"])


def test_actual_study_retains_its_own_size_limit(tmp_path):
    campaign, study = inventory(tmp_path, study_bytes=32 << 20)
    with pytest.raises(ValueError, match="study policy byte cap"):
        policy(tmp_path, campaign, study["identity"])


def test_duplicate_study_is_rejected_even_after_large_calibration(tmp_path):
    campaign, study = inventory(tmp_path, other_bytes=33 << 20)
    campaign["payload"]["references"].append(campaign["payload"]["references"][-1])
    with pytest.raises(ValueError, match="one original frozen study policy"):
        policy(tmp_path, campaign, study["identity"])


def test_mixed_policy_inventory_still_has_a_bounded_document_size(tmp_path):
    campaign, study = inventory(tmp_path)
    campaign["payload"]["references"][0]["artifact"]["bytes"] = (128 << 20) + 1
    with pytest.raises(ValueError, match="artifact inventory cap"):
        policy(tmp_path, campaign, study["identity"])
