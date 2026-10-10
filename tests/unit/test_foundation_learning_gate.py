"""Negative gate cases never infer accelerator readiness from a CPU pass."""

import pytest
from oeis_learn.evaluation.learning_gate import require_g8, check_junit


def test_incomplete_and_skipped_learning_evidence_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="complete"):
        require_g8({"gate": "G8"}, evidence_root=tmp_path, diagnostic=True)
    empty = tmp_path / "empty.xml"
    empty.write_text("<testsuite/>")
    with pytest.raises(ValueError, match="empty"):
        check_junit(empty)
    skipped = tmp_path / "skipped.xml"
    skipped.write_text("<testsuite><testcase><skipped/></testcase></testsuite>")
    with pytest.raises(ValueError, match="skipped"):
        check_junit(skipped)
    incomplete = tmp_path / "incomplete.xml"
    incomplete.write_text('<testsuite><testcase classname="fake"/></testsuite>')
    with pytest.raises(ValueError, match="coverage"):
        check_junit(incomplete, "cache")


def test_accelerator_evidence_cannot_omit_variants_or_use_a_tiny_proxy(tmp_path):
    from oeis_learn.evaluation.learning_gate import check_hardware_junit

    path = tmp_path / "gpu.xml"
    path.write_text(
        '<testsuite><testcase classname="tests.integration.test_foundation_learning_gpu" '
        'name="test_gpu_fp16_scaler_finite_optimizer_update[sinusoidal]"/></testsuite>'
    )
    with pytest.raises(ValueError, match="coverage"):
        check_hardware_junit(path)
