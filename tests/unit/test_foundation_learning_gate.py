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
