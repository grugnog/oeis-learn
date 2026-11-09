import pytest
from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.pipeline_validation.progressive_qualification import matrix, require_hip
from tests.unit.test_progressive_training_policy import profile


def test_serial_matrix_deduplicates_exact_selected_scopes():
    p = profile(0.1)
    m = matrix([p, p], source=digest("source"), image=digest("image"), engine="docker")
    assert m["serial"] and len(m["cases"]) == 4
    assert {c["input_length"] for c in m["cases"]} == {4, 20}
    assert {c["microbatch"] for c in m["cases"]} == {7, 16}
    assert len({c["identity"] for c in m["cases"]}) == 4
    with pytest.raises(ValueError):
        matrix([p], source=digest("s"), image=digest("i"), engine="host")
    with pytest.raises(ValueError):
        matrix([p], source=digest("s"), image=digest("i"), engine="docker", input_lengths=[100])


def test_mandatory_hip_rejects_cpu_without_fallback(monkeypatch):
    import torch

    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(RuntimeError, match="actual HIP"):
        require_hip()
