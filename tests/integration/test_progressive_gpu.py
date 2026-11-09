"""Serial mandatory device matrix; ordinary CPU runs explicitly exclude this marker."""

import json
import os
from pathlib import Path
import pytest

pytestmark = pytest.mark.gpu


def test_serial_selected_actual_hip_matrix(tmp_path):
    from oeis_learn.pipeline_validation.progressive_qualification import qualify, require_hip

    mandatory = os.environ.get("OEIS_REQUIRE_PROGRESSIVE_HIP") == "1"
    try:
        require_hip()
    except RuntimeError:
        if mandatory:
            raise
        pytest.skip("actual HIP unavailable; mandatory mode never skips")
    path = os.environ.get("OEIS_PROGRESSIVE_HIP_MATRIX")
    if not path:
        if mandatory:
            pytest.fail("resolved selected HIP matrix required")
        pytest.skip("no selected HIP qualification request")
    request = json.loads(Path(path).read_text())
    from oeis_learn.synthesis.client import SynthesisClient

    output = os.environ.get("OEIS_PROGRESSIVE_HIP_OUTPUT")
    if mandatory and not output:
        pytest.fail("persistent qualification output required")
    with SynthesisClient(request["identity"]) as client:
        result = qualify(
            request, Path(output) if output else tmp_path / "qualification", client=client
        )
    assert result["status"] == "PASS" and result["requested"] == result["executed"]
    assert result["device"] == "actual_HIP"
