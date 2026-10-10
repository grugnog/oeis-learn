"""Legacy textual localization is explicitly unavailable."""

import pytest
from oeis_learn.sandbox.tracer import locate_divergence_token_span


def test_locate_divergence_token_span_requires_execution_evidence():
    with pytest.raises(ValueError, match="unsupported"):
        locate_divergence_token_span("i256.const 1", divergence_step=2, total_tokens=20)
