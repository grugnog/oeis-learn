"""Integration tests for readiness policy diagnostic overrides and run qualification states."""

from __future__ import annotations

import json
import os
import tempfile
import pytest
from oeis_learn.cli.main import cli
from oeis_learn.evaluation.readiness import load_readiness_policy
from oeis_learn.tracking.run_manager import RunManager, RunMetadata


def test_run_context_override_provenance():
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = RunManager(base_dir=tmpdir)
        ctx = manager.create_run(run_id="009", name="test_override_run")

        assert ctx.metadata.status == "INITIALIZED"
        assert ctx.is_promotable() is False

        # Apply diagnostic override
        entry = ctx.record_override(
            operator="lead_researcher",
            reason="Diagnosing Stage 2 accumulator stalls",
            diagnostic_intent="Collect raw traces",
            failed_gate_ids=["runtime_trap_rate", "curriculum_stage1_competence"],
            policy_id="sha256:" + "0" * 64,
        )

        assert ctx.metadata.qualification_state == "OVERRIDDEN_UNQUALIFIED"
        assert ctx.metadata.status == "OVERRIDDEN_UNQUALIFIED"
        assert ctx.metadata.override is not None
        assert ctx.metadata.override["operator"] == "lead_researcher"
        assert ctx.is_promotable() is False

        # Attempt to mark completed
        ctx.set_status("COMPLETED_QUALIFIED")
        # Since override is present, is_promotable MUST remain False
        assert ctx.is_promotable() is False
