"""Unit tests for recurrence transition frame tracking and loop rotation validation."""

from __future__ import annotations

import pytest
from oeis_learn.decoder.environment_tracker import EnvironmentTracker, RecurrenceFrameTracker


def test_recurrence_frame_transition_lifecycle():
    tracker = RecurrenceFrameTracker(
        state_locals=["$a", "$b"],
        next_locals=["$temp"],
        progress_local="$i",
    )
    assert tracker.phase == "GUARD"
    assert tracker.is_backedge_ready() is False

    # 1. Establish guard condition
    tracker.transition_to_compute_next()
    assert tracker.phase == "COMPUTE_NEXT"

    # 2. Compute next state into temporary
    tracker.record_temp_assigned("$temp")
    tracker.transition_to_commit_all()
    assert tracker.phase == "COMMIT_ALL"

    # 3. Commit state rotation: $a = $b, $b = $temp
    tracker.record_state_commit("$a")
    assert tracker.is_commit_complete() is False
    tracker.record_state_commit("$b")
    assert tracker.is_commit_complete() is True

    # 4. Advance progress local
    tracker.transition_to_advance()
    assert tracker.phase == "ADVANCE"
    tracker.record_progress_advanced()
    assert tracker.is_backedge_ready() is True


def test_recurrence_frame_rejects_premature_backedge():
    tracker = RecurrenceFrameTracker(
        state_locals=["$a", "$b"],
        next_locals=["$temp"],
        progress_local="$i",
    )
    # Trying to jump back to loop without rotating state or advancing progress
    assert tracker.can_emit_backedge() is False
