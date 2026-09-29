"""Unit tests for Curriculum Scheduler and automated graduation gates."""

import pytest
from oeis_learn.curriculum.scheduler import CurriculumScheduler


def test_scheduler_rolling_pass_rate_and_competence():
    scheduler = CurriculumScheduler(initial_stage=1, window_size=10)

    # Register 4 prompts in Stage 1
    scheduler.register_prompt("A000001", stage=1, tags=["core", "easy"])
    scheduler.register_prompt("A000002", stage=1, tags=["core", "easy"])
    scheduler.register_prompt("A000003", stage=1, tags=["core", "easy"])
    scheduler.register_prompt("A000004", stage=1, tags=["core", "easy"])

    # Record 90% success on all prompts
    for pid in ["A000001", "A000002", "A000003", "A000004"]:
        for _ in range(9):
            scheduler.record_outcome(pid, success=True)
        scheduler.record_outcome(pid, success=False)

    c_1, min_cov = scheduler.compute_stage_competence(1)
    assert c_1 == 0.90
    assert min_cov == 0.90

    # Check graduation
    graduated, new_stage = scheduler.check_and_update_graduation()
    assert graduated is True
    assert new_stage == 2
    assert scheduler.active_stage == 2


def test_scheduler_coverage_minimum_gate():
    scheduler = CurriculumScheduler(initial_stage=1, coverage_min_threshold=0.50)

    scheduler.register_prompt("A000001", stage=1)
    scheduler.register_prompt("A000002", stage=1)

    # Prompt 1 has 100% pass rate, Prompt 2 has 20% pass rate
    for _ in range(10):
        scheduler.record_outcome("A000001", success=True)
    for _ in range(2):
        scheduler.record_outcome("A000002", success=True)
    for _ in range(8):
        scheduler.record_outcome("A000002", success=False)

    c_1, min_cov = scheduler.compute_stage_competence(1)
    assert c_1 == 0.60
    assert min_cov == 0.20

    # Coverage minimum fails (0.20 < 0.50)
    graduated, _ = scheduler.check_and_update_graduation()
    assert graduated is False
    assert scheduler.active_stage == 1
