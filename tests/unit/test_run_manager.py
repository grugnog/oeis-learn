"""Unit tests for RunManager and RunContext experiment tracking."""

import json
from pathlib import Path
from oeis_learn.tracking.run_manager import RunManager, RunMetadata


def test_run_manager_creates_and_structures_run(tmp_path):
    manager = RunManager(base_dir=str(tmp_path / "runs"))

    # Allocate first run
    next_id = manager.get_next_run_id()
    assert next_id == "001"

    ctx = manager.create_run(run_id=next_id, name="test_experiment", config={"lr": 0.001, "epochs": 10})
    assert ctx.run_dir.exists()
    assert ctx.checkpoints_dir.exists()
    assert ctx.logs_dir.exists()
    assert ctx.reports_dir.exists()
    assert ctx.config_path.exists()
    assert ctx.metadata_path.exists()

    # Next run ID should now be 002
    assert manager.get_next_run_id() == "002"

    # Status updates
    ctx.set_status("RUNNING")
    assert ctx.metadata.status == "RUNNING"

    ctx.record_summary_metrics({"pass_rate": 0.85, "competence": 0.90})
    assert ctx.metadata.summary_metrics["pass_rate"] == 0.85

    ctx.set_status("COMPLETED")
    assert ctx.metadata.status == "COMPLETED"
    assert ctx.metadata.finished_at is not None

    # List runs
    runs = manager.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_id"] == "001"
    assert runs[0]["name"] == "test_experiment"


def test_run_manager_archive_legacy(tmp_path):
    rep_dir = tmp_path / "reports"
    chk_dir = tmp_path / "checkpoints"
    rep_dir.mkdir()
    chk_dir.mkdir()

    # Create dummy legacy files
    (rep_dir / "long_e2e_run.log").write_text("dummy log")
    (rep_dir / "long_e2e_summary.md").write_text("# Summary")
    (chk_dir / "model_epoch_010.pt").write_text("model weights")

    runs_dir = tmp_path / "runs"
    manager = RunManager(base_dir=str(runs_dir))

    ctx = manager.archive_legacy_run(
        run_id="001_baseline_cold_start",
        reports_dir=str(rep_dir),
        checkpoints_dir=str(chk_dir),
    )

    assert ctx is not None
    assert (ctx.checkpoints_dir / "model_epoch_010.pt").exists()
    assert ctx.log_file.exists()
    assert ctx.summary_path.exists()
    assert ctx.metadata.status == "COMPLETED"
