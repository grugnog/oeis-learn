"""Run Tracking and Experiment Management for OEIS Learn.

Provides a structured directory layout for tracking runs:
runs/
  001_baseline_cold_start/
    config.yaml
    metadata.json
    checkpoints/
      model_epoch_010.pt
      ...
    logs/
      run.log
      telemetry.json
    reports/
      summary.md
      discovered_theorems.md
      preflight_report.json
  002_phase2_bootstrapping/
    ...
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import platform
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

logger = logging.getLogger("oeis_learn.tracking")


@dataclass
class RunMetadata:
    """Metadata recorded for an execution run."""

    run_id: str
    name: str
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    finished_at: Optional[str] = None
    status: str = "INITIALIZED"  # INITIALIZED, PREFLIGHT, BLOCKED, AUTHORIZED, RUNNING, OVERRIDDEN_UNQUALIFIED, etc.
    qualification_state: Optional[str] = None  # AUTHORIZED, BLOCKED, OVERRIDDEN_UNQUALIFIED
    override: Optional[Dict[str, Any]] = None
    host: Dict[str, Any] = field(default_factory=dict)
    summary_metrics: Dict[str, Any] = field(default_factory=dict)
    manifest_snapshots: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class RunContext:
    """Manages the lifecycle, file paths, and output artifacts for an active or past run."""

    def __init__(self, run_dir: Path, metadata: RunMetadata):
        self.run_dir = run_dir
        self.metadata = metadata

        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.logs_dir = self.run_dir / "logs"
        self.reports_dir = self.run_dir / "reports"

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.config_path = self.run_dir / "config.yaml"
        self.metadata_path = self.run_dir / "metadata.json"
        self.log_file = self.logs_dir / "run.log"
        self.telemetry_file = self.logs_dir / "telemetry.json"
        self.summary_path = self.reports_dir / "summary.md"
        self.theorems_path = self.reports_dir / "discovered_theorems.md"
        self.preflight_report_path = self.reports_dir / "preflight_report.json"
        self.synthesis_results_path = self.reports_dir / "synthesis_results.json"

        self._save_metadata()

    def set_status(self, status: str) -> None:
        """Update run execution status."""
        self.metadata.status = status
        if status in ("COMPLETED", "FAILED"):
            self.metadata.finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self._save_metadata()

    def record_summary_metrics(self, metrics: Dict[str, Any]) -> None:
        """Record final summary metrics to metadata."""
        self.metadata.summary_metrics.update(metrics)
        self._save_metadata()

    def save_config(self, config_dict: Dict[str, Any]) -> None:
        """Save a snapshot of the hyperparameters and configuration."""
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    def get_checkpoint_path(self, filename: str) -> str:
        """Get full path to a checkpoint within this run's checkpoints folder."""
        return str(self.checkpoints_dir / filename)

    def set_qualification_state(self, state: str) -> None:
        """Sets run qualification state (AUTHORIZED, BLOCKED, OVERRIDDEN_UNQUALIFIED)."""
        self.metadata.qualification_state = state
        if state == "AUTHORIZED":
            self.metadata.status = "AUTHORIZED"
        elif state == "BLOCKED":
            self.metadata.status = "BLOCKED"
        elif state == "OVERRIDDEN_UNQUALIFIED":
            self.metadata.status = "OVERRIDDEN_UNQUALIFIED"
        self._save_metadata()

    def record_override(
        self,
        operator: str,
        reason: str,
        diagnostic_intent: str,
        failed_gate_ids: List[str],
        policy_id: str,
    ) -> Dict[str, Any]:
        """Records an auditable diagnostic override making the run permanently unqualified."""
        override_id = f"ovr_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        override_entry = {
            "override_id": override_id,
            "operator": operator,
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "reason": reason,
            "diagnostic_intent": diagnostic_intent,
            "failed_gate_ids": failed_gate_ids,
            "policy_id": policy_id,
        }
        self.metadata.override = override_entry
        self.set_qualification_state("OVERRIDDEN_UNQUALIFIED")
        return override_entry

    def allocate_artifact(
        self,
        prefix: str,
        content_bytes: bytes,
        ext: str = "json",
    ) -> Tuple[Path, str]:
        """Allocates an append-only, content-addressed artifact file in reports/."""
        import hashlib
        digest = f"sha256:{hashlib.sha256(content_bytes).hexdigest()}"
        short_hash = digest.split(":")[1][:12]
        filename = f"{prefix}_{short_hash}.{ext}"
        artifact_path = self.reports_dir / filename
        with open(artifact_path, "wb") as f:
            f.write(content_bytes)
        return artifact_path, digest

    def save_manifest_snapshot(self, manifest_path: str, name: str = "benchmark_manifest") -> str:
        """Takes an immutable snapshot of an input manifest and records its digest."""
        import hashlib
        with open(manifest_path, "rb") as f:
            content = f.read()
        digest = f"sha256:{hashlib.sha256(content).hexdigest()}"
        snapshot_file = self.run_dir / f"{name}_snapshot.json"
        with open(snapshot_file, "wb") as f:
            f.write(content)
        self.metadata.manifest_snapshots[name] = digest
        self._save_metadata()
        return digest

    def is_promotable(self) -> bool:
        """Returns True only if run completed normally with explicit qualification."""
        return (
            self.metadata.status == "COMPLETED_QUALIFIED"
            and self.metadata.qualification_state == "AUTHORIZED"
            and self.metadata.override is None
        )

    def _save_metadata(self) -> None:
        """Write metadata to metadata.json."""
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata.to_dict(), f, indent=2)


class RunManager:
    """Manages experiment runs, auto-allocates run IDs, and maintains runs/ directory."""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            # Root repo runs/ directory
            repo_root = Path(__file__).resolve().parent.parent.parent.parent
            self.base_dir = repo_root / "runs"
        else:
            self.base_dir = Path(base_dir).resolve()

        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_next_run_id(self) -> str:
        """Scans runs directory and generates next sequential 3-digit run ID (e.g., '001', '002')."""
        existing = [d.name for d in self.base_dir.iterdir() if d.is_dir()]
        max_num = 0
        for name in existing:
            match = re.match(r"^(\d+)", name)
            if match:
                num = int(match.group(1))
                if num > max_num:
                    max_num = num
        return f"{max_num + 1:03d}"

    def create_run(
        self,
        run_id: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> RunContext:
        """Initializes a new run directory and returns its RunContext."""
        allocated_id = run_id or self.get_next_run_id()
        run_name = name or f"run_{allocated_id}"

        # Directory name format: '001_name' or '001'
        if name and not allocated_id.endswith(name):
            dir_name = f"{allocated_id}_{name}"
        else:
            dir_name = allocated_id

        run_dir = self.base_dir / dir_name
        run_dir.mkdir(parents=True, exist_ok=True)

        host_info = {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
        }

        metadata = RunMetadata(
            run_id=allocated_id,
            name=run_name,
            host=host_info,
        )

        ctx = RunContext(run_dir=run_dir, metadata=metadata)
        if config:
            ctx.save_config(config)

        logger.info(f"Initialized experiment run {allocated_id} at {run_dir}")
        return ctx

    def list_runs(self) -> List[Dict[str, Any]]:
        """Returns metadata summaries for all recorded runs."""
        runs = []
        for d in sorted(self.base_dir.iterdir()):
            if d.is_dir():
                meta_file = d / "metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as f:
                            runs.append(json.load(f))
                    except Exception as e:
                        runs.append({"run_dir": str(d), "error": str(e)})
                else:
                    runs.append({"run_dir": str(d), "status": "NO_METADATA"})
        return runs

    def archive_legacy_run(
        self,
        run_id: str = "001_baseline_cold_start",
        reports_dir: Optional[str] = None,
        checkpoints_dir: Optional[str] = None,
    ) -> Optional[RunContext]:
        """Archives legacy top-level reports/ and checkpoints/ into a structured run directory."""
        repo_root = self.base_dir.parent
        rep_dir = Path(reports_dir) if reports_dir else (repo_root / "reports")
        chk_dir = Path(checkpoints_dir) if checkpoints_dir else (repo_root / "checkpoints")

        run_dir = self.base_dir / run_id
        if run_dir.exists():
            logger.info(f"Legacy run already archived at {run_dir}")
            meta_path = run_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta_dict = json.load(f)
                metadata = RunMetadata(**meta_dict)
                return RunContext(run_dir=run_dir, metadata=metadata)

        run_dir.mkdir(parents=True, exist_ok=True)
        ctx = RunContext(
            run_dir=run_dir,
            metadata=RunMetadata(
                run_id="001",
                name="baseline_cold_start",
                status="COMPLETED",
                summary_metrics={
                    "duration_hours": 18.65,
                    "final_pass_rate": 0.0,
                    "final_stage_competence": 0.0,
                    "note": "18.65h baseline cold start run (Phase 1, without SFT warmup & header FSM)",
                },
            ),
        )

        # Copy existing checkpoints
        if chk_dir.exists():
            for f in chk_dir.glob("*.pt"):
                shutil.copy2(f, ctx.checkpoints_dir / f.name)

        # Copy existing logs & reports
        if rep_dir.exists():
            log_src = rep_dir / "long_e2e_run.log"
            if log_src.exists():
                shutil.copy2(log_src, ctx.log_file)

            sum_src = rep_dir / "long_e2e_summary.md"
            if sum_src.exists():
                shutil.copy2(sum_src, ctx.summary_path)

            thm_src = rep_dir / "long_run_discovered_theorems.md"
            if thm_src.exists():
                shutil.copy2(thm_src, ctx.theorems_path)

        logger.info(f"Successfully archived legacy benchmark outputs to {run_dir}")
        return ctx
