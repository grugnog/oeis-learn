"""Foundation CLI commands implementation (T008).

Strict command group for foundation artifact administration:
  preflight   -- check hardware prerequisites (delegates to foundation_preflight)
  conformance  -- validate artifacts against artifacts schema
  freeze-cohort -- create registry with deterministic IDs
  build-pool    -- build pool of candidate result digests
  train         -- (placeholder) start strict SFT training
  resume        -- (placeholder) resume a strict run
  inspect       -- (placeholder) inspect a checkpoint or run state

Unsupported commands reject with a clear reason. Implemented slices
delegate to their owning modules.

The primary CLI in main.py uses argparse; this module provides the
pure function implementations that main.py dispatches to, along with
an explicit allow-list of supported commands.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Implementation shared by main.py dispatch and any future click wiring
# ---------------------------------------------------------------------------

# The only commands that are supported in foundation/v1 smoke
_SUPPORTED_COMMANDS = frozenset({
    "preflight",
    "conformance",
    "freeze-cohort",
    "build-pool",
})

# Commands planned for later phases (require their owning slices)
_UNSUPPORTED_COMMANDS = {
    "train": "requires US4 implementation (T044–T045); not yet implemented",
    "resume": "requires US4 implementation (T041–T045); not yet implemented",
    "inspect": "requires US4 implementation (T041, T045); not yet implemented",
    "evaluate": "requires US2 implementation (T024–T028); not yet implemented",
    "finalize": "requires US2 implementation (T026, T028); not yet implemented",
}


def _reject_unsupported(command: str) -> None:
    """Print a clear rejection for an unsupported command and exit non-zero."""
    reason = _UNSUPPORTED_COMMANDS.get(
        command,
        "not yet implemented in the current foundation slice"
    )
    print(f"ERROR: foundation {command!r}: {reason}", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Command implementations (pure functions, no argparse dependency)
# ---------------------------------------------------------------------------


def cmd_preflight(
    as_json: bool = False,
    config: str = "configs/foundation/preflight.yaml",
) -> int:
    """Check foundation prerequisites (hardware_ready/hardware_not_ready)."""
    from oeis_learn.cli.foundation_preflight import main as preflight_main

    preflight_main(as_json=as_json, config_path=config)
    return 0


def cmd_conformance(
    as_json: bool = False,
    schema: str = "specs/007-experiment-foundation/contracts/artifacts.schema.json",
    artifacts: Optional[List[str]] = None,
) -> int:
    """Validate artifacts against foundation artifacts schema + data-model rules."""
    from jsonschema import Draft202012Validator

    from oeis_learn.experiments.models import FoundationValidationError, validate_artifact

    with open(schema, "r", encoding="utf-8") as f:
        schema_doc = json.load(f)
    validator = Draft202012Validator(schema_doc)

    errors: List[str] = []
    if artifacts:
        for artifact_path in artifacts:
            try:
                with open(artifact_path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
            except json.JSONDecodeError as e:
                errors.append(f"{artifact_path}: JSON parse error: {e}")
                continue

            # File may be a JSON array (e.g. schema-examples.json) or a single object
            items = raw if isinstance(raw, list) else [raw]

            # JSON shape validation
            for item in items:
                for error in validator.iter_errors(item):
                    errors.append(f"{artifact_path}: {error.message}")

                # Semantic validation (data-model rules the schema cannot express)
                try:
                    validate_artifact(item)
                except FoundationValidationError as e:
                    errors.append(f"{artifact_path}: {e}")

    if errors:
        if as_json:
            print(json.dumps({"status": "FAIL", "errors": errors}))
        else:
            for err in errors:
                print(f"FAIL: {err}")
        return 2
    else:
        if as_json:
            print(json.dumps({"status": "PASS"}))
        else:
            print("PASS: all artifacts conform")
        return 0


def cmd_freeze_cohort(
    root: str,
    as_json: bool = False,
) -> int:
    """Freeze a cohort of artifacts under root (create registry with deterministic IDs)."""
    from oeis_learn.experiments.artifacts import ArtifactRegistry, compute_file_hash

    root_path = Path(root)
    registry = ArtifactRegistry(root_path / "artifacts")

    for kind in ("visible-prompt", "candidate-result"):
        subdir = root_path / f"{kind}s"
        if subdir.exists():
            for path in sorted(subdir.glob("*.json")):
                if path.is_file():
                    with open(path, "r", encoding="utf-8") as f:
                        artifact = json.load(f)
                    registry.store(kind, artifact)

    for kind in ("checkpoint",):
        subdir = root_path / f"{kind}s"
        if subdir.exists():
            for path in sorted(subdir.glob("*.pt")):
                if path.is_file():
                    digest = compute_file_hash(path)
                    registry.store(kind, {"path": str(path), "digest": digest})

    counts = {k: len(v) for k, v in registry._by_kind.items()}
    if as_json:
        print(json.dumps({"root": str(root_path), "counts": counts}))
    else:
        print(f"Registry created at {registry.root}")
        for k, v in counts.items():
            print(f"  {k}: {v}")
    return 0


def cmd_build_pool(
    root: str,
    as_json: bool = False,
    limit: int = 1000,
) -> int:
    """Build a pool of candidate result digests for evaluation."""
    from oeis_learn.experiments.artifacts import (
        ArtifactRegistry,
        compute_canonical_digest,
    )

    registry = ArtifactRegistry(Path(root) / "artifacts")
    pool: List[str] = []
    digest_path = registry.root / "pool" / "candidates.txt"
    digest_path.parent.mkdir(parents=True, exist_ok=True)

    if digest_path.exists():
        with open(digest_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("sha256:"):
                    pool.append(line)

    subdir = registry.root.parent / "candidate-results"
    if subdir.exists():
        for path in sorted(subdir.glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                artifact = json.load(f)
            digest = compute_canonical_digest(artifact, identity_field="digest")
            if digest not in pool:
                pool.append(digest)
                if len(pool) >= limit:
                    break

    with tempfile.NamedTemporaryFile(
        mode="w", dir=digest_path.parent, delete=False
    ) as f:
        for d in pool:
            f.write(d + "\n")
        tmp_path = Path(f.name)
    tmp_path.rename(digest_path)

    if as_json:
        print(json.dumps({"pool_size": len(pool), "limit": limit}))
    else:
        print(f"Pool updated: {len(pool)} candidates")
    return 0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def dispatch(command: str, **kwargs: Any) -> int:
    """Route to the correct command implementation.

    Rejects unsupported commands with a clear reason.
    """
    if command not in _SUPPORTED_COMMANDS:
        _reject_unsupported(command)

    dispatch_map = {
        "preflight": cmd_preflight,
        "conformance": cmd_conformance,
        "freeze-cohort": cmd_freeze_cohort,
        "build-pool": cmd_build_pool,
    }

    func = dispatch_map[command]
    return func(**{k: v for k, v in kwargs.items() if v is not None})


__all__ = [
    "_SUPPORTED_COMMANDS",
    "_UNSUPPORTED_COMMANDS",
    "cmd_preflight",
    "cmd_conformance",
    "cmd_freeze_cohort",
    "cmd_build_pool",
    "dispatch",
    "_reject_unsupported",
]
