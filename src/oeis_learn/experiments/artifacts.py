"""Foundation artifact storage, identity, and registry (T008).

This module provides:

- Canonical artifact dataclasses (VisiblePrompt, CandidateResult, CheckpointManifest)
- Deterministic identity computation (SHA-256 digests of canonical JSON)
- Artifact validation against contracts/artifacts.schema.json and data-model.md
- Safe, deterministic artifact storage with exact final-byte hashes, atomic writes,
  acyclic references, and safe relative paths
- CLI commands: `foundation preflight`, `foundation conformance`, `foundation freeze-cohort`, `foundation build-pool`

The foundation codec `wat_body_decimal_v1` identity is preserved; legacy codec identities
(`i64_scalar_v1`, `i256x4_v1`) remain in legacy modules to avoid incompatible weight reuse.
"""

from __future__ import annotations

import hashlib
import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from jsonschema import Draft202012Validator

from oeis_learn.experiments.config import FoundationConfig, load_config
from oeis_learn.experiments.models import (
    CandidateResult,
    CheckpointManifest,
    Metric,
    VisiblePrompt,
    FoundationValidationError,
    parse_integer_text,
    validate_artifact,
)

# ---------------------------------------------------------------------------


def compute_canonical_digest(obj: Dict[str, Any], identity_field: Optional[str] = None) -> str:
    """Compute SHA-256 digest of canonical JSON representation.

    The JSON is sorted keys, compact separators, ensure_ascii=False.
    If identity_field is provided, it is omitted from the hash.
    """
    filtered = {k: v for k, v in obj.items() if k != identity_field}
    canonical = json.dumps(filtered, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 hash of file contents (final bytes only, no self-reference)."""
    with open(path, "rb") as f:
        data = f.read()
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------


@dataclass
class ArtifactPath:
    """A safe, relative path under a run/cohort root with no traversal escapes."""

    root: Path
    relpath: str

    def __post_init__(self) -> None:
        # Enforce: relative, no .., no absolute, no symlink escapes via path_norm
        if self.relpath.startswith("/") or ".." in self.relpath:
            raise ValueError(f"artifact path {self.relpath!r} is unsafe")
        full = self.root / self.relpath
        try:
            full.resolve().relative_to(self.root.resolve())
        except ValueError:
            raise ValueError(f"artifact path {self.relpath!r} escapes root")

    def as_path(self) -> Path:
        return self.root / self.relpath

    @classmethod
    def make(cls, root: Path, relpath: str) -> ArtifactPath:
        return cls(root=root, relpath=relpath)


# ---------------------------------------------------------------------------


# Note: VisiblePrompt, CandidateResult, CheckpointManifest defined in models.py
# with required semantic checks. The dataclasses there are canonical.

canonical_artifact_types = (VisiblePrompt, CandidateResult, CheckpointManifest)


# ---------------------------------------------------------------------------


class ArtifactRegistry:
    """Registry of canonical artifacts with deterministic IDs.

    Each artifact gets a canonical path like:
        visible-prompts/<digest>.json
        candidate-results/<digest>.json
        checkpoints/<digest>.pt
    """

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._by_kind: Dict[str, Set[str]] = {
            "visible-prompt": set(),
            "candidate-result": set(),
            "checkpoint": set(),
        }

    def _canonical_path(self, kind: str, digest: str) -> Path:
        """Return artifact path for kind/digest, creating subdirectory if needed."""
        subdir = {
            "visible-prompt": "visible-prompts",
            "candidate-result": "candidate-results",
            "checkpoint": "checkpoints",
        }.get(kind, kind + "s")
        path = self.root / subdir / f"{digest}.json" if kind != "checkpoint" else self.root / subdir / f"{digest}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def store(self, kind: str, artifact: Dict[str, Any]) -> ArtifactPath:
        """Store artifact and return its canonical path."""
        # Validate against schema
        validate_artifact(artifact)
        # Compute identity digest (omit identity field if present)
        digest = compute_canonical_digest(artifact, identity_field="digest")
        if kind == "checkpoint":
            digest = compute_file_hash(artifact.get("path", Path()))
        # Write canonical JSON
        path = self._canonical_path(kind, digest)
        if kind != "checkpoint":
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
                json.dump(artifact, f, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                tmp_path = Path(f.name)
            tmp_path.rename(path)
        else:
            # For checkpoints, we store the time of last checkpoint instead of re-serializing
            # The real checkpoint data lives at artifact["path"]
            checkpoint_info = {
                "checkpoint_state": artifact.get("checkpoint_state", "complete"),
                "payload_state_keys": artifact.get("payload_state_keys", []),
                "timestamp_ns": int(artifact.get("timestamp_ns", 0)),
                "digest": digest,
            }
            with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
                json.dump(checkpoint_info, f, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                tmp_path = Path(f.name)
            tmp_path.rename(path)
        # Track
        self._by_kind[kind].add(digest)
        return ArtifactPath(root=self.root, relpath=f"{path.relative_to(self.root).as_posix()}")

    def get(self, kind: str, digest: str) -> Optional[Dict[str, Any]]:
        """Retrieve artifact if stored."""
        path = self._canonical_path(kind, digest)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def lookup(self, digest: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Look up artifact by digest across all kinds."""
        for kind in self._by_kind:
            if digest in self._by_kind[kind]:
                artifact = self.get(kind, digest)
                if artifact:
                    return kind, artifact
        return None


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


import click


@click.group(name="foundation")
def foundation_cli() -> None:
    """Foundation artifact administration commands (T008)."""
    pass


@foundation_cli.command(name="preflight")
@click.option("--json", "as_json", is_flag=True, help="Output summary JSON to stdout, progress to stderr")
@click.option("--config", default="configs/foundation/preflight.yaml", help="Configuration file")
def foundation_preflight(as_json: bool, config: str) -> None:
    """Check foundation prerequisites (hardware_ready/hardware_not_ready)."""
    # Reuse existing preflight module
    from oeis_learn.cli.foundation_preflight import main as preflight_main
    preflight_main(as_json=as_json, config_path=config)


@foundation_cli.command(name="conformance")
@click.option("--json", "as_json", is_flag=True, help="Output summary JSON to stdout, progress to stderr")
@click.option("--schema", default="specs/007-experiment-foundation/contracts/artifacts.schema.json", help="Schema path")
@click.argument("artifacts", nargs=-1, type=click.Path(exists=True))
def foundation_conformance(as_json: bool, schema: str, artifacts: Tuple[str, ...]) -> None:
    """Validate artifacts against foundation artifacts schema."""
    with open(schema, "r", encoding="utf-8") as f:
        schema_doc = json.load(f)
    validator = Draft202012Validator(schema_doc)
    errors = []
    for artifact_path in artifacts:
        try:
            with open(artifact_path, "r", encoding="utf-8") as f:
                artifact = json.load(f)
            for error in validator.iter_errors(artifact):
                errors.append(f"{artifact_path}: {error.message}")
        except json.JSONDecodeError as e:
            errors.append(f"{artifact_path}: JSON parse error: {e}")
    if errors:
        if as_json:
            print(json.dumps({"status": "FAIL", "errors": errors}))
        else:
            for e in errors:
                print(f"FAIL: {e}")
        import sys

        sys.exit(2)
    else:
        if as_json:
            print(json.dumps({"status": "PASS"}))
        else:
            print("PASS: all artifacts conform")
        import sys

        sys.exit(0)


@foundation_cli.command(name="freeze-cohort")
@click.option("--json", "as_json", is_flag=True, help="Output summary JSON to stdout, progress to stderr")
@click.argument("root", type=click.Path(exists=True))
def foundation_freeze_cohort(as_json: bool, root: str) -> None:
    """Freeze a cohort of artifacts under root (create registry with deterministic IDs)."""
    root_path = Path(root)
    registry = ArtifactRegistry(root_path / "artifacts")
    # Scan for visible-prompts/candidate-results checkpoints
    for kind in ("visible-prompt", "candidate-result"):
        subdir = root_path / f"{kind}s"
        if subdir.exists():
            for path in subdir.glob("*.json"):
                if path.is_file():
                    with open(path, "r", encoding="utf-8") as f:
                        artifact = json.load(f)
                    registry.store(kind, artifact)
    for kind in ("checkpoint",):
        subdir = root_path / f"{kind}s"
        if subdir.exists():
            for path in subdir.glob("*.pt"):
                if path.is_file():
                    digest = compute_file_hash(path)
                    registry.store(kind, {"path": str(path), "digest": digest})
    if as_json:
        counts = {k: len(v) for k, v in registry._by_kind.items()}
        print(json.dumps({"root": str(root_path), "counts": counts}))
    else:
        counts = {k: len(v) for k, v in registry._by_kind.items()}
        print(f"Registry created at {registry.root}")
        for k, v in counts.items():
            print(f"  {k}: {v}")


@foundation_cli.command(name="build-pool")
@click.option("--json", "as_json", is_flag=True, help="Output summary JSON to stdout, progress to stderr")
@click.argument("root", type=click.Path(exists=True))
@click.option("--limit", default=1000, help="Maximum pool entries")
def foundation_build_pool(as_json: bool, root: str, limit: int) -> None:
    """Build a pool of candidate result digests for evaluation."""
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
    # Add any new candidate results not in pool
    subdir = registry.root.parent / "candidate-results"
    if subdir.exists():
        for path in subdir.glob("*.json"):
            with open(path, "r", encoding="utf-8") as f:
                artifact = json.load(f)
            digest = compute_canonical_digest(artifact, identity_field="digest")
            if digest not in pool:
                pool.append(digest)
                if len(pool) >= limit:
                    break
    with tempfile.NamedTemporaryFile(mode="w", dir=digest_path.parent, delete=False) as f:
        for d in pool:
            f.write(d + "\n")
        tmp_path = Path(f.name)
    tmp_path.rename(digest_path)
    if as_json:
        print(json.dumps({"pool_size": len(pool), "limit": limit}))
    else:
        print(f"Pool updated: {len(pool)} candidates")


# ---------------------------------------------------------------------------


__all__ = [
    "ArtifactRegistry",
    "ArtifactPath",
    "compute_canonical_digest",
    "compute_file_hash",
    "foundation_cli",
    "canonical_artifact_types",
]
