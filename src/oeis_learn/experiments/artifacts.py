"""Content-addressed JSON artifacts; diagnostic fixtures are isolated from runs.

Checkpoint payload publication/recovery belongs to the checkpoint writer. This
registry stores its manifest as JSON and verifies the existing blob; it never
replaces a .pt payload with JSON or embeds a self-referential digest.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from oeis_learn.experiments.models import (
    CandidateResult,
    CheckpointManifest,
    FoundationValidationError,
    VisiblePrompt,
    check_digest,
    validate_artifact,
)


def load_json(text: str | bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise FoundationValidationError(f"duplicate JSON member {key!r}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise FoundationValidationError(f"non-JSON constant {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant)


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def compute_canonical_digest(obj: dict, identity_field: str | None = None) -> str:
    """Logical identity; file identities must always hash every final byte."""
    if identity_field is not None:
        obj = {k: v for k, v in obj.items() if k != identity_field}
    return "sha256:" + hashlib.sha256(canonical_bytes(obj)).hexdigest()


def compute_file_hash(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class ArtifactPath:
    root: Path
    relpath: str

    def __post_init__(self) -> None:
        self.as_path()

    def as_path(self) -> Path:
        # Require portable canonical spelling before filesystem resolution.
        if not isinstance(self.relpath, str):
            raise FoundationValidationError("artifact path must be text")
        p = PurePosixPath(self.relpath)
        if (
            not self.relpath
            or "\\" in self.relpath
            or ":" in self.relpath
            or "\x00" in self.relpath
            or p.is_absolute()
            or ".." in p.parts
            or p.as_posix() != self.relpath
            or self.relpath == "."
        ):
            raise FoundationValidationError(f"unsafe artifact path {self.relpath!r}")
        full = Path(self.root) / self.relpath
        try:
            full.resolve().relative_to(Path(self.root).resolve())
        except ValueError as exc:
            raise FoundationValidationError("artifact path escapes root") from exc
        return full

    @classmethod
    def make(cls, root: Path, relpath: str) -> ArtifactPath:
        return cls(root, relpath)


def atomic_write(path: ArtifactPath, payload: bytes) -> None:
    """Publish immutable final bytes, flush/fsync, and clean up failed writes.

    The run root has one trusted controller writer, as required by the contract.
    Candidate workers must not have write access to its directories.
    """
    target = path.as_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    path.as_path()  # Recheck containment after directory creation.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication is atomic and cannot overwrite an existing ID.
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise FoundationValidationError(
                    "immutable artifact already exists with different bytes"
                )
        fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def verify_reference_graph(root: Path, references: dict[str, dict]) -> None:
    """Resolve digest/path references, reject missing nodes, tampering and cycles.

    Each entry has exactly `path` and `references` (a list of dependency digests).
    It is supplied by the immutable run manifest, not inferred from filenames.
    """
    active, done = set(), set()

    def visit(digest: str) -> None:
        check_digest(digest, "reference")
        if digest in active:
            raise FoundationValidationError("cyclic artifact references")
        if digest in done:
            return
        node = references.get(digest)
        if (
            not isinstance(node, dict)
            or set(node) != {"path", "references"}
            or not isinstance(node["references"], list)
        ):
            raise FoundationValidationError("missing or malformed artifact reference")
        path = ArtifactPath(root, node["path"]).as_path()
        if not path.is_file() or compute_file_hash(path) != digest:
            raise FoundationValidationError(f"missing or corrupt reference {digest}")
        active.add(digest)
        for child in node["references"]:
            visit(child)
        active.remove(digest)
        done.add(digest)

    for digest in references:
        visit(digest)


def _digests(obj: Any) -> set[str]:
    if isinstance(obj, dict):
        return set().union(*(_digests(v) for v in obj.values()))
    if isinstance(obj, list):
        return set().union(*(_digests(v) for v in obj))
    return {obj} if isinstance(obj, str) and obj.startswith("sha256:") else set()


_KINDS = {
    "visible-prompt": "visible_prompt",
    "candidate-result": "candidate_result",
    "checkpoint": "checkpoint_manifest",
}


class ArtifactRegistry:
    """Read/write exact-byte artifacts with verified provenance on consumption.

    Diagnostic mode is explicit and writes exclusively below `diagnostics/`.
    It validates shape/semantics but makes no executed-evidence claim. A run
    registry requires the expected profiles plus a complete reference graph.
    """

    def __init__(
        self,
        root: Path,
        *,
        diagnostic: bool = False,
        references: dict | None = None,
        expected_profiles: dict | None = None,
    ):
        self.run_root = Path(root)
        self.root = self.run_root / "diagnostics" if diagnostic else self.run_root
        self.diagnostic = diagnostic
        self.references = references if references is not None else {}
        self.expected_profiles = expected_profiles

    def _path(self, kind: str, digest: str) -> ArtifactPath:
        if kind not in _KINDS:
            raise FoundationValidationError(f"unknown registry kind {kind!r}")
        check_digest(digest, "artifact")
        return ArtifactPath(self.root, f"{kind}s/{digest[7:]}.json")

    def _validate(self, kind: str, artifact: dict) -> None:
        validate_artifact(artifact)
        if kind not in _KINDS or artifact["kind"] != _KINDS[kind]:
            raise FoundationValidationError("registry kind does not match artifact kind")
        if self.diagnostic:
            if kind == "candidate-result" and artifact["purpose"] != "conformance":
                raise FoundationValidationError("diagnostic results require conformance purpose")
            return
        if self.expected_profiles is None:
            raise FoundationValidationError("run storage requires expected profile identities")
        required = (
            {"language_profile"}
            if kind == "visible-prompt"
            else (
                {"language_profile", "resource_profile"}
                if kind == "candidate-result"
                else {
                    "contract_sha256",
                    "pool_sha256",
                    "codec_sha256",
                    "runtime_sha256",
                    "effective_config_sha256",
                }
            )
        )
        if not required <= self.expected_profiles.keys():
            raise FoundationValidationError("incomplete expected profile identities")
        validate_artifact(
            artifact, expected_profiles={k: self.expected_profiles[k] for k in required}
        )
        if kind == "candidate-result" and artifact["purpose"] != "model":
            raise FoundationValidationError("conformance results are not run evidence")
        if "sha256:" + "0" * 64 in _digests(artifact):
            raise FoundationValidationError("fixture digest is not run evidence")
        if not _digests(artifact) <= self.references.keys():
            raise FoundationValidationError("artifact has unresolved references")
        verify_reference_graph(self.run_root, self.references)
        if kind == "checkpoint":
            blob = ArtifactPath(self.run_root, artifact["blob_path"]).as_path()
            if not blob.is_file() or compute_file_hash(blob) != artifact["blob_sha256"]:
                raise FoundationValidationError("checkpoint blob missing or corrupt")

    def store(self, kind: str, artifact: dict) -> ArtifactPath:
        self._validate(kind, artifact)
        payload = canonical_bytes(artifact)
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if digest in _digests(artifact):
            raise FoundationValidationError("self-referential artifact")
        path = self._path(kind, digest)
        atomic_write(path, payload)
        return path

    def get(self, kind: str, digest: str) -> dict | None:
        path = self._path(kind, digest).as_path()
        if not path.exists():
            return None
        payload = path.read_bytes()
        if "sha256:" + hashlib.sha256(payload).hexdigest() != digest:
            raise FoundationValidationError("stored artifact digest mismatch")
        artifact = load_json(payload)
        self._validate(kind, artifact)
        return artifact

    def lookup(self, digest: str) -> tuple[str, dict] | None:
        for kind in _KINDS:
            artifact = self.get(kind, digest)
            if artifact is not None:
                return kind, artifact
        return None


canonical_artifact_types = (VisiblePrompt, CandidateResult, CheckpointManifest)
