"""Bounded immutable evidence handoff with fresh G0–G10 replay.

Archive completion is independent of release qualification. An incomplete archive
preserves pending/failed gates; it never authorizes experiments. Only enumerated,
hashed files are copied, without following symlinks or discovering other sources.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile

from oeis_learn.experiments.artifacts import (
    ArtifactPath,
    atomic_write,
    canonical_bytes,
    compute_canonical_digest as digest,
    compute_file_hash,
    load_json,
)
from oeis_learn.experiments.models import check_digest

FILE_INPUTS = frozenset(
    {
        "checkpoint",
        "baseline_bundle",
        "g7",
        "g8",
        "g9",
        "wheel_junit",
        "phase10_junit",
        "role_junit",
        "dispositions",
    }
)
MAX_FILES = 4096
MAX_BYTES = 40 << 30


def _file(root, path):
    file = ArtifactPath(root, path).as_path()
    if any(p.is_symlink() for p in (file, *file.parents) if p != Path(root).parent):
        raise ValueError("handoff symlinks are forbidden")
    if not file.is_file():
        raise ValueError("missing handoff file: " + path)
    return file


def _directory(root, path):
    directory = Path(root) if path == "." else ArtifactPath(root, path).as_path()
    if not directory.is_dir():
        raise ValueError("missing evidence directory: " + path)
    return directory


def validate_inputs(manifest, root):
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"version", "files", "readiness"}
        or manifest["version"] != "handoff-inputs/v1"
        or not isinstance(manifest["files"], list)
        or not 1 <= len(manifest["files"]) <= MAX_FILES
        or not isinstance(manifest["readiness"], dict)
        or set(manifest["readiness"]) - (FILE_INPUTS | {"measurement_root", "release_source"})
    ):
        raise ValueError("closed bounded handoff input manifest required")
    seen, total = set(), 0
    for entry in manifest["files"]:
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError("hashed handoff file reference required")
        file = _file(root, entry["path"])
        if entry["path"] in seen:
            raise ValueError("duplicate handoff file")
        seen.add(entry["path"])
        check_digest(entry["sha256"], "handoff file")
        total += file.stat().st_size
        if total > MAX_BYTES:
            raise ValueError("handoff archive exceeds 40 GiB quota")
        if compute_file_hash(file) != entry["sha256"]:
            raise ValueError("handoff file content mismatch: " + entry["path"])
    for key, path in manifest["readiness"].items():
        if key in FILE_INPUTS:
            if not isinstance(path, str) or path not in seen:
                raise ValueError("readiness file absent from handoff inventory: " + key)
        elif key == "measurement_root":
            _directory(root, path)
        else:
            if not isinstance(path, dict) or set(path) != {
                "cohort",
                "archive",
                "holdouts",
                "snapshot",
            }:
                raise ValueError("release source requires cohort/archive/holdouts/snapshot")
            for name in ("cohort", "archive"):
                _directory(root, path[name])
            if path["holdouts"] not in seen or path["snapshot"] not in seen:
                raise ValueError("release archive holdouts/snapshot absent from inventory")
    return total


def _readiness(manifest, root):
    from oeis_learn.evaluation.readiness import release_readiness

    kwargs = {}
    for key, path in manifest["readiness"].items():
        if key in ("baseline_bundle", "g7", "g8", "g9"):
            file = _file(root, path)
            kwargs[key] = (load_json(file.read_bytes()), file.parent)
        elif key == "checkpoint":
            from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint

            kwargs[key] = load_foundation_checkpoint(_file(root, path))
        elif key == "measurement_root":
            kwargs[key] = _directory(root, path)
        elif key == "release_source":
            from oeis_learn.discovery.program_archive import ProgramArchive
            from oeis_learn.experiments.learning_profiles import load_holdouts

            archive = ProgramArchive(
                _directory(root, path["archive"]),
                holdouts=load_holdouts(load_json(_file(root, path["holdouts"]).read_bytes())),
            )
            kwargs[key] = (
                _directory(root, path["cohort"]),
                archive,
                load_json(_file(root, path["snapshot"]).read_bytes()),
            )
        else:
            kwargs[key] = _file(root, path)
    report = release_readiness(**kwargs)

    # Failure diagnostics may name a staging directory. Preserve their meaning
    # while keeping the immutable report portable after atomic publication.
    def portable(value):
        if isinstance(value, str):
            return value.replace(str(Path(root).resolve()), "<evidence>")
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        return value

    return portable(report)


def _report(manifest, root, total):
    from oeis_learn.evaluation.experiment_runner import implementation_identity

    report = dict(
        version="handoff/v1",
        implementation_sha256=implementation_identity(),
        input_sha256=digest(manifest),
        files=len(manifest["files"]),
        bytes=total,
        readiness=_readiness(manifest, root),
        deferred_scope=[
            "native LODA runtime",
            "paired language comparison",
            "conditional transcoding",
        ],
        archive_is_authorization=False,
    )
    report["report_sha256"] = digest(report)
    return report


def replay_handoff(output):
    root = Path(output)
    manifest = load_json((root / "inputs.json").read_bytes())
    total = validate_inputs(manifest, root / "evidence")
    saved = load_json((root / "handoff.json").read_bytes())
    report = _report(manifest, root / "evidence", total)
    if saved != report:
        raise ValueError("handoff report/current implementation or replay mismatch")
    return report


def archive_handoff(manifest, *, input_root, output):
    root, output = Path(input_root).resolve(), Path(output).absolute()
    if any(p.is_symlink() for p in (output, *output.parents)):
        raise ValueError("handoff output symlinks are forbidden")
    total = validate_inputs(manifest, root)
    if output.exists():
        existing = replay_handoff(output)
        if existing["input_sha256"] != digest(manifest):
            raise ValueError("immutable handoff already exists with different inputs")
        return existing
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".handoff-", dir=output.parent) as temporary:
        stage = Path(temporary) / "archive"
        evidence = stage / "evidence"
        evidence.mkdir(parents=True)
        copied = 0
        for entry in manifest["files"]:
            destination = ArtifactPath(evidence, entry["path"]).as_path()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with _file(root, entry["path"]).open("rb") as source, destination.open("xb") as target:
                while chunk := source.read(1 << 20):
                    copied += len(chunk)
                    if copied > MAX_BYTES:
                        raise ValueError("handoff copy exceeds 40 GiB quota")
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
        # Compute from the copied evidence, so unlisted dependencies cannot authorize.
        validate_inputs(manifest, evidence)
        report = _report(manifest, evidence, total)
        atomic_write(ArtifactPath(stage, "inputs.json"), canonical_bytes(manifest))
        atomic_write(ArtifactPath(stage, "handoff.json"), canonical_bytes(report))
        replay_handoff(stage)
        for directory in sorted((p for p in stage.rglob("*") if p.is_dir()), reverse=True):
            fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        os.rename(stage, output)
        fd = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    args = parser.parse_args()
    print(canonical_bytes(replay_handoff(args.archive)).decode())


if __name__ == "__main__":
    main()
