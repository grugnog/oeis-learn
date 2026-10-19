"""Bounded reports-only exports, with core records reserved before detail."""

import json
import os
from pathlib import Path
import tempfile
import zipfile
from .common import canonical, read, file_hash, confined

CORE = ("run.json", "status.json", "summary.json", "checkpoint-identity.json", "comparison.json")
OPTIONAL_CORE = (
    "input-manifest.json",
    "final-plan.json",
    "lineage.json",
    "final-study.json",
    "analysis.json",
    "budget.json",
)


def pack(root, output, *, max_bytes=64 << 20):
    root = Path(root).resolve()
    output = Path(output).resolve()
    if max_bytes < 16 << 20:
        raise ValueError("bigint bundle minimum is16 MiB")
    if root == output or root in output.parents:
        raise ValueError("bundle output must be outside run")
    included = {}
    omitted = []
    payloads = {}
    used = 0
    for name in (*CORE, *OPTIONAL_CORE):
        path = confined(root, name)
        if not path.exists() and name in OPTIONAL_CORE:
            continue
        if not path.is_file():
            raise ValueError("required core record missing: " + name)
        if path.stat().st_size > 2 << 20:
            raise ValueError("bounded core record exceeds2 MiB: " + name)
        payload = path.read_bytes()
        body = json.loads(payload)
        if (
            read(root / "status.json").get("final_access", "sealed") == "sealed"
            and name not in {"run.json", "input-manifest.json"}
            and _contains_private(body)
        ):
            raise ValueError("private/final payload in sealed report")
        payloads[name] = payload
        included[name] = dict(bytes=len(payload), sha256=file_hash(path))
        used += len(payload)
    if used > 8 << 20:
        raise ValueError("core records exceed8 MiB reserve")
    # Only deliberate report chunks are eligible; role inputs/truth/weights never enter.
    for number, path in enumerate(sorted((root / "details").glob("*.json"))):
        if number >= 4096:
            raise ValueError("detail/omission ledger cap exceeded")
        path = confined(root, str(path.relative_to(root)))
        name = str(path.relative_to(root))
        size = path.stat().st_size
        row = dict(path=name, bytes=size, sha256=file_hash(path))
        if size > 1 << 20 or used + size + 8 * 1024 > max_bytes:
            omitted.append(dict(row, reason="detail cap" if size > 1 << 20 else "bundle cap"))
            continue
        body = path.read_bytes()
        if (
            _contains_private(json.loads(body))
            and read(root / "status.json").get("final_access", "sealed") == "sealed"
        ):
            omitted.append(dict(row, reason="private/final field in sealed report"))
            continue
        payloads[name] = body
        included[name] = {k: v for k, v in row.items() if k != "path"}
        used += size
        if len(omitted) > 4096:
            raise ValueError("omission ledger cap exceeded")
    manifest = dict(
        version="bigint-bundle/v1",
        complete_core=True,
        included=included,
        omitted=omitted,
        max_bytes=max_bytes,
        core_reserve_bytes=8 << 20,
        excluded=["private inputs", "truth", "weights", "final outcomes before authorization"],
    )
    payloads["bundle-manifest.json"] = canonical(manifest) + b"\n"
    if sum(map(len, payloads.values())) > max_bytes:
        raise ValueError("mandatory manifest exceeds bundle cap")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".bundle-", dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(name, "w", zipfile.ZIP_DEFLATED) as archive:
            for key, body in sorted(payloads.items()):
                archive.writestr(key, body)
        if Path(name).stat().st_size > max_bytes:
            raise ValueError("zip exceeds bundle cap")
        os.replace(name, output)
    finally:
        Path(name).unlink(missing_ok=True)
    return dict(
        path=str(output),
        sha256=file_hash(output),
        bytes=output.stat().st_size,
        complete_core=True,
        omitted=len(omitted),
    )


def export_checkpoint(root, output):
    """Explicit, separately named complete checkpoint export, never part of reports."""
    root = Path(root)
    output = Path(output)
    selected = read(root / "checkpoint-identity.json")["selected"]
    manifest = confined(root / "checkpoints", selected["path"])
    if file_hash(manifest) != selected["sha256"]:
        raise ValueError("checkpoint manifest mismatch")
    files = {
        "run.json": root / "run.json",
        "selected.json": root / "selected.json",
        "checkpoint-identity.json": root / "checkpoint-identity.json",
        "checkpoints/" + selected["path"]: manifest,
    }
    for ref in read(manifest)["files"].values():
        path = confined(root / "checkpoints", ref["path"])
        if file_hash(path) != ref["sha256"]:
            raise ValueError("checkpoint blob mismatch")
        files["checkpoints/" + ref["path"]] = path
    if sum(p.stat().st_size for p in files.values()) > 2 << 30:
        raise ValueError("explicit checkpoint export exceeds2 GiB")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".checkpoint-", dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as archive:
            for name, path in sorted(files.items()):
                archive.write(path, name)
        os.replace(temp, output)
    finally:
        Path(temp).unlink(missing_ok=True)
    return dict(
        path=str(output),
        bytes=output.stat().st_size,
        sha256=file_hash(output),
        explicit_checkpoint=True,
    )


def _contains_private(value):
    if isinstance(value, dict):
        return any(
            k in {"truth", "hidden_values", "final_values", "final_outcomes"}
            or k == "final_access"
            and v == "exposed"
            or _contains_private(v)
            for k, v in value.items()
        )
    return isinstance(value, list) and any(_contains_private(v) for v in value)
