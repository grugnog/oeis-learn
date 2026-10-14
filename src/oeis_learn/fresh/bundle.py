"""Portable analysis bundles, with explicit omissions and verified file inventory."""

import os
import tempfile
import zipfile
from collections import Counter, deque
from pathlib import Path

from .common import canonical, confined, digest, file_hash, read, write

REPORT_NAMES = {
    "issuance.json",
    "run.json",
    "status.json",
    "budget.json",
    "budget.jsonl",
    "finalization.json",
    "preparation-report.json",
    "input-manifest.json",
    "result.json",
    "runtime.json",
    "launch.json",
    "progress.json",
    "stdout.log",
    "stderr.log",
    "metrics.jsonl",
    "early-stop.json",
    "admission-completion.json",
}


def pack(root, output, *, checkpoint=False, max_bytes=25 << 20):
    root, output = Path(root).resolve(), Path(output).resolve()
    if root == output or root in output.parents:
        raise ValueError("bundle destination must be outside the run")
    run = read(root / "run.json")
    status = read(root / "status.json")
    # Enumerate report locations only. Private role input views and truth are
    # excluded even if a caller accidentally points this at the host job root.
    candidates = [root / name for name in REPORT_NAMES if (root / name).is_file()]
    for stage in sorted((root / "roles").glob("*")):
        if "-input" in stage.name or stage.name == "workers":
            continue
        candidates += [p for p in stage.iterdir() if p.name in REPORT_NAMES and p.is_file()]
        if (stage / "checkpoints").exists():
            candidates += sorted((stage / "checkpoints").glob("*.json"))
    selected = None
    if checkpoint:
        final = (
            read(root / "finalization.json") if (root / "finalization.json").exists() else status
        )
        selected = confined(root, final["checkpoint"])
        manifest = read(selected)
        base = read(selected.parent / manifest["base"])
        blob = confined(selected.parent, base["blob_path"])
        if file_hash(blob) != base["blob_sha256"]:
            raise ValueError("selected checkpoint blob mismatch")
        candidates += [selected, selected.parent / manifest["base"], blob]
        max_bytes = 40 << 30
    payloads, included, omitted = {}, {}, []
    updates, tokens, training_seconds = 0, 0, 0.0
    training_wall_seconds = 0.0
    first_loss, last_loss = None, None
    first_observation, last_observation = None, None
    failure_classes = Counter()
    for path in sorted(set(candidates)):
        name = str(path.relative_to(root))
        if path.is_symlink() or path.stat().st_nlink != 1:
            raise ValueError(f"unsafe report artifact: {path}")
        if path.name == "result.json":
            result = read(path)
            failure_classes.update(result.get("failure_classes", {}))
        # Avoid counting the same updates from copied training stage pointers:
        # metrics live only in the actual attempt directories.
        if path.name == "metrics.jsonl":
            sampled, tail = [], deque(maxlen=100)
            with path.open() as stream:
                for line in stream:
                    import json

                    row = json.loads(line)
                    updates += 1
                    tokens += row.get("nonpadding_tokens", 0)
                    training_seconds += sum(row.get("phase_timings_ns", {}).values()) / 1e9
                    training_wall_seconds += row.get("update_wall_ns", 0) / 1e9
                    observed = row.get("observed_utc_ns", updates)
                    if first_observation is None or observed < first_observation:
                        first_observation, first_loss = observed, row.get("loss")
                    if last_observation is None or observed > last_observation:
                        last_observation, last_loss = observed, row.get("loss")
                    if len(sampled) < 100:
                        sampled.append(row)
                    else:
                        tail.append(row)
            if path.stat().st_size > 1 << 20:
                payloads[name] = b"".join(canonical(v) + b"\n" for v in sampled + list(tail))
                omitted.append(
                    {
                        "path": name,
                        "reason": "metrics sampled: first100 and last100 rows",
                        "original_sha256": file_hash(path),
                        "original_bytes": path.stat().st_size,
                    }
                )
        size = len(payloads[name]) if name in payloads else path.stat().st_size
        if path.suffix == ".log" and size > 1 << 20:
            with path.open("rb") as stream:
                stream.seek(-(1 << 20), os.SEEK_END)
                payloads[name] = stream.read()
            omitted.append(
                {
                    "path": name,
                    "reason": "log tail only",
                    "original_sha256": file_hash(path),
                    "original_bytes": size,
                }
            )
            size = len(payloads[name])
        if sum(r["bytes"] for r in included.values()) + size > max_bytes - (1 << 20):
            omitted.append(
                {
                    "path": name,
                    "reason": "bundle byte cap",
                    "original_sha256": file_hash(path),
                    "original_bytes": path.stat().st_size,
                }
            )
            payloads.pop(name, None)
            continue
        included[name] = {
            "bytes": size,
            "sha256": digest(payloads[name].decode()) if name in payloads else file_hash(path),
        }
    summary = {
        "version": "fresh-analysis/v1",
        "run_identity": run["identity"],
        "status": status,
        "attempted_update_records": updates,
        "committed_updates": status.get("completed_updates"),
        "nonpadding_tokens": tokens,
        "measured_training_phase_seconds": training_seconds,
        "measured_training_update_wall_seconds": training_wall_seconds,
        "tokens_per_update_wall_second": (
            tokens / training_wall_seconds if training_wall_seconds else None
        ),
        "tokens_per_training_phase_second": tokens / training_seconds if training_seconds else None,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "failure_classes": dict(failure_classes),
        "qualified": False,
        "spec007_release": "pending",
        "checkpoint_included": checkpoint,
        "limitations": [
            "finite diagnostic cohort",
            "attempt records may include work rolled back on recovery",
            "APU and host memory overlap; do not sum them",
        ],
    }
    payloads["analysis.json"] = canonical(summary) + b"\n"
    payloads["summary.md"] = (
        f"# Fresh training analysis\n\nStatus: {status['status']}\n\n"
        f"Attempted update records: {updates}; committed updates: {status.get('completed_updates')}\n\n"
        f"First loss: {first_loss}; last loss: {last_loss}\n\n"
        f"Measured phase throughput: {summary['tokens_per_training_phase_second']} tokens/s\n\n"
        "Diagnostic evidence; spec-007 release and historical compatibility remain pending.\n"
    ).encode()
    import hashlib

    for name, payload in payloads.items():
        if name in included or name in {"analysis.json", "summary.md"}:
            included[name] = {
                "bytes": len(payload),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }
    receipt = {
        "version": "fresh-analysis-bundle/v1",
        "files": included,
        "omissions": omitted,
        "source_run_identity": run["identity"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".bundle-", dir=output.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("bundle.json", canonical(receipt) + b"\n")
            for name in sorted(included):
                if name in payloads:
                    archive.writestr(name, payloads[name])
                else:
                    archive.write(confined(root, name), name)
        with zipfile.ZipFile(temporary) as archive:
            for name, ref in included.items():
                h = hashlib.sha256()
                with archive.open(name) as stream:
                    for chunk in iter(lambda: stream.read(1 << 20), b""):
                        h.update(chunk)
                if "sha256:" + h.hexdigest() != ref["sha256"]:
                    raise ValueError("bundle verification failed; source changed during packing")
        os.replace(temporary, output)
    finally:
        Path(temporary).unlink(missing_ok=True)
    write(
        output.with_suffix(output.suffix + ".sha256.json"),
        {"sha256": file_hash(output), "bytes": output.stat().st_size},
    )
    return {
        "path": str(output),
        "sha256": file_hash(output),
        "bytes": output.stat().st_size,
        "omissions": omitted,
    }
