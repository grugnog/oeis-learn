"""Pinned OEIS Git/LFS import. No per-sequence HTTP requests or inferred metadata."""

import hashlib
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import tempfile
import time
from collections import Counter
from contextlib import closing
from pathlib import Path

from .common import digest, file_hash, lock, read, write

REPOSITORY = "https://github.com/oeis/oeisdata.git"
INTEGER = re.compile(r"[+-]?\d+\Z")
AID = re.compile(r"A\d{6}\Z")


def git(root, *args, capture=False):
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    ).stdout


def metadata(payload, aid, locator):
    if not AID.fullmatch(aid):
        raise ValueError("invalid OEIS ID")
    fields = {}
    for line in payload.splitlines():
        match = re.match(r"%([A-Za-z]) (A\d{6})(?: (.*))?$", line)
        if not match:
            if line.strip():
                raise ValueError("malformed internal-format line")
            continue
        if match[2] != aid:
            raise ValueError("foreign OEIS ID in entry")
        if match[1] in {"O", "K"}:
            if match[1] in fields:
                raise ValueError("duplicate offset/keywords")
            fields[match[1]] = match[3] or ""
    if set(fields) != {"O", "K"} or not fields["K"]:
        raise ValueError("authoritative offset and keywords required")
    offset = fields["O"].split(",")
    if len(offset) != 2 or any(not INTEGER.fullmatch(v.strip()) for v in offset):
        raise ValueError("malformed explicit offset")
    keywords = fields["K"].split(",")
    if any(not re.fullmatch(r"[a-z]+", k) for k in keywords):
        raise ValueError("malformed keywords")
    return {
        "offset": int(offset[0]),
        "keywords": keywords,
        "finite": "fini" in keywords,
        "table": bool({"tabl", "tabf"} & set(keywords)),
        "locator": locator,
    }


def pointer(path):
    if path.stat().st_size > 1024:
        return None
    value = path.read_bytes()
    match = re.fullmatch(
        rb"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n",
        value,
    )
    return {"sha256": "sha256:" + match[1].decode(), "bytes": int(match[2])} if match else None


def blob_hash(payload):
    return hashlib.sha1(b"blob " + str(len(payload)).encode() + b"\0" + payload).hexdigest()


def verified_object(path, row):
    return (
        path.is_file()
        and not path.is_symlink()
        and path.stat().st_size == row["bytes"]
        and file_hash(path) == row["sha256"]
    )


def recover_object(repo, commit, row):
    """Retry an unresolved pinned object via LFS, never a per-sequence HTTP API."""
    path = repo / row["path"]
    expected = {"sha256": row["sha256"], "bytes": row["bytes"]}
    if path.is_symlink() or (path.exists() and pointer(path) != expected):
        raise ValueError(
            f"LFS object verification failed; refusing to overwrite changed data: {path}"
        )
    raw = f"version https://git-lfs.github.com/spec/v1\noid {row['sha256']}\nsize {row['bytes']}\n"
    if git(repo, "show", f"{commit}:{row['path']}", capture=True) != raw:
        raise ValueError(f"LFS recovery plan differs from pinned Git pointer: {path}")
    print(f"RECOVER LFS object {row['path']}: {row['sha256']}, {row['bytes']} bytes", flush=True)
    # Smudge requests exactly this OID, independent of ref/path scanning. Clear
    # skip settings so a successful exit cannot silently stand for a pointer.
    env = dict(os.environ)
    env.pop("GIT_LFS_SKIP_SMUDGE", None)
    env.pop("GIT_LFS_SKIP_DOWNLOAD_ERRORS", None)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".lfs-recovery-", delete=False) as out:
        temporary = Path(out.name)
        process = None
        try:
            process = subprocess.Popen(
                [
                    "git",
                    "-C",
                    str(repo),
                    "-c",
                    "lfs.fetchinclude=",
                    "-c",
                    "lfs.fetchexclude=",
                    "-c",
                    "lfs.skipdownloaderrors=false",
                    "lfs",
                    "smudge",
                    "--",
                    row["path"],
                ],
                stdin=subprocess.PIPE,
                stdout=out,
                env=env,
                start_new_session=True,
            )
            process.stdin.write(raw.encode())
            process.stdin.close()
            deadline = time.monotonic() + 120
            while process.poll() is None:
                if temporary.stat().st_size > row["bytes"]:
                    raise ValueError("LFS recovery exceeded expected object size")
                if time.monotonic() >= deadline:
                    raise TimeoutError("LFS recovery exceeded 120 seconds")
                time.sleep(0.05)
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, process.args)
            if not verified_object(temporary, row):
                raise ValueError("LFS recovery returned a pointer or an incorrect size/hash")
            os.fchmod(out.fileno(), 0o644)
            os.fsync(out.fileno())
            os.replace(temporary, path)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            raise ValueError(
                f"LFS object recovery failed: {row['path']} {row['sha256']} "
                f"({row['bytes']} bytes): {exc}; pinned cache preserved"
            ) from exc
        finally:
            if process is not None:
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                process.wait(timeout=5)
            temporary.unlink(missing_ok=True)


def indexed_terms(path, offset, *, horizon=120):
    """Validate the whole file; retain only a bounded indexed horizon."""
    values = {}
    previous = None
    with Path(path).open() as stream:
        for line in stream:
            if len(line) > 16384:
                raise ValueError("oversized b-file line")
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if (
                len(parts) != 2
                or any(not INTEGER.fullmatch(v) for v in parts)
                or len(parts[1]) > 4096
            ):
                raise ValueError("malformed indexed b-file; unresolved LFS pointer is not data")
            index, value = int(parts[0]), int(parts[1])
            if previous is not None and index <= previous:
                raise ValueError("duplicate or unordered b-file index")
            previous = index
            if offset <= index < offset + horizon:
                values[index] = str(value)
    return values


def fetch(config, cache):
    cache = Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    with lock(cache / "fetch.lock"):
        repo = cache / "oeisdata"
        if not repo.exists():
            subprocess.run(
                [
                    "git",
                    "-c",
                    "filter.lfs.smudge=",
                    "-c",
                    "filter.lfs.required=false",
                    "clone",
                    "--depth",
                    "1",
                    "--no-checkout",
                    REPOSITORY,
                    str(repo),
                ],
                check=True,
            )
        if git(repo, "remote", "get-url", "origin", capture=True).strip() != REPOSITORY:
            raise ValueError("unexpected bulk repository remote")
        pin_path = cache / "snapshot.json"
        if pin_path.exists():
            pin = read(pin_path)
            if config.get("revision") not in (None, pin["commit"]):
                raise ValueError("snapshot already pinned; use a new cache for a new revision")
        else:
            revision = config.get("revision")
            if revision:
                if not re.fullmatch(r"[0-9a-f]{40}", revision):
                    raise ValueError("explicit revision must be a Git commit SHA")
                git(repo, "fetch", "--depth", "1", "origin", revision)
                git(
                    repo,
                    "-c",
                    "filter.lfs.smudge=",
                    "-c",
                    "filter.lfs.required=false",
                    "checkout",
                    "--detach",
                    "FETCH_HEAD",
                )
            else:
                git(
                    repo,
                    "-c",
                    "filter.lfs.smudge=",
                    "-c",
                    "filter.lfs.required=false",
                    "checkout",
                    "--detach",
                    "HEAD",
                )
            pin = {
                "repository": REPOSITORY,
                "commit": git(repo, "rev-parse", "HEAD", capture=True).strip(),
                "exported_at": (repo / "time.txt").read_text().strip(),
            }
            write(pin_path, pin, immutable=True)
        if git(repo, "rev-parse", "HEAD", capture=True).strip() != pin["commit"]:
            raise ValueError("bulk checkout moved away from pinned revision")
        write(cache / "fetch-config.json", config, immutable=True)
        plan_path = cache / "fetch-plan.json"
        if plan_path.exists():
            plan = read(plan_path)
        else:
            candidates, exclusions = [], Counter()
            tree = {}
            for entry in git(
                repo, "ls-tree", "-r", "-z", pin["commit"], "seq", "files", capture=True
            ).split("\0"):
                if not entry:
                    continue
                fields, name = entry.split("\t", 1)
                mode, kind, oid = fields.split()
                if mode == "100644" and kind == "blob":
                    tree[name] = oid
            for seq in sorted((repo / "seq").glob("A*/*.seq")):
                aid = seq.stem
                try:
                    if seq.stat().st_size > 4 << 20:
                        raise ValueError("entry size cap")
                    seq_bytes = seq.read_bytes()
                    if tree.get(str(seq.relative_to(repo))) != blob_hash(seq_bytes):
                        raise ValueError("entry differs from pinned Git tree")
                    meta = metadata(
                        seq_bytes.decode(),
                        aid,
                        f"{REPOSITORY}@{pin['commit']}:{seq.relative_to(repo)}",
                    )
                    if meta["finite"] or meta["table"]:
                        exclusions["finite_or_table"] += 1
                        continue
                    name = f"files/{aid[:4]}/b{aid[1:]}.txt"
                    # Read the pointer from Git even when this working file was already fetched.
                    source_path = repo / name
                    if not source_path.exists():
                        raise ValueError("missing b-file pointer")
                    cached_pointer = pointer(source_path)
                    raw = (
                        f"version https://git-lfs.github.com/spec/v1\noid {cached_pointer['sha256']}\nsize {cached_pointer['bytes']}\n"
                        if cached_pointer
                        else git(repo, "show", f"{pin['commit']}:{name}", capture=True)
                    )
                    if tree.get(name) != blob_hash(raw.encode()):
                        raise ValueError("LFS pointer differs from pinned Git tree")
                    match = re.fullmatch(
                        r"version https://git-lfs.github.com/spec/v1\noid sha256:([0-9a-f]{64})\nsize (\d+)\n",
                        raw,
                    )
                    if not match:
                        raise ValueError("missing LFS b-file pointer")
                    candidates.append(
                        {
                            "aid": aid,
                            "path": name,
                            "sha256": "sha256:" + match[1],
                            "bytes": int(match[2]),
                            "metadata": meta,
                            "entry_sha256": file_hash(seq),
                        }
                    )
                except (ValueError, OSError, UnicodeError, subprocess.CalledProcessError) as exc:
                    exclusions[type(exc).__name__ + ":" + str(exc).splitlines()[0][:120]] += 1
            candidates.sort(key=lambda row: digest([config["seed"], row["aid"]]))
            selected, total = [], 0
            for row in candidates:
                if len(selected) >= config["max_records"]:
                    exclusions["record_cap"] += 1
                elif total + row["bytes"] > config["max_download_bytes"]:
                    exclusions["download_byte_cap"] += 1
                else:
                    selected.append(row)
                    total += row["bytes"]
            plan = {
                "version": "oeis-bulk-plan/v1",
                "snapshot": pin,
                "config": config,
                "selected": selected,
                "bytes": total,
                "exclusions": dict(exclusions),
                "scope": "bounded diagnostic snapshot subset",
            }
            write(plan_path, plan, immutable=True)
        if not plan["selected"]:
            raise ValueError("no eligible b-files in bulk snapshot")
        git(repo, "lfs", "version")
        # This cache only fetches/checks out pinned objects; it never pushes.
        # Configure filters without installing/updating pre-push hooks, which
        # may belong to another LFS version or an existing core.hooksPath.
        git(repo, "lfs", "install", "--local", "--skip-repo")
        for start in range(0, len(plan["selected"]), 100):
            batch = plan["selected"][start : start + 100]
            missing = [r for r in batch if not verified_object(repo / r["path"], r)]
            if missing:
                needed = sum(r["bytes"] for r in missing)
                if shutil.disk_usage(cache).free < config["free_disk_floor_bytes"] + 2 * needed:
                    raise OSError("bulk download would consume the reserved disk floor")
                names = [r["path"] for r in missing]
                print(
                    f"FETCH bulk batch {start // 100 + 1}: {len(names)} files, {needed} bytes",
                    flush=True,
                )
                git(
                    repo,
                    "-c",
                    "lfs.fetchrecentalways=false",
                    "lfs",
                    "fetch",
                    "--exclude=",
                    "--include=" + ",".join(names),
                    "origin",
                    pin["commit"],
                )
                git(repo, "lfs", "checkout", *names)
            for row in batch:
                path = repo / row["path"]
                if not verified_object(path, row):
                    recover_object(repo, pin["commit"], row)
        write(
            cache / "fetch-result.json",
            {"plan_sha256": file_hash(plan_path), "complete": True, "files": len(plan["selected"])},
        )
        return plan


def import_snapshot(cache, output):
    cache, output = Path(cache), Path(output)
    plan = read(cache / "fetch-plan.json")
    if read(cache / "fetch-result.json")["plan_sha256"] != file_hash(cache / "fetch-plan.json"):
        raise ValueError("bulk fetch receipt mismatch")
    repo = cache / "oeisdata"
    output.mkdir(parents=True, exist_ok=True)
    records, exclusions, horizons = [], Counter(), Counter()
    # Connection.__exit__ commits/rolls back but does not close SQLite's files.
    with (
        lock(output / "import.lock"),
        closing(sqlite3.connect(output / "index.sqlite")) as db,
        db,
    ):
        db.execute(
            "CREATE TABLE IF NOT EXISTS entries (aid TEXT PRIMARY KEY, source_sha TEXT, status TEXT, terms INTEGER)"
        )
        for row in plan["selected"]:
            values = {}
            path = repo / row["path"]
            if file_hash(path) != row["sha256"]:
                raise ValueError(f"changed bulk b-file: {path}")
            seq = repo / "seq" / row["aid"][:4] / (row["aid"] + ".seq")
            if file_hash(seq) != row["entry_sha256"]:
                raise ValueError("changed authoritative metadata")
            try:
                values = indexed_terms(path, row["metadata"]["offset"])
                for horizon in (20, 50, 100, 120):
                    indices = range(row["metadata"]["offset"], row["metadata"]["offset"] + horizon)
                    complete = all(i in values for i in indices)
                    horizons[f"complete_{horizon}"] += complete
                    horizons[f"signed256_{horizon}"] += complete and all(
                        -(1 << 255) <= int(values[i]) < 1 << 255 for i in indices
                    )
                if not all(row["metadata"]["offset"] + i in values for i in range(100)):
                    raise ValueError("incomplete100")
                aid = row["aid"]
                names = (f"terms/{aid}.txt", f"metadata/{aid}.json", f"raw/{aid}.seq")
                write(
                    output / names[0],
                    "".join(f"{i} {v}\n" for i, v in sorted(values.items())).encode(),
                    immutable=True,
                )
                write(output / names[1], row["metadata"], immutable=True)
                write(output / names[2], seq.read_bytes(), immutable=True)
                records.append(
                    {
                        "record_id": "OEIS:" + aid,
                        "terms": {"path": names[0], "sha256": file_hash(output / names[0])},
                        "metadata": {"path": names[1], "sha256": file_hash(output / names[1])},
                    }
                )
                status = "imported"
            except (ValueError, UnicodeError) as exc:
                status = str(exc)
                exclusions[status] += 1
            db.execute(
                "INSERT OR REPLACE INTO entries VALUES (?,?,?,?)",
                (row["aid"], row["sha256"], status, len(values)),
            )
        db.commit()
        if not records:
            raise ValueError("bulk snapshot has no complete indexed100 records")
        write(
            output / "manifest.json",
            {
                "version": "indexed-source/v1",
                "release": f"{REPOSITORY}@{plan['snapshot']['commit']}",
                "records": records,
            },
            immutable=True,
        )
        report = {
            "version": "oeis-bulk-import/v1",
            "snapshot": plan["snapshot"],
            "plan_sha256": file_hash(cache / "fetch-plan.json"),
            "records": len(records),
            "selected": len(plan["selected"]),
            "exclusions": dict(exclusions),
            "horizons": dict(horizons),
            "raw_entry_hashes": {r["aid"]: r["entry_sha256"] for r in plan["selected"]},
            "license": "CC BY-SA 4.0; OEIS Foundation",
            "diagnostic": True,
        }
        write(output / "import-report.json", report, immutable=True)
        return report
