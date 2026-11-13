"""Measured checkpoint closures and pre-write limits for owned discovery storage."""

import json
import os
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path

from oeis_learn.pipeline_validation.identity import confined
from .contracts import load, verify_refs, seal

# Shared campaign accounting must fit many 25M+ tasks, not just one short
# diagnostic. This is a finite implementation ceiling; enclosing frozen storage
# and task allowances continue to apply independently.
JOURNAL_BYTES = 2 << 30

_ACTIVE = None
_LOCK = threading.RLock()


def usage(roots):
    """Logical bytes include hard-linked generations: copies may cease sharing."""
    total = files = 0
    seen = set()
    for root in map(Path, roots):
        if not root.exists():
            continue
        for path in [root, *root.rglob("*")]:
            if path.is_symlink():
                raise ValueError("storage aliases are forbidden")
            if path in seen:
                continue
            seen.add(path)
            files += 1
            if path.is_file():
                total += path.stat().st_size
    return total, files


def checkpoint_size(root, ref, *, checkpoint_root=None, research=False, optimizer_allocated=False):
    """Measure every referenced tensor, with AdamW growth and metadata headroom.

    A weights-only file cannot establish a research storage measurement. Research
    requires a full-state producer; bounds include two optimizer moments even if
    a measured initial optimizer has not yet allocated them.
    """
    from oeis_learn.generalization.contracts import artifact_refs

    raw = verify_refs(root, [ref], max_bytes=64 << 30)[0]
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeError):
        value = {}
    full = (
        value.get("version") in {"discovery-state/v1", "generalization-learner-state/v1"}
        if isinstance(value, dict)
        else False
    )
    if not full:
        if research:
            raise ValueError("full-state checkpoint storage measurement required")
        return 4 * ref["bytes"] + (64 << 10), 16
    if checkpoint_root is None:
        raise ValueError("original checkpoint closure root required")
    base = Path(root) if checkpoint_root == "." else confined(root, checkpoint_root)
    seen = {str(confined(root, ref["path"])): ref["bytes"]}
    queue = [value]
    weight_bytes = 0
    for _ in range(10000):
        if not queue:
            break
        item = queue.pop()
        if research and isinstance(item, dict) and item.get("version") == "discovery-state/v1":
            from .contracts import validate

            validate("state", item)
        if isinstance(item, dict) and item.get("version") == "generalization-learner-state/v1":
            if seal(item) != item:
                raise ValueError("checkpoint manifest identity differs")
            if research:
                from oeis_learn.generalization.state import restore

                restore(item, contract=item["contract"], cap=item["ledger"]["cap"], root=base)
            weight_bytes = max(weight_bytes, item["states"]["weights"]["bytes"])
        for child in artifact_refs(item):
            path = str(confined(base, child["path"]))
            if path in seen:
                if seen[path] != child["bytes"]:
                    raise ValueError("conflicting checkpoint closure")
                verify_refs(base, [child], max_bytes=64 << 30)
                continue
            blob = verify_refs(base, [child], max_bytes=64 << 30)[0]
            seen[path] = len(blob)
            if sum(seen.values()) > 64 << 30:
                raise ValueError("bounded checkpoint tensor closure required")
            # Only JSON manifests can introduce more references, never tensors.
            if child["path"].endswith(".json"):
                queue.append(json.loads(blob))
    else:
        raise ValueError("bounded acyclic checkpoint closure required")
    if not weight_bytes:
        raise ValueError("original checkpoint weight tensor missing")
    # Only a producer that has executed complete Adam updates may select the
    # allocated case. Historical/imported snapshots retain growth headroom.
    return sum(seen.values()) + (0 if optimizer_allocated else 2 * weight_bytes) + (64 << 10), len(seen) + 16


def projection_size(root, refs):
    """Upper bound the copied typed inputs, including nested model packages."""
    total = sum(r["bytes"] for r in refs)
    inodes = sum(len(Path(r["path"]).parts) + 2 for r in refs)
    for ref in refs:
        if not ref["path"].endswith(".json"):
            continue
        value = load(confined(root, ref["path"]), maximum=256 << 20)
        if not isinstance(value, dict) or value.get("version") != "discovery-channel/v1":
            continue
        if value.get("kind") in {"sealed_dossier", "review_evidence", "reviewer_registry"}:
            from .claims import review_references

            for child in review_references(root, value["kind"], value["data"]):
                verify_refs(root, [child], max_bytes=512 << 20)
                total += child["bytes"]
                inodes += len(Path(child["path"]).parts) + 2
            continue
        if value.get("kind") not in {
            "checkpoint",
            "recipe",
            "training_request",
            "verified_training_archive",
            "typed_training_source",
            "library",
        }:
            continue
        nested = [value["data"]["artifact"]]
        if value["kind"] == "verified_training_archive":
            from .contracts import validate, reference

            cut = validate("archive", load(confined(root, nested[0]["path"])))
            base = confined(root, nested[0]["path"]).parent
            for child in cut["payload"]["entries"]:
                rebound = reference(root, confined(base, child["path"]))
                if any(rebound[k] != child[k] for k in ("bytes", "sha256")):
                    raise ValueError("original archive closure differs")
                nested.append(rebound)
        if value["kind"] == "training_request":
            package = load(confined(root, nested[0]["path"]), maximum=256 << 20)
            if package.get("weights") is not None:
                nested.append(package["weights"])
            if package.get('full_state') is not None:
                from .import_state import inspect
                nested.extend(inspect(root,package['full_state'])[-1])
            if package.get("origin") is not None:
                from .campaign import initializer_origin

                origin = load(confined(root, package["origin"]["path"]))
                purpose = load(confined(root, origin["campaign"]["path"]))["purpose"]
                _, closure = initializer_origin(
                    root,
                    dict(
                        mode="full_state" if 'full_state' in package else "weights_only", artifact=package['full_state']['artifact'] if 'full_state' in package else package["weights"], origin=package["origin"]
                    ),
                    track="synthetic_strict",
                    purpose=purpose,
                )
                nested.extend(closure)
        verify_refs(root, list({r["path"]: r for r in nested}.values()), max_bytes=64 << 30)
        total += sum(r["bytes"] for r in nested)
        inodes += sum(len(Path(r["path"]).parts) + 1 for r in nested)
    return total, inodes


def copy_file(source, target):
    """Use a private copy-on-write inode where supported, never a hard link.

    Logical limits still reserve the full copy, including on filesystems without
    reflinks. Worker writes cannot mutate the retained source through an alias.
    """
    import errno
    import fcntl

    with writing(target, Path(source).stat().st_size, inodes=1):
        with Path(source).open('rb') as src, Path(target).open('xb') as dst:
            try:
                fcntl.ioctl(dst.fileno(), 0x40049409, src.fileno())  # Linux FICLONE
            except OSError as exc:
                if exc.errno not in {errno.EXDEV, errno.EOPNOTSUPP, errno.ENOTTY, errno.EINVAL, errno.ENOSYS}:
                    raise
                dst.truncate(0)
                dst.seek(0)
                src.seek(0)
                shutil.copyfileobj(src, dst, 1 << 20)
        return target


def journal_limit(root):
    """Original campaigns retain their frozen allowance; no implicit enlargement."""
    path=Path(root)/'journal-storage.json'
    if not path.exists():return 64<<20
    value=load(path)
    if (set(value)!={'campaign','bytes','identity'} or seal(value)!=value or
            type(value['bytes']) is not int or not 0<value['bytes']<=JOURNAL_BYTES):
        raise ValueError('bounded frozen journal storage required')
    return value['bytes']


def journal_write(root, path, value):
    from .contracts import atomic_write

    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    StorageGuard([root], max_bytes=journal_limit(root), max_inodes=8192).check(len(raw), 2)
    return atomic_write(path, value)


class StorageGuard:
    def __init__(
        self, roots, *, max_bytes, max_inodes, prior_bytes=0, prior_inodes=0, reserve_bytes=0, free_reserve_bytes=0
    ):
        self.roots = [Path(r) for r in roots]
        self.max_bytes, self.max_inodes = max_bytes, max_inodes
        self.prior_bytes, self.prior_inodes = prior_bytes, prior_inodes
        self.reserve_bytes = reserve_bytes
        self.free_reserve_bytes = free_reserve_bytes
        for n in (max_bytes, max_inodes, prior_bytes, prior_inodes, reserve_bytes, free_reserve_bytes):
            if type(n) is not int or n < 0:
                raise ValueError("finite storage partition required")

    def check(self, added_bytes=0, added_inodes=0):
        used, inodes = usage(self.roots)
        self._check(used, inodes, added_bytes, added_inodes)

    def check_total(self, total_bytes, total_inodes):
        """Preflight a total peak within these roots, including existing files.

        Prior invocations and reserves remain additional charges. Actual writes
        still use check(): their temporary allocations must fit at publication.
        """
        if any(type(n) is not int or n < 0 for n in (total_bytes, total_inodes)):
            raise ValueError("finite total storage reservation required")
        used, inodes = usage(self.roots)
        self._check(used, inodes, max(0, total_bytes-used), max(0, total_inodes-inodes))

    def _check(self, used, inodes, added_bytes, added_inodes):
        if (
            used + self.prior_bytes + added_bytes + self.reserve_bytes > self.max_bytes
            or inodes + self.prior_inodes + added_inodes > self.max_inodes
        ):
            raise TimeoutError("owned storage byte/inode reservation exhausted")
        existing = next((r for r in self.roots if r.exists()), None)
        if existing is None:
            existing = self.roots[0].parent
            while not existing.exists():
                existing = existing.parent
        if (
            added_bytes + self.reserve_bytes + self.free_reserve_bytes > shutil.disk_usage(existing).free
            or added_inodes > os.statvfs(existing).f_favail
        ):
            raise TimeoutError("free storage byte/inode reserve exhausted")

    @contextmanager
    def activate(self):
        global _ACTIVE
        previous = _ACTIVE
        self.check()
        _ACTIVE = self
        try:
            yield self
        finally:
            _ACTIVE = previous


@contextmanager
def writing(path, size, *, inodes=2):
    """Serialize check + publication across heartbeat and main worker threads."""
    with _LOCK:
        if _ACTIVE is not None and any(Path(path).is_relative_to(r) for r in _ACTIVE.roots):
            missing = 0
            parent = Path(path).parent
            while not parent.exists():
                missing += 1
                parent = parent.parent
            _ACTIVE.check(size, inodes + missing)
        yield


def preflight(size, inodes=0):
    with _LOCK:
        if _ACTIVE is not None:
            _ACTIVE.check(size, inodes)


def owned_guard(authority, *, output, state_root=None):
    c, t = authority["campaign"], authority["task"]
    storage = c["payload"]["storage"]
    prior = authority.get("storage", {})
    roots = [output] + ([state_root] if state_root is not None else [])
    return StorageGuard(
        roots,
        max_bytes=t["payload"]["caps"]["disk_bytes"],
        max_inodes=prior.get("max_inodes", storage["required_inodes"]),
        prior_bytes=prior.get("prior_bytes", 0),
        prior_inodes=prior.get("prior_inodes", 0),
        # Packaging belongs to the shared frozen reserve, not every task's
        # private partition. It must still remain physically free before writes.
        free_reserve_bytes=storage["package_bytes"],
    )


def register_invocation(root, campaign, task, journal_root, output):
    """Host-only inventory: old invocation outputs remain charged on resume.

    Each task owns a finite disk partition for all of its retained generations,
    archives and invocation outputs. The sum of partitions bounds simultaneous
    predecessors without mounting their private data in another worker.
    """
    from .contracts import digest

    root, output = Path(root), Path(output)
    index = Path(journal_root) / "storage-inventory"
    entries = []
    for path in sorted(index.glob("*.json")):
        item = load(path)
        if seal(item) != item or item["campaign"] != campaign["identity"]:
            raise ValueError("original storage inventory required")
        if item["task"] == task["identity"] and item['output'] != str(output.relative_to(root)):
            entries.append(confined(root, item["output"], exists=False))
    prior_bytes, prior_inodes = usage(entries)
    if prior_bytes >= task["payload"]["caps"]["disk_bytes"]:
        raise TimeoutError("retained task output storage exhausted")
    value = seal(
        dict(
            campaign=campaign["identity"],
            task=task["identity"],
            output=str(output.relative_to(root)),
        )
    )
    binding = index / (digest(value)[7:] + '.json')
    if binding.exists():
        if load(binding)!=value:raise ValueError('original storage output binding changed')
    else:journal_write(journal_root,binding,value)
    return dict(
        prior_bytes=prior_bytes,
        prior_inodes=prior_inodes,
        max_inodes=max(
            32,
            (
                campaign["payload"]["storage"]["required_inodes"]
                - (16384 if "task_store_bytes" in campaign["payload"]["storage"] else 0)
            )
            // len(campaign["payload"]["tasks"]),
        ),
    )


def append_json(path, value, *, maximum=64 << 20):
    path = Path(path)
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    with writing(path, len(raw), inodes=1 if not path.exists() else 0):
        if (path.stat().st_size if path.exists() else 0) + len(raw) > maximum:
            raise TimeoutError("storage journal reservation exhausted")
        with path.open("ab") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
