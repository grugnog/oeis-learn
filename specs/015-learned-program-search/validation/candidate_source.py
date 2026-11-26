"""Materialize and check all candidate bytes before importing candidate code."""
import fcntl
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def verify_tree(workspace, candidate, revision):
    entries = git(workspace, "ls-tree", "-rz", revision).split(b"\0")
    expected = {}
    for entry in filter(None, entries):
        meta, name = entry.split(b"\t", 1)
        mode, kind, oid = meta.split()
        if kind != b"blob" or mode not in (b"100644", b"100755"):
            raise ValueError("candidate requires regular tracked files")
        expected[os.fsdecode(name)] = (mode == b"100755", oid.decode())
    actual = set()
    for p in candidate.rglob("*"):
        info = p.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        name = str(p.relative_to(candidate))
        if name not in expected or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise ValueError("unexpected candidate file or alias: " + name)
        h = hashlib.sha1(f"blob {info.st_size}\0".encode())
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                h.update(chunk)
        if (bool(info.st_mode & 0o111), h.hexdigest()) != expected[name]:
            raise ValueError("cached candidate differs: " + name)
        actual.add(name)
    if actual != set(expected):
        raise ValueError("cached candidate is incomplete")


def materialize(root, revision):
    if not re.fullmatch("[0-9a-f]{40}", revision) or git(root, "rev-parse", revision + "^{commit}").decode().strip() != revision:
        raise ValueError("full candidate commit required")
    cache = root / "run-state/discovery-tools/round028-source"
    if root != root.resolve(strict=True) or cache.resolve() != cache:
        raise ValueError("physical workspace and source cache required")
    cache.mkdir(parents=True, exist_ok=True)
    with open(cache / "source.lock", "a", opener=lambda p, f: os.open(p, f | os.O_NOFOLLOW, 0o600)) as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        candidate = cache / revision
        if candidate.is_symlink():
            raise ValueError("candidate cache alias")
        if candidate.exists():
            verify_tree(root, candidate, revision)
        else:
            with tempfile.TemporaryDirectory(dir=cache) as temporary:
                staged = Path(temporary) / "source"
                staged.mkdir()
                with subprocess.Popen(["git", "-C", str(root), "archive", revision], stdout=subprocess.PIPE) as archive:
                    subprocess.run(["tar", "-xf", "-", "-C", str(staged)], stdin=archive.stdout, check=True)
                    if archive.wait():
                        raise ValueError("candidate source extraction failed")
                verify_tree(root, staged, revision)
                staged.rename(candidate)
        return candidate


if __name__ == "__main__":
    print(materialize(Path(sys.argv[1]), sys.argv[2]))
