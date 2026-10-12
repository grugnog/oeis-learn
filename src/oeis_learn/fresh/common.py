"""Small portable filesystem contracts shared by the VM and host controller."""

import fcntl
import hashlib
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return "sha256:" + hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def read(path):
    path = Path(path)
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError(f"regular JSON artifact required: {path}")
    with path.open("rb") as stream:
        payload = stream.read((128 << 20) + 1)
    if len(payload) > 128 << 20:
        raise ValueError(f"JSON artifact exceeds128MiB: {path}")
    return json.loads(payload, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def write(path, value, *, immutable=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value if isinstance(value, bytes) else canonical(value) + b"\n"
    if path.is_symlink():
        raise ValueError(f"symlink output: {path}")
    if immutable and path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"immutable input changed: {path}")
        return path
    fd, name = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(name).unlink(missing_ok=True)
    return path


def confined(root, name):
    root = Path(root).resolve()
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"relative artifact path required: {name}")
    target = root
    for part in path.parts:
        target = target / part
        if target.is_symlink():
            raise ValueError(f"symlink artifact: {target}")
    return target


def inventory(root, *, exclude=()):
    root = Path(root)
    files = {}
    for base, dirs, names in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in exclude)
        for name in dirs + sorted(names):
            path = Path(base) / name
            info = path.lstat()
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError(f"regular unlinked artifact required: {path}")
            files[str(path.relative_to(root))] = {"sha256": file_hash(path), "bytes": info.st_size}
    return files


def verify(root, files):
    for name, ref in files.items():
        path = confined(root, name)
        if (
            not path.is_file()
            or path.stat().st_size != ref["bytes"]
            or file_hash(path) != ref["sha256"]
        ):
            raise ValueError(f"artifact changed or missing: {path}")


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.close(fd)
    with path.open("rb") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
