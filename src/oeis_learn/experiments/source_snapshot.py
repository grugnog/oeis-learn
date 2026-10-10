"""Build-time qualification identity usable from an installed wheel.

The build snapshot includes owning tests and locks, while every installed Python,
WAT and schema byte is rechecked against it. Evidence replay still checks runtime
and native binary identities separately. Building a wheel cannot create PASS evidence.
"""

import hashlib
import json
from pathlib import Path


def file_digest(data):
    return "sha256:" + hashlib.sha256(data).hexdigest()


def collect_snapshot(root):
    root = Path(root)
    files = set()
    for directory in (
        "src/oeis_learn",
        "tests",
        "crates/oeis_wasm_evaluator",
        "configs/foundation",
        "docker/dev",
        "scripts",
        "specs/007-experiment-foundation/contracts",
        "specs/005-trustworthy-synthesis-readiness/contracts",
    ):
        files.update(
            p
            for p in (root / directory).rglob("*")
            if p.is_file()
            and p.suffix
            in (".py", ".rs", ".wat", ".json", ".toml", ".lock", ".yaml", ".yml", ".sh")
            and "__pycache__" not in p.parts
            and "target" not in p.parts
            and p.name != "qualification-sources.json"
        )
    files.update(root / name for name in ("Dockerfile", "pyproject.toml", "uv.lock", "setup.py"))
    return {
        str(p.relative_to(root)): file_digest(p.read_bytes()) for p in sorted(files) if p.exists()
    }


def source_snapshot():
    root = Path(__file__).resolve().parents[3]
    if (root / "src/oeis_learn").is_dir() and (root / "pyproject.toml").is_file():
        return collect_snapshot(root)
    package = Path(__file__).resolve().parents[1]
    snapshot = json.loads((package / "experiments/qualification-sources.json").read_text())
    for name, expected in snapshot.items():
        if name.startswith("src/oeis_learn/"):
            path = package / name.removeprefix("src/oeis_learn/")
            if not path.is_file() or file_digest(path.read_bytes()) != expected:
                raise ValueError("installed wheel source identity mismatch: " + name)
    # Reject newly introduced code/resources too, rather than hashing only expected files.
    actual = {
        str(p.relative_to(package))
        for p in package.rglob("*")
        if p.is_file()
        and p.suffix in (".py", ".wat", ".json")
        and p.name != "qualification-sources.json"
    }
    expected = {
        name.removeprefix("src/oeis_learn/")
        for name in snapshot
        if name.startswith("src/oeis_learn/")
    }
    if actual != expected:
        raise ValueError("installed wheel inventory mismatch")
    return snapshot


def qualification_identity(gate):
    payload = {"gate": gate, "sources": source_snapshot()}
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)).encode()
    # Match artifacts.compute_canonical_digest's exact canonical serializer.
    return file_digest(raw)
