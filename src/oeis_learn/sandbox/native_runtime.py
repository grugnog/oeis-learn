"""Explicit native backend identity and strict compilation adapter."""

from pathlib import Path
import tomllib
from oeis_learn.sandbox.pipeline import digest_bytes


def source_bytes(root):
    repository = Path(root).resolve().parents[1]
    names = set()
    for crate in ("oeis_wasm_evaluator", "oeis_synthesis_core", "oeis_bigint_runtime"):
        for path in (repository / "crates" / crate).rglob("*"):
            if path.is_file() and path.suffix in (".rs", ".toml", ".lock") and not any(
                part in ("target", ".git") for part in path.relative_to(repository).parts
            ):
                names.add(path.relative_to(repository).as_posix())
    names.update((
        "Dockerfile", "docker/dev/build-native.sh", "docker/dev/build-synthesis.sh",
        "docker/dev/install.sh", "src/oeis_learn/synthesis/records.schema.json",
        "src/oeis_learn/sandbox/preamble.wat",
    ))
    return b"".join(
        name.encode() + b"\0" + (repository / name).read_bytes() + b"\0"
        for name in sorted(names)
    )



def native_identity():
    import oeis_wasm_evaluator as native

    root = Path(__file__).resolve().parents[3] / "crates/oeis_wasm_evaluator"
    embedded = bytes(native.source_identity_bytes())
    if not embedded or (root.exists() and embedded != source_bytes(root)):
        raise ValueError(
            "native extension does not match current Rust sources; rebuild Docker image"
        )
    lock = tomllib.loads(embedded.split(b"crates/oeis_wasm_evaluator/Cargo.lock\0", 1)[1].split(b"\0", 1)[0].decode())
    wasmtime_version = next(p["version"] for p in lock["package"] if p["name"] == "wasmtime")
    return {
        "backend": "rust",
        "wasmtime": wasmtime_version,
        "source_sha256": digest_bytes(embedded),
        "binary_sha256": digest_bytes(Path(native.__file__).read_bytes()),
        "consume_fuel": True,
        "execution_threads": 1,
    }


def compile_native(source):
    import oeis_wasm_evaluator as native

    return native.NativeModule(source)
