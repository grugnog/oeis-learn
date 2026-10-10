"""Explicit native backend identity and strict compilation adapter."""

from pathlib import Path
import tomllib
from oeis_learn.sandbox.pipeline import digest_bytes


def source_bytes(root):
    names = (
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "src/lib.rs",
        "src/sandbox.rs",
        "src/engine.rs",
    )
    return b"".join(
        name.encode() + b"\0" + (root / name).read_bytes() + b"\0" for name in sorted(names)
    )


def native_identity():
    import oeis_wasm_evaluator as native

    root = Path(__file__).resolve().parents[3] / "crates/oeis_wasm_evaluator"
    embedded = bytes(native.source_identity_bytes())
    if not embedded or (root.exists() and embedded != source_bytes(root)):
        raise ValueError(
            "native extension does not match current Rust sources; rebuild Docker image"
        )
    lock = tomllib.loads(embedded.split(b"Cargo.lock\0", 1)[1].split(b"\0", 1)[0].decode())
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
