"""Inspect dependency smoke artifacts; never qualifies production arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib

SMOKE_INTEGER = "340282366920938463463374607431768211457"


def sha256(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def target_features(raw):
    """Read the compiler's declared target_features custom section for audit."""
    cursor = 8

    def unsigned():
        nonlocal cursor
        value = 0
        for shift in range(0, 35, 7):
            byte = raw[cursor]
            cursor += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
        raise ValueError("overlong WASM section length")

    def string():
        nonlocal cursor
        size = unsigned()
        value = raw[cursor:cursor + size].decode("utf-8")
        cursor += size
        return value

    if raw[:8] != b"\0asm\x01\0\0\0":
        raise ValueError("invalid WASM header")
    while cursor < len(raw):
        section = raw[cursor]
        cursor += 1
        size = unsigned()
        end = cursor + size
        if end > len(raw):
            raise ValueError("truncated WASM section")
        if section == 0 and string() == "target_features":
            result = []
            for _ in range(unsigned()):
                prefix = chr(raw[cursor])
                cursor += 1
                result.append(prefix + string())
            if cursor != end:
                raise ValueError("invalid target_features section")
            return result
        cursor = end
    return []


def inspect(root=Path("/opt/oeis-build"), info=Path("/opt/oeis-build-info")):
    import wasmtime
    import oeis_wasm_evaluator as native
    from oeis_learn.sandbox.native_runtime import native_identity

    artifact = info / "bigint-dependency-smoke.wasm"
    raw = artifact.read_bytes()
    if not raw or len(raw) > 16 * 1024 * 1024:
        raise ValueError("invalid smoke artifact size")
    config = wasmtime.Config()
    config.consume_fuel = True
    engine = wasmtime.Engine(config)
    module = wasmtime.Module(engine, raw)
    if list(module.imports):
        raise ValueError("standalone guest must have zero imports (including WASI/Python/Z3)")
    store = wasmtime.Store(engine)
    store.set_limits(memory_size=64 * 1024 * 1024, instances=1, memories=1)
    store.set_fuel(10_000_000)
    instance = wasmtime.Instance(store, module, [])
    if instance.exports(store)["production_enabled"](store) != 0:
        raise ValueError("unfinished runtime is enabled")
    if instance.exports(store)["dependency_smoke"](store) != 1:
        raise ValueError("standalone dashu dependency smoke failed")
    if native.synthesis_dependency_smoke() != SMOKE_INTEGER:
        raise ValueError("native binding narrowed the exact Z3 numeral")
    from oeis_learn.synthesis.engine_policy import production_policy
    from oeis_learn.synthesis.contracts import identity

    engine_policy = dict(manifest=production_policy())
    engine_policy["sha256"] = identity(engine_policy["manifest"])
    if identity(json.loads(native.synthesis_engine_policy())) != engine_policy["sha256"]:
        raise ValueError("compiled engine policy differs from packaged policy; rebuild image")
    locks = {}
    for crate in ("oeis_bigint_runtime", "oeis_synthesis_core", "oeis_wasm_evaluator"):
        path = root / "crates" / crate / "Cargo.lock"
        packages = tomllib.loads(path.read_text())["package"]
        locks[crate] = {
            "sha256": sha256(path.read_bytes()),
            "tuple": [{key: p[key] for key in ("name", "version", "source", "checksum") if key in p}
                      for p in packages if p["name"] in ("dashu-int", "z3", "z3-sys", "z3-src", "wasmtime", "pyo3")],
        }
        if crate == "oeis_bigint_runtime" and any(
            p["name"] in ("z3", "z3-sys", "z3-src", "pyo3", "wasmtime") for p in packages
        ):
            raise ValueError("guest lock contains host dependencies")
    return {
        "version": "bigint-build-smoke/v1", "production_enabled": False,
        "qualification": "dependency smoke only; Phase 2 arithmetic gate pending",
        "runtime_sha256": sha256(raw), "runtime_bytes": len(raw),
        "imports": [], "target_features": target_features(raw),
        "exports": [{"name": e.name, "kind": type(e.type).__name__} for e in module.exports],
        "native": native_identity(), "locks": locks, "engine_policy": engine_policy,
        "rustc": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "guest_flags": {"target": "wasm32-unknown-unknown", "profile": "release",
                        "panic": "abort", "lto": True, "codegen_units": 1, "opt_level": "s"},
        "native_flags": {"profile": "release", "z3": "vendored", "locked": True,
                         "wheel_compatibility": "linux", "cxx": "clang++-19", "cxxflags": "-stdlib=libc++", "cxxstdlib": "c++"},
        "cxx_runtime": {p.name: sha256(p.read_bytes()) for p in Path("/opt/oeis-cxx/lib").glob("*.so.1")},
        "binding_exact_integer": SMOKE_INTEGER,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(inspect(), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
