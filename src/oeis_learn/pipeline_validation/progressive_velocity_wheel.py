"""Build or install the exact candidate wheel inside a previously qualified ROCm environment."""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tarfile
import tempfile


def digest(value):
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def file_digest(path):
    with path.open("rb") as body:
        return "sha256:" + hashlib.file_digest(body, "sha256").hexdigest()


def installed(wheel, expected_wheel, expected_source, *, action):
    if file_digest(wheel) != expected_wheel:
        raise ValueError("candidate wheel bytes differ from the issued artifact")
    with tempfile.TemporaryDirectory() as tmp:
        # The CPU acceptance image uses an unseeded uv venv. The qualified
        # ROCm image provides pip. Both install just the already hashed wheel.
        if importlib.util.find_spec("pip") is None:
            installer = [
                "uv",
                "--no-config",
                "--offline",
                "--no-cache",
                "pip",
                "install",
                "--python",
                sys.executable,
            ]
        else:
            installer = [
                sys.executable,
                "-I",
                "-m",
                "pip",
                "--isolated",
                "--disable-pip-version-check",
                "install",
                "--no-cache-dir",
                "--no-compile",
            ]
        subprocess.run(
            [*installer, "--no-index", "--no-deps", "--target", tmp, str(wheel.resolve())],
            check=True,
        )
        sys.path.insert(0, tmp)
        from oeis_learn.experiments import source_snapshot as provenance

        if not Path(provenance.__file__).resolve().is_relative_to(Path(tmp)) or (
            digest(provenance.source_snapshot()) != expected_source
        ):
            raise ValueError("installed application differs from the registered candidate")
        if action in {"prepare-broad", "time-broad"}:
            from oeis_learn.pipeline_validation.progressive_velocity_broad import worker

            worker(action)
        elif action == "checkpoint-matrix":
            from oeis_learn.pipeline_validation.progressive_velocity_checkpoint import worker

            worker()
        elif action == "run":
            runpy.run_module(
                "oeis_learn.pipeline_validation.progressive_velocity_timing", run_name="__main__"
            )


def package(archive, expected_archive, expected_base, expected_source):
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from oeis_learn.sandbox.native_runtime import source_bytes
    import oeis_wasm_evaluator

    if digest(source_snapshot()) != expected_base or file_digest(archive) != expected_archive:
        raise ValueError("qualified base package or candidate source archive changed")
    out = Path("/work/output")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        with tarfile.open(archive, "r:gz") as bundle:
            bundle.extractall(root, filter="data")
        native = source_bytes(root / "crates/oeis_wasm_evaluator")
        if bytes(oeis_wasm_evaluator.source_identity_bytes()) != native:
            raise ValueError("the native runtime was built from different candidate inputs")
        env = dict(os.environ, SOURCE_DATE_EPOCH="315532800", PYTHONHASHSEED="0")
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "pip",
                "--isolated",
                "--disable-pip-version-check",
                "wheel",
                "--no-index",
                "--no-cache-dir",
                "--no-deps",
                "--no-build-isolation",
                "--wheel-dir",
                str(out),
                ".",
            ],
            cwd=root,
            env=env,
            check=True,
        )
    (wheel,) = out.glob("oeis_learn-*-py3-none-any.whl")
    # A fresh interpreter checks the installed wheel without inheriting modules from the base.
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            __file__,
            "check",
            str(wheel),
            file_digest(wheel),
            expected_source,
        ],
        check=True,
    )


def main(argv):
    action, path, *args = argv
    if action == "package":
        package(Path(path), *args)
    elif action in {"run", "check", "prepare-broad", "time-broad", "checkpoint-matrix"}:
        installed(Path(path), *args, action=action)
    else:
        raise ValueError("unknown candidate packaging action")


if __name__ == "__main__":
    main(sys.argv[1:])
