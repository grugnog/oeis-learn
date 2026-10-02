"""Reproducible CPU execution gate; no training, solver or sequence templates."""

from __future__ import annotations

import hashlib
from importlib.resources import files
from pathlib import Path
import random
import sys
import time

from oeis_learn.experiments.artifacts import (
    ArtifactPath,
    atomic_write,
    canonical_bytes,
    compute_canonical_digest,
    load_json,
)
from oeis_learn.experiments.profiles import profile_digests, validate_profile_file
from oeis_learn.sandbox.pipeline import runtime_identity
from oeis_learn.sandbox.worker_pool import WorkerPool


def arithmetic_vectors():
    rng = random.Random(7001)
    edges = [0, 1, -1, 2**63 - 1, -(2**63), 2**64, 2**128 - 1, -(2**192), -(2**255), 2**255 - 1]

    def wide():
        if rng.randrange(2):
            return rng.choice(edges)
        bits = rng.randrange(1, 256)
        return rng.getrandbits(bits) * rng.choice((-1, 1))

    for i in range(10000):
        a, b = wide(), wide()
        op = i % 5
        if op == 0:
            source, expected = f"i256.const {a}", a
        elif op in (1, 2):
            source = f"i256.const {a} i256.const {b} i256." + ("add" if op == 1 else "sub")
            expected = a + b if op == 1 else a - b
        elif op == 3:
            b = rng.choice([-(2**63), -1, 0, 1, 2**63 - 1, rng.randrange(-(2**63), 2**63)])
            source, expected = f"i256.const {a} i64.const {b} i256.mul_scalar", a * b
        else:
            # Cancellation across signed/carry boundaries, followed by scalar.
            source, expected = f"i256.zero i256.const {a} i256.add i64.const -1 i256.mul_scalar", -a
        outcome = None if -(2**255) <= expected < 2**255 else "numeric_limit"
        yield f"arithmetic-{i:05}", source, [0], [str(expected)] if outcome is None else [], outcome


def structured_vectors():
    rng = random.Random(7002)
    mask = 2**64 - 1
    for i in range(256):
        a, b, c = rng.randrange(-10000, 10000), rng.randrange(-31, 32), rng.randrange(1, 11)
        op = rng.choice(["add", "sub", "mul", "xor", "or", "and"])
        modulus, offset = rng.randrange(1, 9), rng.randrange(1, 6)
        source = f"""i64.const {a} local.set $v0
          local.get $n i32.const {modulus} i32.rem_u i32.const {offset} i32.add local.set $c1
          ( block ( loop
            local.get $c0 local.get $c1 i32.ge_s br_if 1
            local.get $v0 i64.const {b} i64.{op} local.set $v0
            local.get $c0 i32.const 1 i32.add local.set $c0 br 0 ) )
          local.get $n i32.const 1 i32.and
          ( if ( result i64 )
            ( then local.get $v0 i64.const {c} i64.add )
            ( else local.get $v0 i64.const {c} i64.sub ) )
          i64.const 0 i64.const 0 i64.const 0"""
        expected = []
        for n in range(100):
            value = a & mask
            for _ in range(n % modulus + offset):
                if op == "add":
                    value += b
                elif op == "sub":
                    value -= b
                elif op == "mul":
                    value *= b
                elif op == "xor":
                    value ^= b
                elif op == "or":
                    value |= b
                else:
                    value &= b
                value &= mask
            value += c if n & 1 else -c
            expected.append(str(value & mask))
        yield f"structured-{i:03}", source, list(range(100)), expected, None


def run_conformance(profile: Path, output: Path) -> dict:
    validate_profile_file(profile)
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("conformance output must be empty; evidence is immutable")
    output.mkdir(parents=True, exist_ok=True)
    data = files("oeis_learn.sandbox").joinpath("conformance")
    known = load_json(data.joinpath("known_regressions.json").read_bytes())
    manifest = load_json(data.joinpath("manifest.json").read_bytes())
    corpus = hashlib.sha256()
    disagreements, interrupted, count, terms = [], [], 0, 0
    started = time.monotonic_ns()

    def cases():
        for row in known:
            yield (
                row["id"],
                row["source"],
                [0],
                [row["expect"]] if row["expect"] is not None else [],
                row["outcome"],
            )
        yield from arithmetic_vectors()
        yield from structured_vectors()

    with WorkerPool(output / "evidence", workers=1) as pool:
        for identity, source, indices, expected, outcome in cases():
            corpus.update(canonical_bytes([identity, source, indices, expected, outcome]) + b"\n")
            result = pool.evaluate(source, indices, request_id=identity)
            count += 1
            terms += len(indices)
            if result.outcome == "execution_limit" or (result.reason or "").startswith(
                "worker_failure:"
            ):
                interrupted.append(
                    {"case": identity, "outcome": result.outcome, "reason": result.reason}
                )
            elif (
                result.outcome != outcome
                or result.outputs != expected
                or result.reference_outputs != expected
            ):
                disagreements.append(
                    {
                        "case": identity,
                        "expected_outcome": outcome,
                        "expected": expected,
                        "actual_outcome": result.outcome,
                        "reason": result.reason,
                        "evidence_file": hashlib.sha256(identity.encode()).hexdigest() + ".json",
                    }
                )
            if count % 500 == 0:
                print(
                    f"conformance: {count} cases, {len(disagreements)} disagreements, {len(interrupted)} interruptions",
                    file=sys.stderr,
                )
    expected_count = len(known) + 10000 + 256
    status = "PASS" if count == expected_count and not disagreements and not interrupted else "FAIL"
    report = {
        "command": "conformance",
        "status": status,
        "gate": "G1",
        "purpose": "conformance",
        "profiles": profile_digests(),
        "runtime": runtime_identity(),
        "manifest_sha256": compute_canonical_digest(manifest),
        "known_regressions_sha256": compute_canonical_digest(known),
        "corpus_sha256": "sha256:" + corpus.hexdigest(),
        "counts": {
            "known": len(known),
            "arithmetic": 10000,
            "structured": 256,
            "executed": count,
            "requested_terms": terms,
        },
        "seeds": manifest["seeds"],
        "unexplained_disagreements": disagreements,
        "interruptions": interrupted,
        "adapters": {
            "python_wasmtime": "qualified" if status == "PASS" else "failed",
            "rust": "unavailable: US5 parity gate not run",
        },
        "elapsed_ns": time.monotonic_ns() - started,
        "gpu_readiness": "not_evaluated",
        "container_isolation": "not_evaluated: T046",
        "proof_status": "not_claimed",
        "result_path": str(output / "report.json"),
        "evidence_path": str(output / "evidence"),
    }
    atomic_write(ArtifactPath(output, "report.json"), canonical_bytes(report))
    return report
