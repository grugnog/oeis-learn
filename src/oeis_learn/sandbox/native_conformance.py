"""Full fixed native/Python/reference parity corpus, with replayable transcripts."""

from pathlib import Path
from oeis_learn.experiments.artifacts import canonical_bytes, compute_canonical_digest, load_json
from oeis_learn.sandbox.pipeline import runtime_identity
from oeis_learn.sandbox.worker_pool import WorkerPool
from oeis_learn.sandbox.conformance_runner import arithmetic_vectors, structured_vectors
from importlib.resources import files


def cases():
    known = load_json(
        files("oeis_learn.sandbox").joinpath("conformance/known_regressions.json").read_bytes()
    )
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


def run_native_conformance(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    with (
        WorkerPool(output / "python", backend="python_wasmtime") as python,
        WorkerPool(output / "rust", backend="rust") as native,
    ):
        for name, source, indices, expected, outcome in cases():
            a = python.evaluate(source, indices, request_id=name)
            b = native.evaluate(source, indices, request_id=name)
            for value in (a, b):
                if (
                    value.outcome != outcome
                    or value.outputs != expected
                    or value.reference_outputs != expected
                ):
                    raise ValueError(
                        f"{name}: conformance disagreement: {value.outcome}:{value.reason}"
                    )
                if value.fuel != sum(value.fuel_terms) or any(
                    v > 1000000 for v in value.fuel_terms
                ):
                    raise ValueError(f"{name}: invalid fuel accounting")
            row = {
                "case": name,
                "input_sha256": compute_canonical_digest(
                    [name, source, indices, expected, outcome]
                ),
                "python": a.to_dict(),
                "native": b.to_dict(),
            }
            rows.append(row)
    transcript = {
        "version": "native-parity/v1",
        "python_runtime": runtime_identity(),
        "native_runtime": runtime_identity("rust"),
        "records": rows,
    }
    transcript["identity_sha256"] = compute_canonical_digest(transcript, "identity_sha256")
    (output / "parity.json").write_bytes(canonical_bytes(transcript))
    return transcript


def check_native_transcript(transcript):
    """Reconstruct the entire declared corpus and check exact observed evidence."""
    if (
        set(transcript)
        != {"version", "python_runtime", "native_runtime", "records", "identity_sha256"}
        or transcript["version"] != "native-parity/v1"
    ):
        raise ValueError("invalid native transcript")
    if transcript["identity_sha256"] != compute_canonical_digest(transcript, "identity_sha256"):
        raise ValueError("native transcript identity mismatch")
    if transcript["python_runtime"] != runtime_identity() or transcript[
        "native_runtime"
    ] != runtime_identity("rust"):
        raise ValueError("native conformance runtime mismatch")
    expected_cases = list(cases())
    if len(transcript["records"]) != len(expected_cases):
        raise ValueError("incomplete native corpus")
    for row, (name, source, indices, expected, outcome) in zip(
        transcript["records"], expected_cases
    ):
        if row["case"] != name or row["input_sha256"] != compute_canonical_digest(
            [name, source, indices, expected, outcome]
        ):
            raise ValueError("native corpus identity mismatch")
        for backend, key in (("python_wasmtime", "python"), ("rust", "native")):
            value = row[key]
            if (
                value["indices"] != indices
                or value["outputs"] != expected
                or value["reference_outputs"] != expected
                or value["outcome"] != outcome
            ):
                raise ValueError("native corpus output/classification mismatch")
            if (
                value["fuel"] != sum(value["fuel_terms"])
                or value["fuel"] > 50000000
                or any(type(v) is not int or not 0 <= v <= 1000000 for v in value["fuel_terms"])
            ):
                raise ValueError("invalid conformance fuel evidence")
            if value["identities"]["runtime_sha256"] != compute_canonical_digest(
                transcript[key + "_runtime"] if key == "native" else transcript["python_runtime"]
            ):
                raise ValueError("conformance runtime binding mismatch")
    return True
