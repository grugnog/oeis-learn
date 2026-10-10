"""Execute the documented CPU quickstart; real limits and diagnostic labels apply."""

import argparse
from pathlib import Path
import subprocess
import sys

from oeis_learn.experiments.artifacts import (
    canonical_bytes,
    compute_file_hash,
    load_json,
    compute_canonical_digest,
)
from oeis_learn.evaluation.experiment_runner import fixture_manifest

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    records = []

    def run(name, arguments):
        command = [
            sys.executable,
            "-m",
            "oeis_learn.cli.main",
            "foundation",
            *map(str, arguments),
            "--json",
        ]
        print("Executing " + name, flush=True)
        try:
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=600)
            stdout, stderr, code = result.stdout, result.stderr, result.returncode
        except subprocess.TimeoutExpired as exc:
            stdout, stderr, code = "", str(exc), 124
        (output / (name + ".stdout")).write_text(stdout)
        (output / (name + ".stderr")).write_text(stderr)
        report = load_json(stdout) if code == 0 else None
        records.append(dict(name=name, command=command, exit_code=code, report=report))
        return report

    fixture = ROOT / "tests/fixtures/foundation"
    cohort = run(
        "cohort",
        [
            "freeze-cohort",
            "--source",
            fixture / "source",
            "--config",
            fixture / "cohort.yaml",
            "--output",
            output / "cohort",
        ],
    )
    pool = (
        run(
            "pool",
            [
                "build-pool",
                "--config",
                ROOT / "configs/foundation/generic_sampler.yaml",
                "--cohort",
                output / "cohort",
                "--output",
                output / "pool",
            ],
        )
        if cohort
        else None
    )
    training = (
        run(
            "train",
            [
                "train",
                "--config",
                ROOT / "configs/foundation/wat_smoke.yaml",
                "--pool",
                output / "pool",
                "--run-dir",
                output / "run",
                "--device",
                "cpu",
                "--diagnostic",
                "--diagnostic-small-host",
                "--stop-after-update",
                "1",
            ],
        )
        if pool
        else None
    )
    resumed = run("resume", ["resume", "--run-dir", output / "run"]) if training else None
    if resumed:
        run("inspect", ["inspect", "--run-dir", output / "run"])
        run(
            "development",
            [
                "evaluate",
                "--checkpoint",
                resumed["checkpoint"],
                "--cohort",
                output / "cohort",
                "--split",
                "development",
                "--protocol",
                fixture / "evaluation.json",
                "--output",
                output / "development",
            ],
        )
        from oeis_learn.evaluation.checkpoint import load_foundation_checkpoint
        from oeis_learn.evaluation.foundation_cohort import load_cohort

        checkpoint = load_foundation_checkpoint(resumed["checkpoint"])
        stop = dict(
            decision="stop",
            reason="Frozen three-update diagnostic completed",
            checkpoint_sha256=checkpoint.manifest["blob_sha256"],
            protocol_sha256=compute_canonical_digest(
                load_json((fixture / "evaluation.json").read_bytes())
            ),
            cohort_id=load_cohort(output / "cohort")["cohort_id"],
        )
        (output / "stopping-record.json").write_bytes(canonical_bytes(stop))
        finalized = run(
            "finalize",
            [
                "finalize",
                "--run-dir",
                output / "run",
                "--checkpoint",
                resumed["checkpoint"],
                "--cohort",
                output / "cohort",
                "--protocol",
                fixture / "evaluation.json",
                "--stopping-record",
                output / "stopping-record.json",
                "--output",
                output / "finalization.json",
            ],
        )
        if finalized:
            run(
                "final",
                [
                    "evaluate",
                    "--finalization",
                    output / "finalization.json",
                    "--split",
                    "final",
                    "--output",
                    output / "final",
                ],
            )
    sources = load_json((fixture / "source/records.json").read_bytes())
    (output / "census-input.json").write_bytes(
        canonical_bytes(
            [dict(terms=row["values"], offset=row["first_index"], tags=[]) for row in sources]
        )
    )
    run(
        "census",
        ["census", "--source", output / "census-input.json", "--output", output / "census"],
    )
    (output / "measurement-input.json").write_bytes(canonical_bytes(fixture_manifest()))
    run(
        "measure-dry",
        [
            "measure",
            "--manifest",
            output / "measurement-input.json",
            "--output",
            output / "measure-dry",
            "--dry-run",
        ],
    )
    run(
        "measure",
        [
            "measure",
            "--manifest",
            output / "measurement-input.json",
            "--output",
            output / "measure",
        ],
    )
    # Failed prerequisites remain visible rather than silently dropping slices.
    for name in (
        "cohort",
        "pool",
        "train",
        "resume",
        "inspect",
        "development",
        "finalize",
        "final",
        "census",
        "measure-dry",
        "measure",
    ):
        if not any(record["name"] == name for record in records):
            records.append(
                dict(
                    name=name,
                    command=None,
                    exit_code=None,
                    report=None,
                    reason="prerequisite failed",
                )
            )
    passed = all(row["exit_code"] == 0 for row in records)
    (output / "quickstart.json").write_bytes(
        canonical_bytes(
            dict(
                status="PASS" if passed else "FAIL",
                qualification="cpu_diagnostic",
                full_007_complete=False,
                slices=records,
            )
        )
    )
    files = [
        dict(path=p.relative_to(output).as_posix(), sha256=compute_file_hash(p))
        for p in sorted(output.rglob("*"))
        if p.is_file()
    ]
    (output / "handoff-inputs.json").write_bytes(
        canonical_bytes(
            dict(
                version="handoff-inputs/v1", files=files, readiness={"measurement_root": "measure"}
            )
        )
    )
    run(
        "handoff",
        ["handoff", "--manifest", output / "handoff-inputs.json", "--output", output / "handoff"],
    )
    print("CPU quickstart " + ("PASS" if passed else "FAIL; see saved slice results"))
    return 0 if passed and records[-1]["exit_code"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
