"""Container-only phase-8 CLI acceptance using actual G7/G8 artifact replay."""

import argparse
from pathlib import Path

from oeis_learn.cli.main import cli
from oeis_learn.data.generic_programs import default_config
from oeis_learn.discovery.program_archive import ProgramArchive
from oeis_learn.experiments.artifacts import canonical_bytes, load_json
from oeis_learn.experiments.learning_profiles import extend_learning_contract
from oeis_learn.experiments.config import load_qualified_config
from oeis_learn.sandbox.runner import FoundationRunner
from tests.unit.test_foundation_archive_curriculum import generic_fixture


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    root = Path(args.evidence)
    phase = root / "learning"
    phase.mkdir(exist_ok=True)
    sample, outputs, holdouts = generic_fixture()
    archive = ProgramArchive(phase / "archive", holdouts=holdouts)
    with FoundationRunner(phase / "admission-runtime", backend="rust") as runner:
        archive.admit(
            sample["canonical_source"],
            outputs[:20],
            outputs,
            origin="generic_sample",
            provenance={"config": default_config(), "sample": sample},
            runner=runner,
            request_id="phase8-cli-admission",
        )
    profile_root = root / "profiles"
    base = load_json((root / "qualified-contract.json").read_bytes())
    g8 = load_json((phase / "g8.json").read_bytes())
    contract = extend_learning_contract(base, g8, profile_root, holdouts=holdouts)
    config = phase / "learning-contract.json"
    config.write_bytes(canonical_bytes(contract))
    common = [
        "--config",
        str(config),
        "--gate-report",
        str(root / "g7.json"),
        "--learning-gate-report",
        str(phase / "g8.json"),
        "--profiles",
        str(profile_root),
        "--archive",
        str(phase / "archive"),
        "--device",
        "cpu",
        "--diagnostic",
        "--json",
    ]
    assert (
        cli(
            [
                "foundation",
                "learning-train",
                *common,
                "--output",
                str(phase / "training"),
                "--updates",
                "1",
            ]
        )
        == 0
    )
    checkpoint = phase / "training" / "qualified-00000000.json"
    assert checkpoint.exists()
    assert (
        cli(
            [
                "foundation",
                "learning-train",
                *common,
                "--output",
                str(phase / "training"),
                "--updates",
                "1",
                "--resume",
                str(checkpoint),
            ]
        )
        == 0
    )
    checkpoint = phase / "training" / "qualified-00000001.json"
    prompt = phase / "visible.json"
    prompt.write_bytes(canonical_bytes({"values": outputs[:20]}))
    assert (
        cli(
            [
                "foundation",
                "learning-search",
                *common,
                "--checkpoint",
                str(checkpoint),
                "--prompt",
                str(prompt),
                "--portfolio",
                "greedy",
                "--output",
                str(phase / "search"),
            ]
        )
        == 0
    )
    report = load_json((phase / "search" / "search.json").read_bytes())
    assert report["qualified"] is False and report["spent"]["attempts"] == 1

    def resolve(key):
        return load_json((profile_root / (key[7:] + ".json")).read_bytes())

    try:
        load_qualified_config(
            config,
            g7_report=load_json((root / "g7.json").read_bytes()),
            evidence_root=root,
            resolve=resolve,
            g8_report=g8,
            g8_evidence_root=phase,
            diagnostic=False,
        )
    except ValueError as exc:
        assert "hardware" in str(exc)
    else:
        raise AssertionError("CPU-only evidence cannot authorize GPU learning")
    (phase / "cli-acceptance.json").write_bytes(
        canonical_bytes(
            {
                "status": "PASS",
                "qualification": "cpu_diagnostic",
                "updates": 2,
                "search_attempts": 1,
                "hardware_activation": "rejected",
            }
        )
    )


if __name__ == "__main__":
    main()
