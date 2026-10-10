"""Container acceptance of actual G9 replay and public discovery/certificate CLI."""

import argparse
from copy import deepcopy
from pathlib import Path
from oeis_learn.cli.main import cli
from oeis_learn.discovery.certificates import make_certificate
from oeis_learn.experiments.artifacts import canonical_bytes, load_json
from oeis_learn.experiments.learning_profiles import holdout_profile
from oeis_learn.sandbox.logical_ir import profile_identity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", required=True)
    args = parser.parse_args()
    root = Path(args.evidence)
    phase = root / "discovery"
    # Reuse the independently admitted generated-program archive from the G8
    # acceptance fixture. Passing its actual holdouts does not waive any check.
    contract = load_json((root / "learning" / "learning-contract.json").read_bytes())
    from oeis_learn.experiments.learning_profiles import load_holdouts

    admission = root / "profiles" / (contract["profiles"]["admission"][7:] + ".json")
    holdouts = load_holdouts(load_json(admission.read_bytes()))
    selected = phase / "holdouts.json"
    selected.write_bytes(canonical_bytes(holdout_profile(holdouts)))
    common = [
        "foundation",
        "discover",
        "--archive",
        str(root / "learning" / "archive"),
        "--holdouts",
        str(selected),
        "--discovery-gate-report",
        str(phase / "g9.json"),
        "--json",
    ]
    assert cli([*common, "--output", str(phase / "cli-discovery")]) == 0
    report = load_json((phase / "cli-discovery" / "discovery.json").read_bytes())
    assert report["information_policy"] == "verified_training_programs_only"
    assert report["external_lookup_performed"] is False
    # A re-sealed digest cannot make obsolete source/build evidence current.
    changed = deepcopy(load_json((phase / "g9.json").read_bytes()))
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    changed["implementation_sha256"] = digest("other-source")
    changed["report_sha256"] = digest(changed, "report_sha256")
    stale = phase / "stale-g9.json"
    stale.write_bytes(canonical_bytes(changed))
    rejected = common[:]
    rejected[rejected.index(str(phase / "g9.json"))] = str(stale)
    assert cli([*rejected, "--output", str(phase / "stale-discovery")]) == 2
    assert not (phase / "stale-discovery" / "discovery.json").exists()
    certificate = make_certificate(
        "polynomial_program",
        ["(pure (return (add (index) (const 0))))", "(pure (return (index)))"],
        [1, -1],
    )
    cert_path, profile_path = phase / "certificate.json", phase / "logical-profile.json"
    cert_path.write_bytes(canonical_bytes(certificate))
    profile_path.write_bytes(canonical_bytes(profile_identity("pure")))
    assert (
        cli(
            [
                "foundation",
                "check-certificate",
                "--certificate",
                str(cert_path),
                "--profile",
                str(profile_path),
                "--output",
                str(phase / "cli-certificate"),
                "--json",
            ]
        )
        == 0
    )
    acceptance = {
        "status": "PASS",
        "discovery": "actual_verified_archive",
        "certificate": "fresh_isolated_replay",
        "stale_evidence": "rejected",
        "gpu_required": False,
    }
    (phase / "cli-acceptance.json").write_bytes(canonical_bytes(acceptance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
