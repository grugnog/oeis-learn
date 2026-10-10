"""Diagnostic CLI acceptance shared by checkout and installed-wheel verification.

Actual current G7/G8/G9 artifacts and a native runtime are mandatory. CPU learning
is explicitly diagnostic; these fixtures cannot authorize a hardware learning run.
"""

from pathlib import Path

from oeis_learn.cli.main import cli
from oeis_learn.data.generic_programs import default_config
from oeis_learn.discovery.program_archive import ProgramArchive
from oeis_learn.experiments.artifacts import canonical_bytes, load_json
from oeis_learn.experiments.learning_profiles import extend_learning_contract
from oeis_learn.experiments.config import load_qualified_config
from oeis_learn.sandbox.runner import FoundationRunner



from copy import deepcopy
from oeis_learn.discovery.certificates import make_certificate
from oeis_learn.experiments.learning_profiles import holdout_profile
from oeis_learn.sandbox.logical_ir import profile_identity

def generic_fixture():
    """Small deterministic generated program; no named sequence or teacher injection."""
    from oeis_learn.data.generic_programs import GenericSampler
    from oeis_learn.data.structural_splits import StructuralHoldouts, structural_signature, split_for_structure
    from oeis_learn.sandbox.reference import execute_term
    from oeis_learn.sandbox.wat_ast import parse
    sampler = GenericSampler(default_config())
    for counter in range(1000):
        sample = sampler.sample(counter)
        if sample["sampling_failure"] is not None or len(sample["body_tokens"]) > 70:
            continue
        program = parse(sample["canonical_source"])
        results = [execute_term(program, i, step_limit=10000) for i in range(24)]
        if all(result.outcome is None for result in results):
            source = sample["canonical_source"]
            signature = structural_signature(source)
            seed = next(i for i in range(1000) if split_for_structure(signature, i) == "train")
            return (
                sample,
                [r.value for r in results],
                StructuralHoldouts(seed=seed, reserved_compositions=()),
            )
    raise AssertionError("no bounded valid generic fixture")


def verify_learning_cli(evidence):
    root = Path(evidence)
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


def verify_discovery_cli(evidence):
    root = Path(evidence)
    phase = root / "discovery"
    qualification = load_json((root / "discovery-qualification-cli.json").read_bytes())
    assert qualification == load_json((phase / "g9.json").read_bytes())
    assert qualification["status"] == "PASS"
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

