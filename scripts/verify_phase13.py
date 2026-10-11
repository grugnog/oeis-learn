"""Replay available CPU inputs, preserving separate source/model/hardware blockers.

This driver does not manufacture a source release, checkpoint or model stratum.
It samples the first bounded generic counters under the unchanged registered prior.
"""

import argparse
import gzip
from pathlib import Path
import runpy

from oeis_learn.data.generic_programs import GenericSampler
from oeis_learn.data.program_admission import validate_sample
from oeis_learn.evaluation.foundation_cohort import write_json
from oeis_learn.evaluation.legacy_wrappers import wrapper_inventory, load_legacy_configuration
from oeis_learn.evaluation.overflow_diagnostics import diagnose_overflow, archive_strata, STRATA
from oeis_learn.evaluation.source_policy import analyze_sequence_cohort
from oeis_learn.experiments.artifacts import compute_file_hash, compute_canonical_digest as digest
from oeis_learn.evaluation.experiment_runner import implementation_identity
import yaml

ROOT = Path(__file__).resolve().parents[1]


def verify_inputs(output, *, count=256, archive=None, development=None):
    if type(count) is not int or not 1 <= count <= 256:
        raise ValueError("bounded generic counter inventory required (1..256)")
    output = Path(output)
    snapshot = None if archive is None else archive.freeze()
    strata = {name: [] for name in STRATA} if archive is None else archive_strata(archive, snapshot)
    config = yaml.safe_load((ROOT / "configs/foundation/generic_sampler.yaml").read_text())
    sampler = GenericSampler(config)
    packets, rejected, distinct = {}, [], {}
    for counter in range(count):
        sample = sampler.sample(counter)
        reason = validate_sample(sample, sampler)
        if reason is not None:
            rejected.append(dict(counter=counter, reason=reason, sample_sha256=digest(sample)))
            continue
        packet = dict(config=config, sample=sample)
        key = digest(packet)
        packets[key] = packet
        distinct.setdefault(
            sample["canonical_source"],
            dict(source=sample["canonical_source"], origin="generic_sample", provenance_sha256=key),
        )
    strata["random_generic"] = list(distinct.values())
    bootstrap = dict(
        config_sha256=digest(config),
        issued_counters=list(range(count)),
        rejected=rejected,
        distinct_sources=len(distinct),
        provenance_sha256=digest(packets),
    )
    write_json(output, "bootstrap-inputs.json", bootstrap)
    overflow = diagnose_overflow(
        strata,
        output / "overflow",
        archive=archive,
        snapshot=snapshot,
        development=development,
        provenance_packets=packets,
    )
    configs = []
    for path in sorted((ROOT / "runs").glob("*/config.yaml")):
        parsed = load_legacy_configuration(path)
        weights = sorted(
            p
            for p in path.parent.rglob("*")
            if p.is_file() and p.suffix in (".pt", ".pth", ".ckpt")
        )
        configs.append(
            dict(
                path=path.relative_to(ROOT).as_posix(),
                sha256=compute_file_hash(path),
                effective_configuration_sha256=digest(parsed),
                parent_weights=[
                    dict(path=p.relative_to(ROOT).as_posix(), sha256=compute_file_hash(p))
                    for p in weights
                ],
                status="available_for_parity" if weights else "missing_trained_weights",
            )
        )
    # Stripped sample has values but no authoritative offsets/indices/keywords.
    parser = runpy.run_path(str(ROOT / "scripts/extract_oeis_data.py"))["stream_stripped"]
    stripped = ROOT / "data/sample_oeis/stripped"
    # The supplied sample is plain text; the historical importer consumes gzip.
    # Wrap the exact bytes deterministically without inventing source metadata.
    transport = output / "source-stripped.gz"
    transport.write_bytes(gzip.compress(stripped.read_bytes(), mtime=0))
    census = analyze_sequence_cohort([dict(terms=terms) for _, terms in parser(str(transport))])
    write_json(output, "available-source-census.json", census)
    blockers = dict(
        authoritative_source=[
            "No indexed-source/v1 manifest with hashed authoritative metadata in the supplied checkout.",
            "Stripped/names sample cannot qualify offsets, exclusions or a release cohort.",
        ],
        trained_artifacts=[]
        if any(c["parent_weights"] for c in configs)
        else [
            "No trained checkpoint bytes in the supplied runs. Wrapper parent parity and model samples remain unexecuted."
        ],
        release_archive=[]
        if archive is not None
        else ["No supplied strict archive bound to every authoritative release prefix."],
        development=[]
        if development is not None
        else ["No supplied authoritative development cohort for sealed real-source scoring."],
        hardware=["Workstation HIP, Docker roles and live GPU telemetry belong to phase 14."],
    )
    report = dict(
        version="phase13-cpu-inputs/v1",
        code_sha256=implementation_identity(),
        qualified=False,
        tasks={"T113": "pending_external_inputs", "T114": "partial_cpu_evidence"},
        wrappers=wrapper_inventory(ROOT),
        historical_configurations=configs,
        source=dict(
            path=stripped.relative_to(ROOT).as_posix(),
            sha256=compute_file_hash(stripped),
            census_sha256=digest(census),
        ),
        bootstrap=bootstrap,
        overflow_sha256=overflow["report_sha256"],
        blockers=blockers,
    )
    report["report_sha256"] = digest(report)
    write_json(output, "cpu-inputs.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--archive")
    parser.add_argument("--holdouts")
    parser.add_argument("--development")
    args = parser.parse_args()
    archive = None
    if args.archive:
        if not args.holdouts:
            parser.error("--archive requires its exact --holdouts")
        from oeis_learn.discovery.program_archive import ProgramArchive
        from oeis_learn.experiments.learning_profiles import load_holdouts
        from oeis_learn.experiments.artifacts import load_json

        archive = ProgramArchive(
            args.archive, holdouts=load_holdouts(load_json(Path(args.holdouts).read_bytes()))
        )
    report = verify_inputs(
        args.output, count=args.count, archive=archive, development=args.development
    )
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
