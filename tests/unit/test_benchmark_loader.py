"""Unit tests for benchmark manifest loader, alias handling, and leakage detection."""

from __future__ import annotations

import pytest
from oeis_learn.data.benchmark import (
    BenchmarkCohort,
    BenchmarkTarget,
    check_leakage_fingerprints,
    compute_term_fingerprint,
    load_benchmark_manifest,
)


def sample_target(oeis_id="A000045", terms=None, program_fps=None) -> BenchmarkTarget:
    if terms is None:
        terms = [str(i) for i in range(120)]
    observed = terms[:20]
    unseen = terms[20:120]
    fp = compute_term_fingerprint(terms)
    return BenchmarkTarget(
        oeis_id=oeis_id,
        name=f"Sequence {oeis_id}",
        offset=0,
        family="POLYNOMIAL",
        curriculum_stage=1,
        observed_terms=observed,
        unseen_terms=unseen,
        result_profile="i64_scalar_v1",
        terms_sha256="sha256:" + "0" * 64,
        formula_definition_id=None,
        term_fingerprint=fp,
        program_fingerprints=program_fps or [],
        tags=["core"],
    )


def test_term_fingerprint_normalization():
    terms1 = [1, 2, 3, 4]
    terms2 = ["1", "2", "3", "4"]
    assert compute_term_fingerprint(terms1) == compute_term_fingerprint(terms2)
    assert compute_term_fingerprint(terms1).startswith("sha256:")


def test_leakage_detection_matching_terms():
    t = sample_target("A000045", terms=[str(i) for i in range(120)])
    cohort = BenchmarkCohort(
        schema_version="1.0",
        cohort_id="test",
        manifest_sha256="sha256:" + "0" * 64,
        source={"name": "test", "revision": "1", "retrieved_at": "2026-09-04T00:00:00Z", "content_sha256": "sha256:" + "0" * 64},
        observed_horizon=20,
        unseen_horizon=100,
        targets=[t],
        exclusions=[],
    )

    # Candidate with identical 120 terms
    candidate_terms = [i for i in range(120)]
    has_leakage, reason = check_leakage_fingerprints(
        candidate_terms=candidate_terms,
        candidate_program_hashes=[],
        cohort=cohort,
    )
    assert has_leakage is True
    assert "A000045" in reason


def test_leakage_detection_matching_program_hash():
    prog_hash = "sha256:" + "p" * 64
    t = sample_target("A000290", terms=[str(i * i) for i in range(120)], program_fps=[prog_hash])
    cohort = BenchmarkCohort(
        schema_version="1.0",
        cohort_id="test",
        manifest_sha256="sha256:" + "0" * 64,
        source={"name": "test", "revision": "1", "retrieved_at": "2026-09-04T00:00:00Z", "content_sha256": "sha256:" + "0" * 64},
        observed_horizon=20,
        unseen_horizon=100,
        targets=[t],
        exclusions=[],
    )

    # Candidate with distinct terms but identical program hash
    candidate_terms = [i * 3 for i in range(120)]
    has_leakage, reason = check_leakage_fingerprints(
        candidate_terms=candidate_terms,
        candidate_program_hashes=[prog_hash],
        cohort=cohort,
    )
    assert has_leakage is True
    assert "program fingerprint match" in reason.lower()


def test_no_leakage_for_disjoint_sequence():
    t = sample_target("A000045", terms=[str(i) for i in range(120)])
    cohort = BenchmarkCohort(
        schema_version="1.0",
        cohort_id="test",
        manifest_sha256="sha256:" + "0" * 64,
        source={"name": "test", "revision": "1", "retrieved_at": "2026-09-04T00:00:00Z", "content_sha256": "sha256:" + "0" * 64},
        observed_horizon=20,
        unseen_horizon=100,
        targets=[t],
        exclusions=[],
    )

    candidate_terms = [1000 + i for i in range(120)]
    has_leakage, reason = check_leakage_fingerprints(
        candidate_terms=candidate_terms,
        candidate_program_hashes=["sha256:" + "x" * 64],
        cohort=cohort,
    )
    assert has_leakage is False
    assert reason is None
