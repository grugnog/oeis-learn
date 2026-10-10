"""Adversarial acceptance for program-derived, scoped discovery."""

import numpy as np
import pytest
from oeis_learn.data.models import SequenceRef
from oeis_learn.discovery.numerical_validator import validate_numerical_relation
from oeis_learn.discovery.relation_identity import canonicalize_relation
from oeis_learn.discovery.pslq_solver import PslqRelationSolver
from oeis_learn.discovery.vector_search import VectorRelationSearcher


def test_one_index_coincidence_does_not_fit():
    a = list(range(1, 31))
    b = [n * n + 7 for n in range(1, 31)]
    c = [n * n * n + 9 for n in range(1, 31)]
    c[15] = a[15] + b[15]
    relation, _ = PslqRelationSolver().find_relation([a, b, c], term_index=15)
    assert relation is None


@pytest.mark.parametrize(
    "fit,val,final", [([], [1], [2]), ([0], [0], [2]), ([0], [1], []), ([0], [1], [1])]
)
def test_invalid_partitions_fail_closed(fit, val, final):
    with pytest.raises(ValueError):
        validate_numerical_relation(
            [SequenceRef("a"), SequenceRef("b")],
            [1, -1],
            {"a": [1] * 4, "b": [1] * 4},
            search_indices=fit,
            validation_indices=val,
            unseen_indices=final,
        )


def test_canonicalization_never_truncates():
    with pytest.raises(ValueError):
        canonicalize_relation([SequenceRef("a"), SequenceRef("b")], [1])


def test_rank_nullity_and_source_identity_are_measured():
    result = validate_numerical_relation(
        [SequenceRef("a"), SequenceRef("b")],
        [1, -1],
        {"a": [10**100 + n for n in range(12)], "b": [10**100 + n for n in range(12)]},
        search_indices=[0, 1, 2],
        validation_indices=[3, 4, 5],
        unseen_indices=[6, 7, 8],
    )
    assert result.outcome == "VERIFIED"
    assert (result.matrix_rank, result.matrix_nullity) == (1, 1)
    assert result.source_sha256 != "sha256:" + "0" * 64
    assert result.minimal_support is None


def test_retrieval_is_ranked_and_order_independent():
    vectors = {str(n): np.array([np.cos(n), np.sin(n)]) for n in range(10)}
    s = VectorRelationSearcher(eps_distance=2)
    a = s.search_additive_triples(vectors, max_candidates=5)
    b = s.search_additive_triples(dict(reversed(list(vectors.items()))), max_candidates=5)
    assert [(c.sequences, c.vector_distance, c.candidate_id) for c in a] == [
        (c.sequences, c.vector_distance, c.candidate_id) for c in b
    ]
    assert [c.vector_distance for c in a] == sorted(c.vector_distance for c in a)


from copy import deepcopy
from oeis_learn.discovery.certificates import make_certificate
from oeis_learn.discovery.certificate_checker import check_certificate

POLY_A = "(pure (return (mul (index) (add (index) (const 1)))))"
POLY_B = "(pure (return (add (mul (index) (index)) (index))))"
STATE_A = "(stream (init) (next (set 0 (sub (const 1) (get 0))) (return (get 0))))"
STATE_B = "(stream (init) (next (set 0 (xor (get 0) (const 1))) (return (get 0))))"
REC_A = "(stream (init (set 1 (const 1))) (next (set 2 (get 0)) (set 0 (get 1)) (set 1 (add (get 2) (get 1))) (return (get 2))))"
REC_B = "(stream (init (set 1 (const 1))) (next (set 3 (get 1)) (set 1 (add (get 0) (get 1))) (set 2 (get 0)) (set 0 (get 3)) (return (get 2))))"
LOOP = "(pure (set 0 (index)) (while (gt (get 0) (const 0)) (set 0 (sub (get 0) (const 1)))) (return (get 0)))"


@pytest.mark.parametrize(
    "kind,sources,domain,template",
    [
        ("polynomial_program", [POLY_A, POLY_B], None, None),
        ("linear_recurrence", [REC_A, REC_B], None, None),
        ("finite_state", [STATE_A, STATE_B], None, None),
        ("bounded_machine", [POLY_A, POLY_B], {"start": 0, "end": 5, "excluded_indices": []}, None),
        (
            "loop_invariant",
            [LOOP],
            {"start": 0, "end": 100, "excluded_indices": []},
            {
                "invariants": ["(ge (get 0) (const 0))"],
                "ranking": "(get 0)",
                "postcondition": "(eq (get 0) (const 0))",
            },
        ),
    ],
)
def test_certificate_families_fresh_replay_and_mutation(kind, sources, domain, template):
    certificate = make_certificate(
        kind, sources, [1] if len(sources) == 1 else [1, -1], domain=domain, template=template
    )
    result = check_certificate(certificate)
    assert result["result"] == "verified", result
    assert result["wasm_theorem"] is False
    for field in ("payload", "compiler", "statement", "template"):
        mutated = deepcopy(certificate)
        mutated[field] = {}
        assert check_certificate(mutated)["result"] != "verified"


def test_nonterminating_ranking_cannot_certify():
    source = LOOP.replace("(sub (get 0) (const 1))", "(add (get 0) (const 1))")
    with pytest.raises(ValueError, match="unproved"):
        make_certificate(
            "loop_invariant",
            [source],
            [1],
            domain={"start": 0, "end": 100, "excluded_indices": []},
            template={
                "invariants": ["(ge (get 0) (const 0))"],
                "ranking": "(get 0)",
                "postcondition": "(eq (get 0) (const 0))",
            },
        )


from oeis_learn.discovery.transforms import Transform


@pytest.mark.parametrize(
    "transform,index,expected",
    [
        (Transform("shift", ("a",), (1,)), 3, 4),
        (Transform("reindex", ("a",), (2, 1)), 3, 7),
        (Transform("difference", ("b",)), 3, 7),
        (Transform("sum", ("a", "b")), 3, 12),
        (Transform("product", ("a", "b")), 3, 27),
        (Transform("interleave", ("a", "b")), 3, 1),
        (Transform("convolution", ("a", "b")), 3, 6),
        (Transform("composition", ("a", "b")), 3, 9),
    ],
)
def test_transforms_independent_positive_and_domain_negative(transform, index, expected):
    programs = {"a": list(range(128)), "b": [n * n for n in range(128)]}
    value, cost = transform.evaluate(index, programs)
    assert value == expected and cost > 0
    with pytest.raises((ValueError, TimeoutError)):
        transform.evaluate(4095, programs)


def test_sealed_novelty_cannot_feed_strict_worker_or_mutate_claim():
    from oeis_learn.discovery.novelty import seal_proposal, analyze_novelty

    proposal = seal_proposal(
        {"relation": ["a", "b"], "coefficients": ["1", "-1"], "evidence_kind": "empirical_relation"}
    )
    with pytest.raises(ValueError, match="analysis"):
        analyze_novelty(proposal, lambda p: None, role="learner")
    original = deepcopy(proposal)

    def lookup(copy):
        copy["coefficients"] = ["999"]
        return {
            "relation_known": True,
            "implementation_known": False,
            "source_reference": "example:known-formula",
        }

    assert (
        analyze_novelty(proposal, lookup, role="analysis")["classification"] == "new_implementation"
    )
    assert proposal == original
    assert (
        analyze_novelty(proposal, lambda p: None, role="analysis")["classification"]
        == "novelty_unknown"
    )


def test_alpha_renamed_features_and_semantic_mutation():
    from oeis_learn.discovery.program_features import program_features

    a = program_features("(pure (set 4 (index)) (return (get 4)))")
    b = program_features("(pure (set 7 (index)) (return (get 7)))")
    assert a["structure_sha256"] == b["structure_sha256"]
    assert a["source_sha256"] != b["source_sha256"]
    changed = program_features("(pure (set 7 (add (index) (const 1))) (return (get 7)))")
    assert b["structure_sha256"] != changed["structure_sha256"]


def test_macro_expansion_cycles_and_trap_order():
    from oeis_learn.discovery.macro_mining import expand_macros
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    definition = "(div (const 1) (const 0))"
    dictionary = {
        "version": "typed-macros/v1",
        "definitions": {
            "m0": {"source": definition, "type": "wide", "occurrences": 4, "token_benefit": 8}
        },
        "archive_root_sha256": digest("fixture-training-root"),
        "net_token_benefit": 8,
    }
    dictionary["dictionary_sha256"] = digest(dictionary)
    assert (
        expand_macros("(pure (return (macro m0)))", dictionary)
        == "(pure (return " + definition + "))"
    )
    cyclic = deepcopy(dictionary)
    cyclic["definitions"]["m0"]["source"] = "(macro m0)"
    cyclic["dictionary_sha256"] = digest(cyclic, "dictionary_sha256")
    with pytest.raises(ValueError, match="cyclic"):
        expand_macros("(macro m0)", cyclic)


def test_program_archive_discovery_owns_provenance_and_keeps_implementations(tmp_path):
    from tests.unit.test_foundation_archive_curriculum import generic_fixture
    from oeis_learn.data.generic_programs import default_config
    from oeis_learn.discovery.program_archive import ProgramArchive
    from oeis_learn.sandbox.runner import FoundationRunner
    from oeis_learn.discovery.program_features import ProgramFeatureIndex
    from oeis_learn.discovery.pipeline import run_program_discovery

    sample, outputs, holdouts = generic_fixture()
    archive = ProgramArchive(tmp_path / "archive", holdouts=holdouts)
    with FoundationRunner(tmp_path / "runtime", backend="python_wasmtime") as runner:
        key = archive.admit(
            sample["canonical_source"],
            outputs[:20],
            outputs,
            origin="generic_sample",
            provenance={"config": default_config(), "sample": sample},
            runner=runner,
            request_id="phase9-generated",
        )
    index = ProgramFeatureIndex(archive)
    index.validate()
    assert index.rows[0]["key"] == key and index.rows[0]["genealogy"]["sample"] == sample
    report = run_program_discovery(
        archive,
        fitting_indices=range(8),
        validation_indices=range(8, 16),
        final_indices=range(16, 24),
    )
    assert report["information_policy"] == "verified_training_programs_only"
    assert report["external_lookup_performed"] is False
    assert report["archive_root_sha256"] == archive.freeze()["root_sha256"]
    index.rows[0]["source"] = "i256.const 999"
    with pytest.raises(ValueError):
        index.validate()


@pytest.mark.parametrize("size", [128, 256, 512])
def test_ranked_scaling_planted_and_shuffled_negative_controls(size):
    rng = np.random.default_rng(9002 + size)
    width = 32
    va = rng.normal(size=width)
    vb = va + rng.normal(scale=0.01, size=width)
    vectors = {f"p{i:04d}": rng.normal(size=width) for i in range(size - 3)}
    vectors.update(
        {
            "planted_a": va,
            "planted_b": vb,
            "planted_c": va / np.linalg.norm(va) + vb / np.linalg.norm(vb),
        }
    )
    search = VectorRelationSearcher(eps_distance=1e-7, neighbors=8, seconds=10)
    retained = search.search_additive_triples(vectors, max_candidates=50)
    assert any(c.sequences == ("planted_a", "planted_b", "planted_c") for c in retained)
    assert all(set(c.sequences) == {"planted_a", "planted_b", "planted_c"} for c in retained)
    assert search.last_metrics["pairs"] <= 4096
    assert search.last_metrics["distance_evaluations"] <= size * size + 4096 * size
    assert search.last_metrics["block_bytes"] <= 128 * width * 8
    permuted = dict(reversed(list(vectors.items())))
    assert [(c.sequences, c.vector_distance) for c in search.search_additive_triples(permuted)] == [
        (c.sequences, c.vector_distance) for c in retained
    ]


def test_certificate_resealed_semantics_compiler_and_coefficient_mutations():
    from oeis_learn.experiments.artifacts import compute_canonical_digest as digest

    original = make_certificate("polynomial_program", [POLY_A, POLY_B], [1, -1])
    for mutation in ("coefficient", "compiler", "assumption", "source"):
        changed = deepcopy(original)
        if mutation == "coefficient":
            changed["statement"]["coefficients"][1] = "-2"
        elif mutation == "compiler":
            changed["compiler"]["implementation_sha256"] = digest("other-compiler")
        elif mutation == "assumption":
            changed["statement"]["assumptions"] = []
        else:
            changed["statement"]["programs"][0]["source"] = POLY_A.replace("(const 1)", "(const 2)")
        changed["statement_sha256"] = digest(changed["statement"])
        changed["certificate_sha256"] = digest(changed, "certificate_sha256")
        assert check_certificate(changed)["result"] != "verified"
