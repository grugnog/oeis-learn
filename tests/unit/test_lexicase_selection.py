"""Unit tests for down-sampled lexicase rollout filtering."""

import pytest
from oeis_learn.rl.prompt_weighting import filter_downsampled_lexicase


def test_lexicase_filters_constant_compromise_solutions():
    # Target: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    target = list(range(10))

    # Candidate 0: Constant shortcut [4, 4, 4, 4, 4, 4, 4, 4, 4, 4] (low average MSE, fails non-zero points)
    cand_const = [4] * 10
    # Candidate 1: Exact specialist [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    cand_exact = list(range(10))
    # Candidate 2: Partial specialist [0, 1, 2, 3, 0, 0, 0, 0, 0, 0]
    cand_partial = [0, 1, 2, 3, 0, 0, 0, 0, 0, 0]

    batch_outputs = [cand_const, cand_exact, cand_partial]

    lex_res = filter_downsampled_lexicase(
        candidates_outputs=batch_outputs,
        target_terms=target,
        subsample_size=5,
        seed=42,
    )

    assert len(lex_res.test_case_indices) > 0
    # Exact candidate must be among survivors
    assert 1 in lex_res.surviving_candidates
    # Constant candidate is eliminated on non-4 test points
    if any(tc != 4 for tc in lex_res.test_case_indices):
        assert 0 not in lex_res.surviving_candidates
