"""Unit tests for batch-level cross-input mutual information proxy."""

import pytest
from oeis_learn.rl.reward import compute_cross_input_mutual_information_proxy


def test_mutual_information_proxy_distinct_sequences():
    # Minibatch of 3 distinct dynamic sequence outputs
    seq1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    seq2 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    seq3 = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

    batch = [seq1, seq2, seq3]
    mi_scores = compute_cross_input_mutual_information_proxy(batch, temperature=0.1)

    assert len(mi_scores) == 3
    # Outputs are distinguishable -> cosine similarity is lower -> R_MI is relatively high (less negative)
    for score in mi_scores:
        assert isinstance(score, float)


def test_mutual_information_proxy_identical_constant_penalty():
    # Minibatch where all tasks emit identical constant shortcuts [16, 16, 16...]
    const_seq = [16] * 10
    batch_identical = [const_seq, const_seq, const_seq]

    mi_identical = compute_cross_input_mutual_information_proxy(batch_identical, temperature=0.1)

    # Distinct batch
    batch_distinct = [
        [0, 1, 2, 3, 4, 5],
        [1, 2, 4, 8, 16, 32],
        [0, 0, 1, 1, 2, 2],
    ]
    mi_distinct = compute_cross_input_mutual_information_proxy(batch_distinct, temperature=0.1)

    # Identical constant outputs must receive much harsher penalties (lower score) than distinct outputs
    assert sum(mi_identical) / 3.0 < sum(mi_distinct) / 3.0
