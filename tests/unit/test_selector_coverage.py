"""Independent expectations for an experimental, visible-only selector."""
import copy
from dataclasses import replace

import pytest

from oeis_learn.synthesis.admission import VisiblePrompt, source_identity
from oeis_learn.synthesis.contracts import ResourceProfile
from oeis_learn.synthesis.reference import evaluate

INDICES = tuple(map(str, range(20)))
PROMPT = VisiblePrompt(INDICES, INDICES)
GOOD = '(program (return (add $index (int 0))))'
BAD = '(program (local $b0 bigint) (if (eq $index (int 46)) (then (set $b0 (int 0))) (else (set $b0 $index))) (return (get $b0)))'


def candidate(source):
    return dict(source=source, source_sha256=source_identity(source))


def test_reference_observation_preserves_legacy_and_arithmetic():
    seen, legacy = {}, []
    def branch(site, value):
        seen.setdefault(site, set())
        if value is not None:
            seen[site].add(value)
    before = evaluate(BAD, INDICES)
    after = evaluate(BAD, INDICES, observer=lambda *event: legacy.append(event), branch_observer=branch)
    assert before == after
    assert list(seen.values()) == [{False}]
    assert legacy == [('branch', False)] * 20


def test_nested_unreached_sites_and_local_dependent_conditions():
    from oeis_learn.generalization.selector_coverage import observe
    source = '(program (local $b0 bigint) (set $b0 $index) (if (eq (get $b0) (int 46)) (then (if (lt $index (int 3)) (then (set $b0 (int 1))) (else (set $b0 (int 2))))) (else)) (return (get $b0)))'
    result = observe(source, PROMPT)
    assert result['complete'] is True
    assert result['branch_sites'] == 2
    assert result['missing_branch_edges'] == 3


def test_equal_syntax_at_different_sites_stays_distinct_and_select_is_eager():
    from oeis_learn.generalization.selector_coverage import observe
    piecewise = '(program (return (select (lt $index (int 10)) $index (add $index (int 0)))))'
    assert observe(piecewise, PROMPT)['missing_branch_edges'] == 0
    source = '(program (local $b0 bigint) (if (lt $index (int 10)) (then) (else)) (if (lt $index (int 10)) (then) (else)) (return $index))'
    assert observe(source, PROMPT)['branch_sites'] == 2
    eager = '(program (return (select (true) $index (mul (int 255) (int 255)))))'
    limits = replace(ResourceProfile(), intermediate_bits=8)
    result = observe(eager, PROMPT, limits=limits)
    assert result['complete'] is False
    assert result['reason'] == 'integer_limit'


def test_zero_iteration_loop_leaves_nested_branch_unobserved():
    from oeis_learn.generalization.selector_coverage import observe
    source = '(program (local $c0 control) (loop $c0 (control 0) (if (lt $index (int 10)) (then) (else))) (return $index))'
    assert observe(source, PROMPT)['missing_branch_edges'] == 2


def test_ranking_is_score_blind_deterministic_and_does_not_mutate():
    from oeis_learn.generalization.selector_coverage import rank
    rows = [candidate(BAD), candidate(GOOD)]
    before = copy.deepcopy(rows)
    result = rank(rows, PROMPT, original=source_identity(BAD))
    assert result['selected'] == source_identity(GOOD)
    assert result['complete'] is True
    assert rank(list(reversed(rows)), PROMPT, original=source_identity(BAD))['selected'] == result['selected']
    assert rows == before
    for field in ('success', 'family', 'category', 'hidden_values'):
        poisoned = dict(rows[0], **{field: True})
        with pytest.raises(ValueError, match='candidate fields'):
            rank([poisoned], PROMPT, original=poisoned['source_sha256'])


def test_unavailable_trace_never_wins_and_preserves_original():
    from oeis_learn.generalization.selector_coverage import rank
    result = rank([candidate(BAD), candidate(GOOD)], PROMPT, original=source_identity(BAD),
                  limits=replace(ResourceProfile(), aggregate_reference_steps=1))
    assert result['complete'] is False
    assert result['selected'] == source_identity(BAD)


def test_exact_integers_mismatch_and_visible_count():
    from oeis_learn.generalization.selector_coverage import observe
    n = 10**80
    prompt = VisiblePrompt(INDICES, tuple(str(i+n) for i in range(20)))
    source = f'(program (return (add $index (int {n}))))'
    assert observe(source, prompt)['complete'] is True
    assert observe(source, PROMPT)['reason'] == 'visible_mismatch'
    with pytest.raises(ValueError, match='20'):
        observe(GOOD, VisiblePrompt(('0',), ('0',)))
