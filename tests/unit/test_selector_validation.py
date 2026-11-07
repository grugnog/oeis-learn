"""The follow-up changes selection only and keeps development/fresh data separate."""
import copy
import time

import pytest

from oeis_learn.pipeline_validation.identity import seal
from oeis_learn.synthesis.admission import VisiblePrompt, source_identity
from oeis_learn.synthesis.contracts import identity

GOOD = '(program (return (add $index (int 0))))'
BAD = '(program (return (select (eq $index (int 46)) (int 0) $index)))'
PROMPT = VisiblePrompt(tuple(map(str, range(20))), tuple(map(str, range(20))))


def envelope(source):
    # A v1 fixture exercises the bank path as well as the v2 portfolio path.
    row = dict(source=source, source_sha256=source_identity(source), verified=True,
               visible_count=20, cost=1.)
    return dict(selected=dict(row, selection='visible-only/v1', seal_sha256=identity(row)),
                verification=dict(indices=list(PROMPT.indices), values=list(PROMPT.values),
                                  prompt_sha256=PROMPT.sha256), prompt_sha256=PROMPT.sha256)


def test_same_pool_selector_seal_and_existing_scorer_contract():
    from oeis_learn.generalization.selector_validation import rerank
    from oeis_learn.fresh.synthesis_policy import validate_selection_envelope, transfer_success
    original, pool = envelope(BAD), [envelope(BAD), envelope(GOOD)]
    before = copy.deepcopy(pool)
    chosen, observation = rerank(pool, original, PROMPT, deadline=time.monotonic()+10)
    assert observation['complete'] and observation['changed']
    assert chosen['selected']['source'] == GOOD
    assert pool == before
    validate_selection_envelope(chosen)
    indices = list(map(str, range(100)))
    execution = dict(source_sha256=source_identity(GOOD), indices=indices, values=indices, outcome='ok')
    assert transfer_success(chosen['selected'], indices, indices, execution, execution)
    changed = copy.deepcopy(chosen)
    changed['selected']['selector_policy'] = 'sha256:'+'0'*64
    with pytest.raises(ValueError):
        validate_selection_envelope(changed)


def test_timeout_preserves_selection_but_marks_trace_incomplete():
    from oeis_learn.generalization.selector_validation import rerank
    original = envelope(BAD)
    chosen, observation = rerank([original, envelope(GOOD)], original, PROMPT, deadline=0)
    assert chosen == original
    assert not observation['complete']


def test_subset_or_another_prompt_cannot_enter_comparison():
    from oeis_learn.generalization.selector_validation import rerank
    with pytest.raises(ValueError):
        rerank([envelope(GOOD)], envelope(BAD), PROMPT, deadline=float('inf'))
    bad = envelope(BAD); bad['prompt_sha256'] = 'sha256:'+'0'*64
    with pytest.raises(ValueError):
        rerank([bad], bad, PROMPT, deadline=float('inf'))


def test_fresh_draw_is_separate_and_legacy_is_unchanged():
    from oeis_learn.generalization.selector_validation import fresh_design, fresh_draw
    from oeis_learn.generalization.structural import amended_design, prior_protocol, draw
    historical = amended_design(prior_protocol())
    old = draw(historical)
    protocol = fresh_design()
    new = fresh_draw(protocol)
    assert new != old and len(new) == len(old) == 2288
    assert protocol['cells'] == historical['cells']
    assert protocol['seed'] != historical['seed']
    assert draw(historical) == old
    changed = seal(dict(protocol, seed=protocol['seed']+1))
    with pytest.raises(ValueError):
        fresh_draw(changed)


def test_probe_publication_preserves_training_role_and_rejects_selection():
    from oeis_learn.generalization.selector_validation import probe_views
    probes = [dict(id='p', role='training', indices=list(map(str, range(100))),
                   values=list(map(str, range(100))), source=GOOD, tokens=[1])]
    inventory = seal(dict(role='training', families=[dict(id='f', probes=['p'], rows=['r'])]))
    visible, targets, roster = probe_views(probes, inventory, research=False)
    assert set(visible[0]) == {'id','indices','values'} and len(visible[0]['values']) == 20
    assert targets[0]['role'] == 'training' and len(targets[0]['values']) == 100
    assert roster == {'p': dict(family='f', category='training')}
    probes[0]['role'] = 'selection'
    with pytest.raises(ValueError):
        probe_views(probes, inventory, research=False)


def test_summary_keeps_censoring_and_trace_failures_visible():
    from oeis_learn.generalization.selector_validation import paired_metrics
    roster = {str(i): dict(family='f', category='certified') for i in range(3)}
    old = {str(i): dict(success=False, complete=i!=2) for i in range(3)}
    new = {str(i): dict(success=i!=1, complete=True) for i in range(3)}
    result = paired_metrics(roster, old, new)
    assert result['original']['complete'] == 2 and result['alternative']['success'] == 2
    assert result['joint']['recoveries'] == 1 and result['joint']['count'] == 2
    assert result['delta_bounds_pp'] == pytest.approx([100/3, 200/3])
