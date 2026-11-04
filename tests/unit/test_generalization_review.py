"""Independent per-anchor timelines and exhaustive production cycle offsets."""
import copy,json
from pathlib import Path
import pytest


def test_anchor_acquisition_grades_incidental_and_failed_commit():
    from oeis_learn.generalization.review import ReviewQueue
    q=ReviewQueue({'a':'family'})
    for event in json.loads(Path('tests/fixtures/generalization/pac-policy.json').read_text())['queue']:
        incidental=event['kind']=='incidental'
        q.commit(update=event['update'],event_id=f"update/{event['update']}",
                 designated=None if incidental else 'a',grade=event['grade'],exposed=['a'])
        state=q.anchors['a']
        assert state['acquired'] and state['queue']==event['queue'] and state['due_update']==event['due']
    before=copy.deepcopy(q.snapshot())
    with pytest.raises(ValueError):q.commit(update=2648,event_id='wrong-anchor',designated='other',grade=True,exposed=['a'])
    assert q.snapshot()==before
    with pytest.raises(ValueError):q.commit(update=2136,event_id='duplicate',designated='a',grade=True,exposed=['a'])
    assert ReviewQueue.restore(json.loads(json.dumps(q.snapshot())),anchors={'a':'family'}).snapshot()==before


@pytest.mark.parametrize('offset',range(32))
def test_128_simultaneous_anchors_all_cycle_offsets(offset):
    from oeis_learn.generalization.review import ReviewQueue,review_slot
    q=ReviewQueue({f'a{i:03}':f'f{i//4:02}' for i in range(128)})
    last={};max_delay=0
    for update in range(1,8705):
        if review_slot(update-1+offset):
            anchor=q.choose(update)
            if anchor is not None:
                a=q.anchors[anchor];max_delay=max(max_delay,update-a['due_update'])
                if anchor in last:assert update-last[anchor]<=8704
                last[anchor]=update
                q.commit(update=update,event_id=f'update/{update}',designated=anchor,
                         grade=False if a['acquired'] else None,exposed=[anchor])
    assert len(last)==128 and max_delay<=512
    assert len(q.anchors)==128 and all(a['queue']==0 for a in q.anchors.values())
    # No claim about service after training terminates.
    q.finish();assert q.choose(100000) is None and q.pending()['bound_applies'] is False


def test_capacity_no_due_and_incidental_cannot_promote():
    from oeis_learn.generalization.review import ReviewQueue,review_slot
    with pytest.raises(ValueError):ReviewQueue({str(i):'f' for i in range(129)})
    q=ReviewQueue({'a':'f','b':'f'})
    assert q.choose(1)=='a'
    q.commit(update=1,event_id='u1',designated=None,grade=None,exposed=['a','b'])
    assert q.choose(2) is None
    with pytest.raises(ValueError):q.commit(update=2,event_id='u2',designated=None,grade=True,exposed=['a'])
    assert sum(review_slot(i) for i in range(32))==8
    assert 24/32*.25==.1875


def test_no_due_fallback_is_uniform_family_not_kernel_credit():
    import random
    from oeis_learn.generalization.review import ReviewQueue,review_choice
    q=ReviewQueue({'a':'fa','b':'fb'})
    q.commit(update=1,event_id='acquire',designated=None,grade=None,exposed=['a','b'])
    rng=random.Random(1);before=q.snapshot()
    choices=[review_choice(q,completed_updates=24,families=['fa','fb'],rng=rng) for _ in range(20)]
    assert {c['family'] for c in choices}=={'fa','fb'}
    assert all(c['channel']=='fallback' and c['anchor'] is None for c in choices)
    assert q.snapshot()==before
