"""A carried result has one source, one successor task and at most one debit."""

import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write, reference, seal
from oeis_learn.pipeline_validation import progressive_round024_successor as recovery
from oeis_learn.pipeline_validation.progressive_workstation_state import Blocked


def inputs(root, name, *, change=None):
    folder = root / name
    folder.mkdir()

    def put(filename, value):
        path = folder / filename
        atomic_write(path, value)
        return reference(root, path)

    model = put('weights', dict(weights='other' if change == 'weights' else 'original'))
    envelope = put(
        'model', seal(dict(kind='checkpoint', role='generator', data=dict(artifact=model, control='random'))))
    truth = put('truth', dict(values=[2 if change == 'truth' else 1]))
    slot = dict(task='other' if change == 'slot' else 'original',
                operation='generate',
                output='candidates',
                kind='candidate')
    req = put(
        'request',
        dict(image=name,
             operation='score',
             seconds=25,
             reservation=dict(attempts=3 if change == 'cap' else 2),
             artifacts=dict(model=[envelope['path']], score_truth=[truth['path']], candidates=[slot])))
    tasks = [
        dict(id=n,
             policy=req['sha256'],
             seed=3 if change == 'seed' else 1,
             arm='random',
             initializer={},
             slot='c',
             stage='c') for n in (*recovery.CARRIED, recovery.TASK)
    ]
    refs = [dict(name='requests/' + t['id'], artifact=req) for t in tasks]
    refs.extend(dict(name=v['path'], artifact=v) for v in (envelope, truth))
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    return recovery.input_scopes(root, tasks, refs, lambda **ref: checked_file(root, ref))


@pytest.mark.parametrize('change', [None, 'weights', 'truth', 'slot', 'cap', 'seed'])
def test_scopes_accept_requalified_image_and_relocated_identical_inputs_only(tmp_path, change):
    assert (inputs(tmp_path, 'original') == inputs(tmp_path, 'replacement', change=change)) == (change
                                                                                                is None)


def test_foreign_campaign_cannot_claim_replacement(tmp_path):
    path = tmp_path / 'foreign.json'
    atomic_write(path, {})
    campaign = dict(payload=dict(references=[dict(name=recovery.NAME, artifact={})]))
    with pytest.raises(Blocked, match='single authorized'):
        with recovery.research_scope(tmp_path, campaign, path, None, None):
            pytest.fail('unexpected authorization')


def test_only_reviewed_rehearsal_can_open_preparation(tmp_path):
    from oeis_learn.pipeline_validation.progressive_workstation_state import write
    out = tmp_path / recovery.REHEARSAL
    write(out / 'stage/proof.json', seal(dict(patch='other')))
    write(out / 'result.json', seal(dict(status='PASS', patch='other')))
    with pytest.raises(Blocked, match='four-member'):
        recovery.rehearsal(tmp_path)
