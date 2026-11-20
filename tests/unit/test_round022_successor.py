"""The outcome-blind census and a single identical audit are the only carried scope."""
from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive.contracts import reference
from oeis_learn.pipeline_validation import progressive_round022_successor as successor
from oeis_learn.pipeline_validation.progressive_workstation_state import Blocked, write


def campaign_parts(root, folder, *, witnesses=('one', 'two'), attempts=16):
    plans, references = [], []
    for name, operation, sources in (('capability-census', 'census', ['cohort', 'exposure']),
                                     ('capability-language', 'audit-language', witnesses)):
        paths = []
        for source in sources:
            path = folder/source
            write(path, {'source': source})
            ref = reference(root, path)
            paths.append(ref['path'])
            references.append(dict(name=source, role='custodian', artifact=ref))
        reservation = dict(tokens=0, attempts=attempts if operation == 'audit-language' else 0)
        path = folder/(name+'.json')
        write(path, dict(operation=operation, seconds=3600, reservation=reservation,
                         image=folder.name, artifacts=dict(sources=paths)))
        ref = reference(root, path)
        references.append(dict(name='requests/'+name, role='custodian', artifact=ref))
        plans.append(dict(id=name, policy=ref['sha256'], caps=reservation, arm='refreshed',
                          stage='capability', seed=1))
    return plans, references


@pytest.mark.parametrize(('witnesses','attempts','authorized'), [
    (('one','two'), 16, True), (('two','one'), 16, False), (('one','three'), 16, False),
    (('one','two'), 17, False),
])
def test_replacement_checks_exact_order_inputs_and_reservation(tmp_path, witnesses, attempts, authorized):
    plans, refs = campaign_parts(tmp_path, tmp_path/'old')
    write(tmp_path/successor.BASE/'frozen/campaign.json', dict(payload=dict(tasks=plans, references=refs)))
    now, new_refs = campaign_parts(tmp_path, tmp_path/'new', witnesses=witnesses, attempts=attempts)
    approval = dict(replacement_reservation=dict(tokens=0, attempts=16))
    if authorized:
        successor.check_inputs(tmp_path, now, new_refs, approval)
    else:
        with pytest.raises(Blocked):
            successor.check_inputs(tmp_path, now, new_refs, approval)


def test_proof_cannot_launch_a_second_session(tmp_path):
    path=tmp_path/'other/campaign.json'
    write(path, {})
    payload=dict(references=[dict(name=successor.NAME, artifact={})])
    with pytest.raises(Blocked, match='only to the accepted round023'):
        with successor.research_scope(tmp_path, dict(payload=payload), path, SimpleNamespace()):
            pytest.fail('unexpected launch')
