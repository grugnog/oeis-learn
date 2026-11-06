"""Audit source pools and attach outcomes only after selection."""
import copy
import json
import zipfile

import pytest

from oeis_learn.synthesis.admission import source_identity

SOURCE = '(program (return $index))'
HASH = source_identity(SOURCE)


def score(success=True, complete=True, sha=HASH):
    return dict(success=success, complete=complete, candidate_sha256=sha)


def test_known_unknown_censored_and_conflicting_outcome_joins():
    from oeis_learn.generalization.selector_audit import known_outcome
    rows = {'equal_budget': score(), 'absent': score()}
    assert known_outcome(HASH, rows)['success'] is True
    assert known_outcome('sha256:'+'0'*64, rows)['status'] == 'UNSCORED'
    assert known_outcome(HASH, {'raw': score(False, False)})['status'] == 'UNSCORED'
    with pytest.raises(ValueError, match='conflicting'):
        known_outcome(HASH, dict(rows, rescue=score(False)))


def record():
    return {'archive': {'retained': [], 'omitted': [{'id': HASH}]},
            'details': [{'native': {'assignments': [{'source': SOURCE, 'source_sha256': HASH}]}}]}


def test_pool_recovers_omitted_eligible_sources_not_unadmitted_proposals():
    from oeis_learn.generalization.selector_audit import candidate_pool
    row = record(); row['details'][0]['native']['assignments'].append(
        {'source': '(program (return (int 0)))', 'source_sha256': source_identity('(program (return (int 0)))')})
    before = copy.deepcopy(row)
    assert candidate_pool(row) == [{'source': SOURCE, 'source_sha256': HASH}]
    assert row == before


def test_pool_rejects_missing_or_tampered_eligible_source():
    from oeis_learn.generalization.selector_audit import candidate_pool
    row = record(); row['details'] = []
    with pytest.raises(ValueError, match='missing eligible'):
        candidate_pool(row)
    row = record(); row['details'][0]['native']['assignments'][0]['source'] += ' '
    with pytest.raises(ValueError, match='source hash'):
        candidate_pool(row)


@pytest.mark.parametrize('names', [['../plan.json'], ['plan.json', 'plan.json'], ['weights.pt']])
def test_zip_rejects_unexpected_paths_and_duplicates(tmp_path, names):
    from oeis_learn.generalization.selector_audit import read_details
    path = tmp_path/'bad.zip'
    with zipfile.ZipFile(path, 'w') as z:
        for name in names:
            z.writestr(name, '{}')
    with pytest.raises(ValueError):
        read_details(path)


def test_json_duplicate_and_nonfinite_rejection():
    from oeis_learn.generalization.selector_audit import json_record
    for value in ['{"a":1,"a":2}', '{"a":NaN}', '{"a":1e999}']:
        with pytest.raises(ValueError):
            json_record(value)


def test_original_choice_must_reproduce():
    from oeis_learn.generalization.selector_audit import check_original
    candidates = [{'source': SOURCE, 'source_sha256': HASH}]
    check_original(candidates, HASH)
    with pytest.raises(ValueError, match='original selection'):
        check_original(candidates, None)
