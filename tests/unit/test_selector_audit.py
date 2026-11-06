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


def synthetic_records():
    """Closed small authority fixture; not production/HIP qualification evidence."""
    from oeis_learn.generalization.selector_audit import ARMS, CONTROLS, paths
    from oeis_learn.pipeline_validation.identity import seal, digest
    def entry(value):
        return value, digest(value), len(json.dumps(value).encode())
    study = seal(dict(source=digest('source'), arms=list(ARMS)))
    plan = seal(dict(source=study['source'], arms=list(ARMS)))
    inputs = dict(experiment=study, scoring={}, training={}, outcomes={}, roster={'task': {'category':'certified'}}, stages={})
    records, receipts = {'plan.json': entry(plan)}, []
    for arm in ARMS:
        gp, sp, tp = paths(arm)
        outcomes = {c: {'task': True} for c in CONTROLS}
        scoring = seal(dict(study=study['identity'], roster=inputs['roster'], outcomes=outcomes,
                            scored_cases={'task': {c: score() for c in CONTROLS}}))
        training = seal(dict(arm=arm))
        inputs['scoring'][arm], inputs['training'][arm], inputs['outcomes'][arm] = scoring['identity'], training['identity'], outcomes
        for name, value in [(gp, {}), (sp, scoring), (tp, training)]:
            records[name] = entry(value)
        for stage, names in [(f'evaluate-{arm}', [gp, sp]), (f'train-{arm}', [tp])]:
            inputs['stages'][stage] = dict(producer=digest(stage), binding=digest(stage+'binding'), elapsed_seconds=1)
            receipts.append(dict(stage=stage, **inputs['stages'][stage], artifacts=[
                dict(path=n.split('/',1)[1], sha256=records[n][1], bytes=records[n][2]) for n in names]))
    workflow = seal(dict(plan=plan['identity'], receipts=receipts))
    records['workflow.json'] = entry(workflow)
    return records, inputs, {'plan': plan['identity']}


def test_original_producer_and_report_binding_counterexamples():
    from oeis_learn.generalization.selector_audit import verify_details, paths
    from oeis_learn.pipeline_validation.identity import seal
    records, inputs, evidence = synthetic_records()
    verify_details(records, inputs, evidence)
    changed = copy.deepcopy(records)
    name = paths('D')[0]
    changed[name] = (changed[name][0], 'sha256:'+'0'*64, changed[name][2])
    with pytest.raises(ValueError, match='artifact mismatch'):
        verify_details(changed, inputs, evidence)
    changed = copy.deepcopy(records)
    workflow = changed['workflow.json'][0]
    workflow['receipts'][0]['producer'] = 'sha256:'+'0'*64
    changed['workflow.json'] = (seal(workflow), '', 0)
    with pytest.raises(ValueError, match='stage differs'):
        verify_details(changed, inputs, evidence)
    changed = copy.deepcopy(records)
    name = paths('B')[1]
    scoring = changed[name][0]
    scoring['scored_cases']['task']['raw']['success'] = False
    changed[name] = (scoring, changed[name][1], changed[name][2])
    with pytest.raises(ValueError, match='identity mismatch'):
        verify_details(changed, inputs, evidence)


def envelope(source, values):
    from oeis_learn.synthesis.admission import VisiblePrompt
    from oeis_learn.pipeline_validation.identity import digest
    sha = source_identity(source)
    prompt = VisiblePrompt(tuple(map(str, range(20))), tuple(values))
    verification = dict(indices=list(prompt.indices), values=list(prompt.values), admitted=True,
                        prompt_sha256=prompt.sha256, source_sha256=sha)
    selected = dict(source=source, source_sha256=sha, selection='visible-only/v2',
                    provenance=dict(prompt=prompt.sha256, assignment={}, original_source_sha256=sha,
                                    transformed_source_sha256=sha, verification=digest(verification)))
    selected['seal_sha256'] = digest({k:v for k,v in selected.items() if k != 'selection'})
    return dict(selected=selected, verification=verification, assignment={}, original_source=source,
                transformed_source=source, original_source_sha256=sha, prompt_sha256=prompt.sha256)


def test_audit_task_does_not_use_outcomes_and_exports_no_visible_terms():
    from oeis_learn.generalization.selector_audit import audit_task, CONTROLS
    import time
    # Both fit visible zeros, but one contains an unobserved branch. The correct
    # outcome of the alternative is intentionally unknown; no 100-term execution.
    n = '123456789012345678901234567890'
    short = f'(program (return (select (lt $index (int 20)) (int {n}) (int 0))))'
    long = f'(program (return (add (add (int {n}) (mul $index (int 0))) (mul $index (int 0)))))'
    assert len(short) < len(long)
    ids = [source_identity(s) for s in (short, long)]
    portfolio = dict(archive=dict(omitted=[], retained=[dict(id=h, source=s) for h,s in zip(ids,(short,long))]), details=[])
    record = dict(details={'correct':portfolio}, selections={'equal_budget':envelope(short, [n]*20)})
    scored = {c:score(False, sha=ids[0]) for c in CONTROLS}
    first = audit_task(record, scored, deadline=time.monotonic()+10)
    assert first['alternative'] == ids[1]
    assert first['alternative_outcome']['status'] == 'UNSCORED'
    assert n not in json.dumps(first)
    changed = {c:score(True, sha=ids[0]) for c in CONTROLS}
    second = audit_task(record, changed, deadline=time.monotonic()+10)
    assert first['alternative'] == second['alternative']
    assert first['candidates'] == second['candidates']
    scored['raw']['reason'] = '/private/SECRET-PATH: arbitrary native error'
    private = audit_task(record, scored, deadline=time.monotonic()+10)
    assert 'SECRET-PATH' not in json.dumps(private)
    assert private['original_controls']['raw']['reason'] is None
    assert private['original_controls']['raw']['reason_omitted'] is True


def test_output_exclusivity_precedes_reading_inputs(tmp_path):
    from oeis_learn.generalization.selector_audit import run
    output = tmp_path/'existing'; output.mkdir(); marker = output/'keep'; marker.write_text('original')
    with pytest.raises(ValueError, match='fresh'):
        run(tmp_path/'missing-report', tmp_path/'missing-details', output)
    assert marker.read_text() == 'original'


def test_expected_bundle_digest_checked_before_report_verification(tmp_path):
    from oeis_learn.generalization.selector_audit import run
    details = tmp_path/'details'; details.write_bytes(b'changed bytes')
    with pytest.raises(ValueError, match='expected original bytes'):
        run(tmp_path/'absent', details, tmp_path/'output', expected_details_sha256='sha256:'+'0'*64)
    assert not (tmp_path/'output').exists()
