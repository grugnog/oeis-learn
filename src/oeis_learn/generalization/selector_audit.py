"""Offline, exploratory replay of original PAC candidate selection.

No generation, fitting, checkpoint loading, hidden-term execution or production
policy dispatch. Outcome records are attached after a closed visible-only rank.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import stat
import time
import zipfile

from oeis_learn.pipeline_validation.identity import digest, file_hash, seal, write
from oeis_learn.synthesis.admission import VisiblePrompt, source_identity
from oeis_learn.fresh.synthesis_policy import validate_selection_envelope
from .selector_coverage import POLICY, rank

ARMS = ('B', 'C', 'D')
CONTROLS = ('raw', 'rescue', 'equal_budget', 'shuffled', 'absent', 'template', 'bank')
REASONS = frozenset(('success', 'continuation_mismatch', 'unsat_in_scope',
    'resource_exhausted:timeout', 'token_cap_without_eos', 'grammar_dead_end',
    'deadline', 'owner_deadline', 'unsupported', 'invalid_source', 'bank_application',
    'scoring_incomplete', 'not_attempted', 'selection_record_missing', 'legacy_arm_evidence_missing'))


def paths(arm):
    prefix = f'evaluate-{arm}/execution/evaluation/{arm}'
    return (prefix+'/generation/role/generation/selections.json',
            prefix+'/scoring/role/scoring.json', f'train-{arm}/execution/result.json')


def json_record(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def number(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('nonfinite JSON number')
        return value
    def constant(_):
        raise ValueError('nonfinite JSON constant')
    return json.loads(data, object_pairs_hook=pairs, parse_float=number, parse_constant=constant)


def read_details(path):
    expected = {'plan.json', 'workflow.json'} | {p for a in ARMS for p in paths(a)}
    if Path(path).stat().st_size > 256 << 20:
        raise ValueError('detail ZIP byte limit')
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) != len(expected) or {i.filename for i in infos} != expected:
            raise ValueError('exact eleven detail ZIP members required; duplicates/extra paths forbidden')
        if sum(i.file_size for i in infos) > 256 << 20:
            raise ValueError('detail ZIP expansion limit')
        records = {}
        for info in infos:
            if info.file_size > 128 << 20 or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError('detail member byte/type limit')
            data = archive.read(info)
            records[info.filename] = (json_record(data), 'sha256:'+hashlib.sha256(data).hexdigest(), len(data))
        return records


def verify_details(records, inputs, evidence):
    plan, workflow = (records[n][0] for n in ('plan.json', 'workflow.json'))
    if (seal(plan) != plan or seal(workflow) != workflow or workflow['plan'] != plan['identity']
            or evidence['plan'] != plan['identity'] or plan['source'] != inputs['experiment']['source']
            or plan['arms'] != list(ARMS) or inputs['experiment']['arms'] != list(ARMS)):
        raise ValueError('plan/workflow/report binding mismatch')
    entries = {r['stage']: r for r in workflow['receipts']}
    if len(entries) != len(workflow['receipts']):
        raise ValueError('duplicate stage receipt')
    for arm in ARMS:
        for name in paths(arm):
            stage, relative = name.split('/', 1)
            if any(entries[stage][key] != inputs['stages'][stage][key]
                   for key in ('producer', 'binding', 'elapsed_seconds')):
                raise ValueError('original stage differs from report')
            matches = [r for r in entries[stage]['artifacts'] if r['path'] == relative]
            if len(matches) != 1 or matches[0]['sha256'] != records[name][1] or matches[0]['bytes'] != records[name][2]:
                raise ValueError('original producer artifact mismatch')
        _, scoring_path, training_path = paths(arm)
        scoring, training = records[scoring_path][0], records[training_path][0]
        if (seal(scoring) != scoring or seal(training) != training
                or scoring['identity'] != inputs['scoring'][arm] or training['identity'] != inputs['training'][arm]
                or scoring['roster'] != inputs['roster'] or scoring['outcomes'] != inputs['outcomes'][arm]
                or scoring['study'] != inputs['experiment']['identity']):
            raise ValueError('original analysis identity mismatch')
        if set(scoring['outcomes']) != set(CONTROLS) or set(scoring['scored_cases']) != set(inputs['roster']):
            raise ValueError('original scoring population/control mismatch')
        for task, cases in scoring['scored_cases'].items():
            if set(cases) != set(CONTROLS):
                raise ValueError('missing scored controls')
            for control, case in cases.items():
                if type(case['complete']) is not bool or type(case['success']) is not bool:
                    raise ValueError('boolean scored outcome required')
                value = case['success'] if case['complete'] else None
                if value is not scoring['outcomes'][control][task]:
                    raise ValueError('scored cases disagree with report outcomes')
    return plan, workflow


def candidate_pool(portfolio):
    archive = portfolio['archive']
    rows = archive['retained'] + archive['omitted']
    if len(rows) > 64 or len(portfolio['details']) > 8:
        raise ValueError('original eight-attempt candidate limit')
    eligible = {row['id'] for row in rows}
    found = {}
    def add(sha, source):
        if sha not in eligible:
            return
        if not isinstance(source, str) or len(source.encode()) > 65536 or source_identity(source) != sha:
            raise ValueError('eligible source hash mismatch')
        found[sha] = source
    for row in archive['retained']:
        add(row['id'], row['source'])
    for attempt in portfolio['details']:
        assignments = (attempt.get('native') or {}).get('assignments', [])
        if len(assignments) > 8:
            raise ValueError('original assignment count limit')
        for row in assignments:
            add(row['source_sha256'], row['source'])
    if set(found) != eligible:
        raise ValueError('missing eligible candidate source; subset replay forbidden')
    return [dict(source=found[sha], source_sha256=sha) for sha in sorted(found)]


def check_original(candidates, original):
    chosen = min(candidates, key=lambda c: (len(c['source'].encode()), c['source_sha256']))['source_sha256'] if candidates else None
    if chosen != original:
        raise ValueError('original selection not reproduced from eligible source set')


def known_outcome(sha, scored):
    matches = [(control, row['success']) for control, row in sorted(scored.items())
               if sha is not None and row['candidate_sha256'] == sha and row['complete']]
    values = {value for _, value in matches}
    if len(values) > 1:
        raise ValueError('conflicting completed outcomes for same task and exact source')
    return dict(status='RECORDED' if matches else 'UNSCORED',
                success=matches[0][1] if matches else None, controls=[c for c, _ in matches])


def visible_prompt(record, original):
    selected = record['selections']['equal_budget']
    validate_selection_envelope(selected)
    if selected['selected']['selection'] != 'visible-only/v2' or selected['selected']['source_sha256'] != original:
        raise ValueError('original visible-only/v2 selection required')
    verification = selected['verification']
    prompt = VisiblePrompt(tuple(verification['indices']), tuple(verification['values']))
    if (len(prompt.indices) != 20 or prompt.sha256 != selected['prompt_sha256']
            or verification['source_sha256'] != original or verification['admitted'] is not True):
        raise ValueError('verified visible prompt mismatch')
    return prompt


def public_outcome(row):
    reason = row.get('reason')
    safe = reason is None or isinstance(reason, str) and reason in REASONS
    return dict(success=row['success'] if row['complete'] else None,
                reason=reason if safe else None, reason_omitted=not safe)


def audit_task(record, scored, *, deadline):
    # Only original generation evidence reaches the ranker. No scores, family,
    # category, control outcomes or task IDs are supplied to it.
    portfolio = record['details']['correct']
    candidates = candidate_pool(portfolio)
    selected = record['selections']['equal_budget']
    original = selected['selected']['source_sha256'] if selected else None
    check_original(candidates, original)
    if scored['equal_budget']['candidate_sha256'] != original:
        raise ValueError('scored candidate differs from original selection')
    if candidates:
        prompt = visible_prompt(record, original)
        decision = rank(candidates, prompt, original=original, deadline=deadline)
    else:
        decision = dict(complete=True, selected=None, candidates={})
    # Outcome joining is deliberately subsequent and cannot influence the choice.
    alternative = decision['selected']
    result = dict(original=original, alternative=alternative,
                  changed=original != alternative, observation_complete=decision['complete'],
                  candidates=decision['candidates'], eligible_candidates=len(candidates),
                  original_outcome=known_outcome(original, scored),
                  alternative_outcome=known_outcome(alternative, scored),
                  original_controls={c:public_outcome(v) for c, v in scored.items()})
    return result


def summarize(tasks):
    result = {}
    for category in ('certified', 'ambiguous', 'unresolved'):
        rows = [v for v in tasks.values() if v['category'] == category]
        counts = Counter(tasks=len(rows))
        for row in rows:
            before = row['original_controls']['equal_budget']['success']
            after = row['alternative_outcome']['success']
            counts['original_successes'] += before is True
            counts['original_completed'] += before is not None
            counts['eligible_candidates'] += row['eligible_candidates']
            counts['changed'] += row['changed']
            counts['observation_incomplete'] += not row['observation_complete']
            counts['first_fit_success_lost_by_original'] += row['original_controls']['rescue']['success'] is True and before is False
            counts['changed_known_recoveries'] += row['changed'] and before is False and after is True
            counts['changed_known_regressions'] += row['changed'] and before is True and after is False
            counts['changed_unscored'] += row['changed'] and after is None
            counts['changed_known_same_success'] += row['changed'] and before is True and after is True
            counts['changed_known_same_failure'] += row['changed'] and before is False and after is False
        result[category] = dict(counts)
    return result


def run(report, details, output, *, seconds=600, expected_details_sha256=None):
    if not math.isfinite(seconds) or not 0 < seconds <= 3600:
        raise ValueError('finite audit ceiling in (0,3600] seconds required')
    started = time.monotonic()
    deadline = started + seconds
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('fresh audit output directory required')
    details_hash = file_hash(details)
    if expected_details_sha256 is not None and details_hash != expected_details_sha256:
        raise ValueError('detail ZIP differs from expected original bytes')
    from oeis_learn.pipeline_validation.generalization_archive import verify_archive
    verified = verify_archive(report)
    if verified['integrity'] != 'PASS':
        raise ValueError('original report integrity must pass')
    with zipfile.ZipFile(report) as archive:
        inputs = json_record(archive.read('analysis/research-inputs.json'))
        evidence = json_record(archive.read('evidence.json'))
    records = read_details(details)
    plan, workflow = verify_details(records, inputs, evidence)
    if len(inputs['roster']) != 128:
        raise ValueError('original 128-task population required')
    arms = {}
    for arm in ARMS:
        generation_path, scoring_path, _ = paths(arm)
        generation, scoring = records[generation_path][0], records[scoring_path][0]
        if set(generation) != set(inputs['roster']):
            raise ValueError('generation population mismatch')
        tasks = {}
        for key in sorted(inputs['roster']):
            tasks[key] = dict(audit_task(generation[key], scoring['scored_cases'][key], deadline=deadline),
                              **inputs['roster'][key])
        arms[arm] = dict(tasks=tasks, summary=summarize(tasks))
        print(f'{arm}: audited {len(tasks)} tasks; original choices reproduced', flush=True)
    implementation_files = [Path(__file__), Path(__file__).with_name('selector_coverage.py')]
    import oeis_learn.synthesis.reference as reference
    implementation_files.extend(Path(reference.__file__).parent/name for name in
                                ('reference.py', 'reference_parser.py', 'contracts.py', 'admission.py'))
    result = seal(dict(version='pac-selector-audit/v1', status='EXPLORATORY',
        policy=POLICY, implementation=digest({p.name:file_hash(p) for p in implementation_files}),
        inputs=dict(report=file_hash(report), details=details_hash, plan=plan['identity'],
                    workflow=workflow['identity'], research_inputs=inputs['identity']),
        generation_provenance='operator-supplied detail ZIP and its workflow inventory; generation digests are absent from public report',
        original_verification=verified, arms=arms,
        observation_seconds=time.monotonic()-started, requested_seconds=seconds,
        scoring='existing exact-source outcomes only; unscored alternatives remain unknown',
        scope='correct-input original eligible pools; original controls retained; no new inference or hidden-term execution'))
    output.mkdir(parents=True, exist_ok=False)
    write(output/'audit.json', result)
    lines = ['# Exploratory selector audit', '',
             'Original choices reproduced. No new continuation scoring, training or policy adoption.', '',
             '| Arm | Certified tasks | Changed | Known recoveries | Known regressions | Changed unscored | Observation incomplete |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for arm, value in arms.items():
        s = value['summary']['certified']
        lines.append(f"| {arm} | {s['tasks']} | {s['changed']} | {s['changed_known_recoveries']} | {s['changed_known_regressions']} | {s['changed_unscored']} | {s['observation_incomplete']} |")
    lines.extend(['', 'Known outcomes are exact-source joins, not a new accuracy estimate.',
                  'Missing scores and unavailable observations are not successes or failures.',
                  'This hypothesis was informed by these outcomes; fresh confirmation is required.', ''])
    (output/'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--details', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=600)
    parser.add_argument('--expected-details-sha256', help='Optional externally recorded sha256: digest of the original detail ZIP')
    args = parser.parse_args(argv)
    try:
        result = run(args.report, args.details, args.output, seconds=args.seconds,
                     expected_details_sha256=args.expected_details_sha256)
    except (ValueError, OSError, KeyError, TypeError, RecursionError, zipfile.BadZipFile) as exc:
        # Operator-local diagnostics; no raw input data or exception text exported.
        print(json.dumps(dict(status='FAIL', error=type(exc).__name__,
                              message='Audit rejected input or could not publish; inspect the local traceback.')))
        raise
    print(json.dumps(dict(status=result['status'], identity=result['identity'], output=str(args.output))))
    return 0
