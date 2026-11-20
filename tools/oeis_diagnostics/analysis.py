"""Read-only, separately identified OEIS diagnostics. Never imported by training."""
import hashlib
import json
from pathlib import Path
import time


def digest(value):
    return 'sha256:' + hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def seal(value):
    value = {k: v for k, v in value.items() if k != 'identity'}
    return dict(value, identity=digest(value))


def metrics(source, target):
    """Syntactic indicators, not an assertion of memorisation or a penalty."""
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.synthesis.reference_parser import parse
    program = parse(source)
    nodes = branches = equalities = literal_digits = literal_bits = 0
    literals, singleton_indices = [], []
    def walk(node):
        nonlocal nodes, branches, equalities, literal_digits, literal_bits
        if not isinstance(node, tuple):
            return
        nodes += 1
        op = node[0]
        branches += op in {'if', 'select'}
        equalities += op == 'eq'
        if op in {'int', 'control'}:
            value = int(node[1])
            literals.append(value)
            literal_digits += len(node[1].lstrip('-'))
            literal_bits += abs(value).bit_length() + (value < 0)
        if op == 'eq':
            for index, literal in ((node[1], node[2]), (node[2], node[1])):
                if index == '$index' and isinstance(literal, tuple) and literal[0] == 'int':
                    singleton_indices.append(literal[1])
        for child in node[1:]:
            walk(child)
    for item in (*program.statements, program.result):
        walk(item)
    return dict(concrete_tokens=len(encode_body(source)), target_tokens=len(encode_body(target)),
                ast_nodes=nodes, branches=branches, equalities=equalities,
                literal_count=len(literals), literal_decimal_digits=literal_digits,
                literal_magnitude_bits=literal_bits, largest_literal_bits=max((abs(v).bit_length() for v in literals), default=0),
                direct_index_equalities=len(singleton_indices),
                distinct_direct_index_exceptions=len(set(singleton_indices)),
                heuristic='direct $index == integer tests only; includes legitimate base cases and misses disguised lookups',
                selection_changed=False)


def compare(source, indices, expected, client, *, deadline):
    from oeis_learn.synthesis.reference import evaluate
    if not indices or len(indices) > 100 or len(indices) != len(expected):
        raise ValueError('bounded nonempty exact diagnostic horizon required')
    branches = {}
    def branch(site, decision):
        row = branches.setdefault(str(site), {'true': 0, 'false': 0})
        if decision is not None:
            row['true' if decision else 'false'] += 1
    ref = evaluate(source, indices, deadline=deadline, branch_observer=branch)
    native = client.execute(source, indices, deadline=deadline)
    result = dict(reference_outcome=ref.outcome, native_outcome=native.get('outcome'),
                  reference_reason=ref.reason, native_reason=native.get('reason'),
                  checked=len(indices), first_index=indices[0], last_index=indices[-1],
                  branches=branches, reference_steps=ref.steps)
    if time.monotonic() >= deadline or ref.outcome != 'ok' or native.get('outcome') != 'ok':
        return dict(result, status='INCOMPLETE')
    actual = list(ref.values)
    if actual != native.get('values'):
        return dict(result, status='EXECUTOR_DISAGREEMENT')
    mismatches = [i for i, a, b in zip(indices, actual, expected) if a != b]
    return dict(result, status='MISMATCH' if mismatches else 'MATCH',
                mismatch_count=len(mismatches), first_mismatch=mismatches[0] if mismatches else None)


def analyse(request, client, output):
    """Continue independent cases; persist each outcome before the next case."""
    from oeis_learn.generalization.progressive.archive import Archive
    from oeis_learn.generalization.progressive.admission import validate_member
    from oeis_learn.fresh.common import write
    output = Path(output)
    plan = request['plan']
    if seal(plan) != plan or seal(request) != request or request['plan']['version'] != 'oeis-observer-plan/v1':
        raise ValueError('sealed independent diagnostic request required')
    if (plan.get('seconds') != 600 or plan.get('per_case_seconds') != 8
            or plan.get('visible') != 20 or plan.get('admitted') != 100 or plan.get('diagnostic_total') != 200
            or not isinstance(plan['sample'], list) or not 1 <= len(plan['sample']) <= 32
            or not 1 <= len(request['pool']) <= 512 or len(request['admissions']) > 512):
        raise ValueError('fixed bounded diagnostic scope required')
    members = {m['id']: validate_member(m) for m in request['pool']}
    if len(members) != len(request['pool']):
        raise ValueError('duplicate training member')
    if digest(request['pool']) != plan['pool_identity']:
        raise ValueError('training pool differs from frozen diagnostic plan')
    selected = {r['task']: r for r in plan['sample']}
    if (len(selected) != len(plan['sample']) or not set(selected) <= members.keys()
            or len({r['group'] for r in plan['sample']}) != len(selected)
            or any(r['group'] != members[r['task']]['group'] or r['status'] not in {'READY','UNAVAILABLE'} for r in plan['sample'])):
        raise ValueError('diagnostic roster differs')
    rows, admissions = [], {}
    start = time.monotonic()
    deadline = start + plan['seconds']
    for record in request['admissions']:
        r = Archive.validate_entry(record)
        p = r['payload']
        m = members[p['task']]
        if (p['member'] != digest(m) or p['group'] != m['group'] or p['membership'] != 'training'
                or p['values_hash'] != digest(m['values']) or p['indices'] != m['indices']
                or p['source_snapshot'] != m['source_snapshot'] or p['visible_values'] != m['values'][:20]
                or p['native_verified'] is not True):
            raise ValueError('admission does not belong to original training member')
        if r['identity'] in admissions:
            raise ValueError('duplicate admission in diagnostic request')
        admissions[r['identity']] = r
    if set(admissions) != set(request['accepted_identities']) or len(set(request['accepted_identities'])) != len(request['accepted_identities']):
        raise ValueError('complete original yield admission inventory required')
    # Complexity is measured for every admitted candidate, not just favourable
    # extrapolation cases. Extra truth is restricted to the prospective sample.
    for identity, record in sorted(admissions.items()):
        p = record['payload']
        began = time.monotonic()
        row = dict(admission=identity, task=p['task'], source_hash=p['source_hash'])
        try:
            if time.monotonic() >= deadline:
                raise TimeoutError('diagnostic allowance exhausted')
            row['metrics'] = metrics(p['source'], p['target_source'])
            sample = selected.get(p['task'])
            if sample is None:
                row['continuation'] = dict(status='NOT_SAMPLED')
            elif sample['status'] != 'READY':
                row['continuation'] = dict(status='UNAVAILABLE', reason=sample['reason'])
            else:
                truth = request['extra'][p['task']]
                if digest(truth) != sample['truth_identity']:
                    raise ValueError('extended truth differs from frozen diagnostic plan')
                m = members[p['task']]
                offset = int(m['indices'][0])
                if truth['indices'] != list(map(str, range(offset + 100, offset + 200))) or len(truth['values']) != 100:
                    raise ValueError('exact frozen next-hundred-term range required')
                row['baseline'] = compare(p['source'], m['indices'], m['values'], client,
                    deadline=min(deadline, time.monotonic() + 4))
                # Continue tail checks even after a completed baseline mismatch;
                # all failures remain visible and can never promote admission.
                row['continuation'] = compare(p['source'], truth['indices'], truth['values'], client,
                    deadline=min(deadline, time.monotonic() + 4))
            row['status'] = 'MEASURED'
        except Exception as exc:
            import traceback
            row.update(status='INCOMPLETE', error=type(exc).__name__, reason=str(exc), traceback=traceback.format_exc())
        row['elapsed_seconds'] = time.monotonic() - began
        rows.append(row)
        write(output / 'cases' / (identity[7:] + '.json'), row)
    for task in sorted(selected):
        if not any(r['task'] == task for r in rows):
            rows.append(dict(task=task, status='NO_ADMITTED_PROGRAM',
                             continuation=dict(status='NOT_TESTED'), availability=selected[task]['status']))
    from collections import Counter
    result = seal(dict(version='oeis-observer-report/v1', plan=plan['identity'],
        request=request['identity'], diagnostic_only=True, admission_changed=False,
        training_changed=False, assessment_exposed=False, originals=request.get('originals', []),
        sampled_groups=len(selected), admitted_programs=len(admissions),
        continuation_counts=dict(Counter(r.get('continuation', {}).get('status', 'INCOMPLETE') for r in rows)),
        elapsed_seconds=time.monotonic() - start, rows=rows,
        interpretation='finite training-pool diagnostic only; exceptions are not proof of memorisation; no gate or automatic refutation'))
    write(output / 'report.json', result)
    return result


def regressions(client, output):
    """Exercise actual existing admission/archive code without changing it."""
    from oeis_learn.generalization.progressive.admission import admit, AdmissionRejected
    from oeis_learn.generalization.progressive.archive import Archive
    from oeis_learn.fresh.common import write
    member = dict(id='OEIS:A999999', group='diagnostic-fixture', role='training',
                  source_snapshot=digest('fixture'), indices=list(map(str, range(100))),
                  values=list(map(str, range(100))))
    # Literal return per visible index, no formula-based continuation.
    cases = ' '.join(f'(if (eq $index (int {i})) (then (set $b0 (int {i}))) (else))' for i in range(20))
    lookup = f'(program (local $b0 bigint) {cases} (return (get $b0)))'
    legit = '(program (return (select (eq $index (int 0)) (int 0) $index)))'
    simple = '(program (return $index))'
    outcomes = []
    try:
        admit(member, lookup, client=client)
        raise AssertionError('prefix lookup admitted')
    except AdmissionRejected:
        outcomes.append(dict(case='prefix_lookup_rejected', status='PASS'))
    conditional = admit(member, legit, client=client)
    compact = admit(member, simple, client=client)
    outcomes.append(dict(case='legitimate_conditional_admitted', status='PASS'))
    archive = Archive(Path(output) / 'private/archive')
    cut = archive.publish([conditional, compact])
    if archive.entries(cut)[0]['identity'] != compact['identity']:
        raise AssertionError('archive failed to prefer shortest correct program')
    outcomes.append(dict(case='shortest_verified_program_preferred', status='PASS'))
    result = dict(cases=outcomes, native=client.runtime_identity('bigint-body/v1'), diagnostic_only=True)
    write(Path(output) / 'regressions.json', result)
    return result
