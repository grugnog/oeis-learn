"""Installed native, outcome-blind research supply census; no model execution.

Produces private candidate/support reservoirs and public deficits. A reservoir is
not a frozen learner input or scoreable cohort. No assessment truth is exported.
"""
from collections import Counter, defaultdict
from pathlib import Path
import json
import sqlite3
import time
import traceback

from .identity import seal
from .progressive_workstation_state import write
from .progressive_research_inputs import ExposureIndex, checked_file, _json
from oeis_learn.generalization.progressive.contracts import digest, reference


class Union:
    def __init__(self, count):
        self.parent = list(range(count))
    def root(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i
    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        if a != b:
            self.parent[max(a,b)] = min(a,b)


def features(source):
    """Exact normalized subtree combinations, not a mathematical family claim."""
    from oeis_learn.synthesis.reference_parser import parse
    p = parse(source)
    roots = [*p.statements, p.result]
    nodes, proper, operators = set(), set(), set()
    def normalized(node):
        if isinstance(node, tuple):
            return (node[0], '0') if node[0] in {'int','control'} else tuple(normalized(x) for x in node)
        return node
    def visit(node, depth):
        if not isinstance(node, tuple):
            return
        operators.add(node[0]); nodes.add(digest(normalized(node)))
        if depth and node[0] not in {'int','control'}:
            proper.add(digest(normalized(node)))
        for child in node[1:]:
            visit(child, depth+1)
    for root in roots:
        visit(root, 0)
    return dict(nodes=nodes, proper=proper, operators=operators)


def classify(row, support):
    if row['structure'] in support['structures']:
        return 'parameter'
    f = features(row['source'])
    if f['proper'] <= support['nodes'] and f['operators'] <= support['operators']:
        return 'composition'
    return 'unseen_structure'


def reserve_panels(rows, *, seed, required, minimum_groups, panels, deadline=None):
    """Assign whole detected components before quotas; never substitute/reseed."""
    linked, seen = Union(len(rows)), {}
    for i, row in enumerate(rows):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('panel grouping exhausted census finalization allowance')
        keys = [('window', digest(row['values'][j:j+20])) for j in range(81)]
        keys += [('behavior', digest(row['values'])), ('id', row['id'])]
        if row.get('structure'):
            keys.append(('structure', row['structure']))
        keys += [(k, row[k]) for k in ('group','family') if row.get(k)]
        for key in keys:
            if key in seen:
                linked.join(i, seen[key])
            else:
                seen[key] = i
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[linked.root(i)].append(row)
    assigned = defaultdict(lambda: defaultdict(list))
    # The retention panel is independent and uses parameter-transfer candidates.
    owners = [*panels, 'retention']
    for group in components.values():
        identity = digest(sorted(r['id'] for r in group))
        owner = owners[int(digest([seed, identity])[7:23], 16) % len(owners)]
        for row in group:
            if owner == 'retention' and row['stratum'] != 'parameter':
                continue
            assigned[(owner, row['stratum'])][identity].append(row)
    roster, counts, deficits = {}, {}, []
    for panel in owners:
        roster[panel] = []
        strata = ['parameter'] if panel == 'retention' else ['parameter','composition','unseen_structure','oeis']
        for stratum in strata:
            chosen = []
            groups = assigned[(panel, stratum)]
            for group in sorted(groups, key=lambda g: digest([seed, panel, stratum, g])):
                for row in sorted(groups[group], key=lambda r: digest([seed, r['id']]))[:8]:
                    if len(chosen) < required:
                        chosen.append(dict(row, group=group, role=panel))
            n = len({r['group'] for r in chosen})
            counts[panel+'/'+stratum] = dict(tasks=len(chosen), groups=n,
                available_groups=len(groups), available_rows=sum(map(len,groups.values())),
                task_deficit=max(0, required-len(chosen)),
                group_deficit=max(0, minimum_groups-n))
            if len(chosen) < required or n < minimum_groups:
                deficits.append('panel_supply:'+panel+'/'+stratum)
            roster[panel].extend(chosen)
    return roster, counts, deficits


def validate_request(request):
    from .progressive_research_bridge import request as expected
    required = set(expected(seal(dict(version='progressive-registered-data/v1',
        candidates={}, exposure={}, recipe={}))))
    if not isinstance(request, dict) or set(request) != required or seal(request) != request:
        raise ValueError('closed sealed census request required')
    if request['version'] != 'progressive-research-census-request/v1':
        raise ValueError('unknown census request')
    if type(request['diagnostic']) is not bool:
        raise ValueError('explicit census purpose required')
    if not request['diagnostic']:
        dummy = seal(dict(version='progressive-registered-data/v1',
                         **{k: request[k] for k in ('candidates','exposure','recipe')}))
        template = expected(dummy)
        if any(request[k] != template[k] for k in template if k not in {'identity','registration'}):
            raise ValueError('prospective research census limits changed')
    for field, cap in [('training_draws',8192),('evaluation_draws',16384),('seconds',21600),
                       ('shard_rows',512),('required_per_stratum',128),('minimum_groups',16)]:
        if type(request[field]) is not int or not 1 <= request[field] <= cap:
            raise ValueError('bounded census field: '+field)
    if (request['panels'] != ['development','confirmation-diversity','assessment']
            or request['per_case_seconds'] != 4 or request['max_shards'] != 64
            or request['disk_bytes'] != 8 << 30 or request['memory_bytes'] != 64 << 30
            or request['token_target'] != 25000000 or request['proposed_refresh_updates'] != 8
            or type(request['seed']) is not int or not 0 <= request['seed'] < 2**32):
        raise ValueError('fixed census scope required')
    return request


def run(root, output, request, *, client):
    from oeis_learn.generalization.progressive.generator import default_policy, draw
    from .progressive_preparation import verify_draw
    request = validate_request(request)
    root, output = Path(root), Path(output)
    private = output / 'private'; private.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = private / 'census.sqlite'
    if path.exists():
        raise ValueError('prior census retained; no automatic repetition or budget reset')
    candidates = _json(checked_file(root, request['candidates']), maximum=128 << 20)
    recipe = _json(checked_file(root, request['recipe']))
    batch=recipe['learning']['batch_size']
    if type(batch) is not int or batch < 1:
        raise ValueError('positive original effective batch required')
    exposure = checked_file(root, request['exposure'])
    if not isinstance(candidates,list) or len(candidates)>4096 or len({r['id'] for r in candidates}) != len(candidates):
        raise ValueError('bounded unique original OEIS projection required')
    began = time.monotonic(); deadline = began + request['seconds']
    work_deadline = deadline-min(600,max(1,request['seconds']//10))
    counters, failures, evaluation, training = Counter(), [], [], []
    support = dict(structures=set(), nodes=set(), operators=set())
    law = default_policy(seed=request['seed'])
    total = request['training_draws'] + request['evaluation_draws']
    law.update(max_draws=total, max_work_units=total)
    db = sqlite3.connect(path)
    db.execute('PRAGMA journal_mode=DELETE')
    page_size=db.execute('PRAGMA page_size').fetchone()[0]
    # Two 1-GiB SQLite stores plus their rollback journals, <=512 MiB of
    # reservoir JSON (including atomic-write headroom), and a 512-MiB native
    # cache fit within the declared 8-GiB output reservation.
    db.execute('PRAGMA max_page_count='+str((request['disk_bytes']//8)//page_size))
    db.execute('CREATE TABLE draws(ordinal INTEGER PRIMARY KEY,status TEXT,row TEXT,reason TEXT)')
    db.execute('CREATE TABLE binding(value TEXT)')
    db.execute('INSERT INTO binding VALUES (?)', (json.dumps(request),)); db.commit()
    def progress():
        write(output/'census-progress.json',seal(dict(version='progressive-census-progress/v1',
            requested=total, outcomes=dict(counters), elapsed_seconds=time.monotonic()-began,
            failures=failures[:32], private_outputs_retained=True)))
    def classify_fresh(row, historical, training_index):
        stratum = classify(row, support)
        # Structure sharing alone is allowed only for parameter transfer.
        forbidden = set(historical.matches(row)) | set(training_index.matches(row))
        if stratum == 'parameter':
            forbidden.discard('structure')
        if forbidden:
            return None, 'exposed:'+','.join(sorted(forbidden))
        return seal(dict(row, stratum=stratum, membership='evaluation_candidate')), None
    from types import SimpleNamespace
    try:
        with ExposureIndex(exposure, readonly=True) as history, ExposureIndex(private/'training-exposure.sqlite') as train_index:
            index_page_size=train_index.db.execute('PRAGMA page_size').fetchone()[0]
            train_index.db.execute('PRAGMA max_page_count='+str((request['disk_bytes']//8)//index_page_size))
            for ordinal in range(total):
                if time.monotonic() >= work_deadline:
                    counters['unattempted'] += total-ordinal
                    break
                with db:
                    db.execute('INSERT INTO draws VALUES (?,?,?,?)',(ordinal,'RESERVED',None,None))
                row, reason, status = None, None, 'REJECTED'
                try:
                    row, reason = verify_draw(draw(law, ordinal), client=client,
                        budget=SimpleNamespace(deadline=min(work_deadline,time.monotonic()+request['per_case_seconds'])))
                    if row is not None:
                        row = seal(dict(row, id='mechanical:'+row['draw_identity']))
                        if ordinal < request['training_draws']:
                            f=features(row['source'])
                            with train_index.db:
                                train_index.add(row, origin='prospective-support', role='training')
                            training.append(row)
                            support['structures'].add(row['structure'])
                            support['nodes'].update(f['nodes']); support['operators'].update(f['operators'])
                            status='TRAINING'
                        else:
                            row, reason=classify_fresh(row,history,train_index)
                            if row is not None:
                                evaluation.append(row); status='EVALUATION'
                except Exception as exc:
                    status='FAILED'; reason=type(exc).__name__+': '+str(exc)
                    if len(failures)<32:
                        failures.append(dict(ordinal=ordinal,reason=reason,traceback=traceback.format_exc()))
                with db:
                    db.execute('UPDATE draws SET status=?,row=?,reason=? WHERE ordinal=?',
                               (status,json.dumps(row) if row else None,reason,ordinal))
                counters[status]+=1
                if ordinal % 64 == 0:
                    progress(); print('Research native census: '+str(ordinal+1)+'/'+str(total),flush=True)
            # No generation/model is involved in the independent source census.
            for ordinal, original in enumerate(candidates):
                if time.monotonic() >= work_deadline:
                    counters['oeis_unattempted'] += len(candidates)-ordinal
                    break
                try:
                    hits=set(history.matches(original)) | set(train_index.matches(original))
                    if hits:
                        counters['oeis_exposed']+=1
                    else:
                        evaluation.append(dict(original,stratum='oeis')); counters['oeis_available']+=1
                except Exception as exc:
                    counters['oeis_failed']+=1
                    if len(failures)<32:
                        failures.append(dict(source_ordinal=ordinal,reason=type(exc).__name__+': '+str(exc),
                                             traceback=traceback.format_exc()))
        roster, panels, deficits=reserve_panels(evaluation,seed=request['seed'],
            required=request['required_per_stratum'],minimum_groups=request['minimum_groups'],panels=request['panels'],deadline=deadline)
        # Reservoirs remain private and are not scoreable cohorts/learner channels.
        for value in (training,roster):
            if len(json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()) > 128 << 20:
                raise ValueError('private reservoir exceeds original 128-MiB publication bound')
        write(private/'training-reservoir.json',training)
        write(private/'panel-reservations.json',roster)
        minimum=min((len(r['tokens']) for r in training), default=0)
        shards=min(request['max_shards'],len(training)//request['shard_rows'])
        updates=(request['token_target']+minimum*batch-1)//(minimum*batch) if minimum else None
        required=(updates+request['proposed_refresh_updates']-1)//request['proposed_refresh_updates'] if updates else None
        bins=Counter('frontier' if {'loop','select'} & set(r['operations']) else 'familiar' for r in training)
        if not shards or any(bins[k]<2 for k in ('frontier','familiar')):
            deficits.append('native_training_support_or_probe_supply')
        if required is None or required>shards:
            deficits.append('refresh_cadence_requires_prospective_revision')
        if counters['FAILED'] or counters['unattempted'] or counters['oeis_unattempted'] or counters['oeis_failed']:
            deficits.append('incomplete_native_or_source_census')
        result=seal(dict(version='progressive-research-census/v1',request=request['identity'],
            status='INCOMPLETE' if deficits else 'COMPLETE', diagnostic=request['diagnostic'],
            research_ready=False, qualification=False, training_started=False, assessment_exposed=False,
            native=client.runtime_identity('bigint-body/v1'), elapsed_seconds=time.monotonic()-began,
            counts=dict(counters), panels=panels, deficits=deficits, failures=failures,
            reserved_panel_rows=sum(len(rows) for rows in roster.values()),
            unreserved_evaluation_rows=len(evaluation)-sum(len(rows) for rows in roster.values()),
            support=dict(rows=len(training),bins=dict(bins),canonical_structures=len(support['structures']),
                minimum_target_tokens=minimum,effective_batch=batch,maximum_updates=updates,
                available_full_shards=shards,proposed_refresh_updates=request['proposed_refresh_updates'],
                required_shards=required,minimum_feasible_refresh_updates=(updates+shards-1)//shards if shards and updates else None,
                scope='conservative supply estimate; no schedule revision applied'),
            artifacts=[reference(output,private/name) for name in
                       ('training-reservoir.json','panel-reservations.json','census.sqlite','training-exposure.sqlite')],
            detector_limits='exact finite 20-term windows, normalized structures and source identities; not semantic equivalence',
            strata_rule='parameter: supported normalized structure; composition: new structure with supported proper subtrees/operators; unseen: remaining structures',
            disposition='private reservations only; independent proof/census/publication and exact-input calibration required before freeze'))
        write(output/'census.json',result)
        return result
    finally:
        db.close(); progress()


def main():
    from oeis_learn.synthesis.client import SynthesisClient
    root=Path('/work/inputs'); output=Path('/work/output')
    request=_json(output/'request.json')
    with SynthesisClient(digest(['research-census',request['identity']]),cache_root=output/'private/native') as client:
        run(root,output,request,client=client)


if __name__=='__main__':
    main()
