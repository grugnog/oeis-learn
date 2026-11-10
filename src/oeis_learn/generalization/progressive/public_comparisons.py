"""Public comparison provenance checks, separate from the scientific reducer."""
import json
from .contracts import digest, seal


def verify_authority(comparison, authorities, campaign):
    """Return missing evidence explicitly; reject altered complete evidence."""
    binding = comparison.get('binding')
    if binding and 'cycle' in binding:
        return verify_cycle(comparison, authorities, campaign)
    if binding is None:
        return ['comparison_binding:' + comparison['id']]
    if set(binding) - {'chain','authorization','bundle'} != {'registration', 'source_slot', 'target_slot', 'producers'}:
        raise ValueError('closed public comparison binding required')
    registrations = [a for a in authorities if a.get('identity') == binding['registration']
                     and a.get('version') == 'progressive-gate-registration/v1']
    if len(registrations) != 1:
        raise ValueError('original frozen comparison registration required')
    registration = registrations[0]
    if set(registration) != {'version','identity','source_slot','target_slot','panels','seeds','design','producers'}:
        raise ValueError('closed original gate registration required')
    if (seal(registration) != registration or registration['source_slot'] != binding['source_slot']
            or registration['target_slot'] != binding['target_slot']
            or registration['design'] != digest(comparison['design'])
            or registration['seeds'] != comparison['seeds']):
        raise ValueError('comparison registration differs')
    panels = {a.get('identity'):a for a in authorities if a.get('version') == 'discovery-cohort/v1'}
    roster = []
    for key in registration['panels']:
        if key not in panels:
            return ['comparison_panel:' + key]
        panel = panels[key]
        if panel['purpose'] != campaign['purpose'] or panel['track'] != campaign['track'] or panel['payload'].get('role') not in {'confirmation','development'}:
            raise ValueError('comparison panel purpose/track differs')
        roster.extend(dict(task=r['id'], group=sorted(r['groups'])[0], stratum=r['stratum'])
                      for r in panel['payload']['rows'])
    canonical = lambda rows: sorted(rows, key=lambda r:(r.get('seed',-1),r.get('arm',''),r['task']))
    if not roster or canonical(roster) != canonical(comparison['roster']):
        raise ValueError('comparison must retain exact original panel roster')
    members = {r['task']:r for r in roster}
    cells = {(arm,seed) for arm in ('baseline','candidate') for seed in registration['seeds']}
    found, derived = set(), []
    if len(binding['producers']) != len(registration['producers']):
        return ['comparison_producers:' + comparison['id']]
    state, tip = {}, campaign['identity']
    chain = binding.get('chain', [])
    if len(chain) > 4096:
        raise ValueError('bounded comparison ownership chain required')
    for transition in chain:
        if seal(transition) != transition or transition['previous'] != tip:
            raise ValueError('comparison ownership chain differs')
        state, tip = transition['state'], transition['identity']
    for declared, producer in zip(registration['producers'], binding['producers']):
        registered = resolve_public_producer(declared, producer, campaign, state)
        cell = registered['arm'], registered['seed']
        if cell not in cells or cell in found:
            raise ValueError('paired comparison producer differs')
        found.add(cell)
        if set(producer)-{'resolved','ownership'} != {'candidates','scores'}:
            raise ValueError('closed original public producers required')
        values = {}
        for kind in ('candidates','scores'):
            raw = producer[kind].encode()
            ref = registered[kind]
            import hashlib
            if len(raw) != ref['bytes'] or 'sha256:'+hashlib.sha256(raw).hexdigest() != ref['sha256']:
                raise ValueError('comparison original producer bytes differ')
            from .verify import _json
            values[kind] = _json(raw)
        generated, scored = values['candidates'], values['scores']
        if (seal(generated) != generated or seal(scored) != scored
                or generated.get('checkpoint') != registered['checkpoint']['sha256']
                or (registered['model'] is not None and generated.get('model') != registered['model'])
                or set(generated.get('records',{})) != set(members)
                or scored.get('candidates') != registered['candidates']['sha256']
                or scored.get('denominator') != len(roster)):
            raise ValueError('comparison checkpoint/generator/scorer differs')
        for task, outcome in scored['outcomes'].items():
            if task not in members:
                raise ValueError('comparison foreign outcome')
            primary = outcome['equal_budget']
            if type(primary.get('success')) is not bool or type(primary.get('complete')) is not bool:
                raise ValueError('exact scored completion flags required')
            status = 'success' if primary['success'] else 'failure' if primary['complete'] else 'unresolved'
            if primary['outcome'] != status:
                raise ValueError('comparison outcome/completion differs')
            derived.append(dict(members[task], seed=cell[1], arm=cell[0], outcome=status))
    supplied = [{k:r[k] for k in ('task','group','stratum','seed','arm','outcome')} for r in comparison['rows']]
    if found != cells or canonical(derived) != canonical(supplied):
        raise ValueError('comparison rows differ from original producers')
    action = comparison['action']
    if action == 'EXPLORE_RESERVED':
        return [] # independently replayed after scientific reduction
    if action == 'ADVANCE_REGISTERED' and not any(
        e['source']==binding['source_slot'] and e['target']==binding['target_slot']
        and e['condition']==comparison['verdict'] and e['action']==action
        for e in campaign['payload']['graph']['edges']):
        raise ValueError('unreserved comparison advancement')
    if action not in {'ADVANCE_REGISTERED','RETAIN_BASELINE','DEFER_BRANCH','STOP_AFFECTED'}:
        raise ValueError('unsupported comparison action')
    return []


def cost_tasks(campaign, source_slot, seeds):
    nodes = {n['id']:n for n in campaign['payload']['graph']['nodes']}
    selected, pending = set(), [source_slot]
    while pending:
        slot = pending.pop()
        if slot not in nodes:
            raise ValueError('unknown comparison cost scope')
        if slot not in selected:
            selected.add(slot)
            pending.extend(nodes[slot]['dependencies'])
    baseline = {'static', 'continued_synthetic', 'no_library', 'without_source'}
    result = {'baseline':[], 'candidate':[]}
    for task in campaign['payload']['tasks']:
        if task['slot'] in selected and task['seed'] in seeds:
            arm = 'baseline' if task['arm'] in baseline else 'candidate'
            result[arm].append(digest(task['id']))
    return {k:sorted(v) for k,v in result.items()}


def measured_costs(comparison, campaign, accounting):
    if comparison['design']['claim'] != 'efficiency':
        return None
    binding, costs = comparison.get('binding'), comparison.get('costs')
    if not binding or costs is None:
        return None
    expected = cost_tasks(campaign, binding['source_slot'], comparison['seeds'])
    if costs != expected or any(not v for v in costs.values()):
        raise ValueError('comparison cost task scope differs')
    original = {r['task']:r for r in accounting}
    if any(t not in original for tasks in costs.values() for t in tasks):
        return None
    measured = {arm:sum(original[t]['elapsed_us'] for t in tasks) for arm,tasks in costs.items()}
    return measured if all(v > 0 for v in measured.values()) else None


def export_binding(root, campaign, registration, store):
    from pathlib import Path
    from .contracts import verify_refs, load
    from oeis_learn.pipeline_validation.progressive_study import original, weight_ref
    def read(ref):
        return verify_refs(root, [ref], max_bytes=4<<20)[0].decode()
    producers = []
    for declared in registration['producers']:
        resolved, ownership = dict(declared), {}
        for kind in ('checkpoint','candidates','scores'):
            slot = declared[kind]
            if 'operation' not in slot:
                continue
            task_id = slot['task']
            if store is None:
                raise ValueError('committed owned comparison output required')
            plan = store.resolved_plan(next(p for p in campaign['payload']['tasks'] if p['id']==task_id))
            cut = store.state['outputs'][task_id]
            proof = dict(plan=plan, cut=cut, result=read(cut['result']))
            if kind == 'checkpoint':
                ref = original(root, store, task_id, 'train', 'checkpoint_record')
                checkpoint = json.loads(read(ref))
                base = Path(root)/'run-state/discovery-learning'/campaign['identity'][7:]/checkpoint['payload']['task'][7:]/'checkpoints'
                full = verify_refs(base,[checkpoint['payload']['full_state']],max_bytes=4<<20)[0].decode()
                proof.update(checkpoint=read(ref), full=full)
                resolved[kind] = weight_ref(root, store, task_id)
            else:
                resolved[kind] = original(root,store,task_id,slot['operation'],slot['output'])
            ownership[kind] = proof
        producers.append(dict(candidates=read(resolved['candidates']),scores=read(resolved['scores']),
                              resolved=resolved, ownership=ownership))
    chain = [load(p) for p in sorted(store.root.glob('transition-*.json'))] if store else []
    return dict(registration=registration['identity'],source_slot=registration['source_slot'],
                target_slot=registration['target_slot'],producers=producers,chain=chain)


def resolve_public_producer(declared, producer, campaign, state):
    from .contracts import record
    from .trial_scopes import resolve
    resolved = producer.get('resolved', declared)
    ownership = producer.get('ownership', {})
    for kind in ('checkpoint','candidates','scores'):
        slot = declared[kind]
        if 'operation' not in slot:
            if resolved[kind] != slot:
                raise ValueError('static comparison producer changed')
            continue
        proof = ownership.get(kind)
        if proof is None:
            raise ValueError('missing owned comparison producer')
        plan, cut = proof['plan'], proof['cut']
        original = next((p for p in campaign['payload']['tasks'] if p['id']==slot['task']),None)
        if original is None:
            raise ValueError('foreign comparison producer task')
        if 'trial_choice' in plan:
            original = resolve(original, plan['trial_choice'])
        if original != plan or plan['seed'] != declared['seed']:
            raise ValueError('frozen comparison producer task differs')
        task = record('task',dict(plan,campaign=campaign['identity']),created_at=campaign['created_at'],
                      purpose=campaign['purpose'],track=campaign['track'])
        if (cut['task'] != task['identity'] or cut['operation'] != slot['operation']
            or state.get('outputs',{}).get(slot['task']) != cut
            or state.get('tasks',{}).get(slot['task']) != 'COMPLETE'):
            raise ValueError('uncommitted comparison output')
        result = checked_json(proof['result'],cut['result'])
        if seal(result) != result or result['operation'] != slot['operation']:
            raise ValueError('original comparison result differs')
        if kind == 'checkpoint':
            checkpoint = checked_json(proof['checkpoint'],result['metrics']['checkpoint_record'])
            full = checked_json(proof['full'],checkpoint['payload']['full_state'])
            if (checkpoint['payload']['task'] != task['identity']
                or full['states']['weights']['sha256'] != resolved[kind]['sha256']):
                raise ValueError('comparison checkpoint lineage differs')
        elif any(result['metrics'][slot['output']][k] != resolved[kind][k] for k in ('sha256','bytes')):
            raise ValueError('owned comparison artifact differs')
    if any(resolved[k] != declared[k] for k in ('arm','seed','model')):
        raise ValueError('comparison cell/model differs')
    return resolved


def checked_json(raw, ref):
    import hashlib
    encoded = raw.encode()
    if len(encoded) != ref['bytes'] or 'sha256:'+hashlib.sha256(encoded).hexdigest()!=ref['sha256']:
        raise ValueError('original comparison bytes differ')
    from .verify import _json
    return _json(encoded)


def verify_exploration(comparison, authorities, campaign, metrics):
    if comparison['action'] != 'EXPLORE_RESERVED':
        return []
    binding = comparison.get('binding') or {}
    audit, bundle = binding.get('authorization'), binding.get('bundle')
    if not audit or not bundle:
        return ['comparison_exploratory_authority:'+comparison['id']]
    if (seal(audit)!=audit or seal(bundle)!=bundle or audit['bundle']!=bundle['identity']
        or bundle['decision']['identity']!=comparison['id']
        or any(bundle[k]!=comparison[k] for k in ('rows','roster','seeds','design'))):
        raise ValueError('original exploratory decision differs')
    policies = [a for a in authorities if a.get('identity')==campaign['payload']['target_policy']]
    if len(policies)!=1:
        return ['exploratory_policy:'+comparison['id']]
    if (audit['kwargs']!=bundle['kwargs'] or bundle['source_slot']!=binding['source_slot']
        or bundle['target_slot']!=binding['target_slot']
        or bundle['decision']['payload']['next_action']!=comparison['action']):
        raise ValueError('exploratory action arguments differ')
    previous=[];matched=False
    for transition in binding.get('chain',[]):
        decisions=list(transition['state'].get('decisions',{}).values())
        if any(d.get('bundle')==bundle['identity'] for d in decisions):
            if previous!=audit['prior_decisions']:
                raise ValueError('exploratory prior decisions differ')
            matched=True;break
        previous=decisions
    if not matched:return ['original_exploratory_transition:'+comparison['id']]
    policy=policies[0]['payload']
    target=next((t for t in policy['targets'] if t['id']==audit['kwargs']['target_id']),None)
    branch=next((b for b in policy['diagnostics'] if target and b['id']==target['budget_ref']),None)
    if not target or not policy['selected'] or 'EXPLORE_RESERVED' not in target['allowed_actions'] or not branch or not branch.get('trigger'):
        raise ValueError('unselected exploratory information goal')
    if (binding['target_slot'] not in branch['task_slots'] or branch['maximum_visits']!=1
        or any(d.get('failure_class')==branch['failure_class'] or d.get('panel')==branch['panel'] for d in audit['prior_decisions'])):
        raise ValueError('repeated or unreserved exploratory task/panel')
    if not any(e['source']==binding['source_slot'] and e['target']==binding['target_slot']
               and e['condition']==comparison['verdict'] and e['action']=='EXPLORE_RESERVED'
               for e in campaign['payload']['graph']['edges']):
        raise ValueError('unreserved exploratory edge')
    panels=[a for a in authorities if a.get('identity')==branch['panel'] or branch['panel'] in a.get('_reference_names',[])]
    if len(panels)!=1 or panels[0]['payload']['role'] in {'training','assessment'}:
        raise ValueError('original unused exploratory panel required')
    signal=branch['signal']
    effects=list(metrics.get('seed_effects',{}).values())
    effects=[v for v in effects if v is not None]
    observed=False
    if signal=='seed_disagreement':
        observed=len(effects)>=2 and min(effects)<=0<max(effects)
        designs=[a for a in authorities if a.get('identity')==branch['design'] or digest(a)==branch['design']]
        if not any(a.get('original_design')==digest(comparison['design']) and a.get('original_seeds')==comparison['seeds'] and a.get('branch')==branch['id'] for a in designs):
            raise ValueError('original third-seed design required')
    elif branch['design']!=digest(comparison['design']):
        raise ValueError('exploratory design differs')
    elif signal=='support_deficit':
        observed=metrics.get('groups',0)<16
    elif signal=='heldout_yield':
        observed=metrics.get('mean_gain') is not None and (metrics['mean_gain']<0.02 or any(v<=0 for v in effects) or metrics.get('lower') is None or metrics['lower']<=0)
    elif signal=='runtime_bottleneck':
        runtime=audit['kwargs'].get('runtime',{})
        rule=branch.get('runtime_rule',{})
        originals=[a for a in authorities if a.get('identity')==rule.get('producer',{}).get('sha256') or a.get('measurements')==runtime.get('measurements')]
        observed=bool(originals and runtime.get('producer')==rule.get('producer') and runtime.get('scope')=='observed execution; training loss excluded' and type(runtime.get('measurements',{}).get(rule.get('metric'))) is int and runtime['measurements'][rule['metric']]>rule['ceiling_us'])
    if not observed:
        raise ValueError('exploratory trigger unsupported by independent evidence')
    # Replay the original debit prefix; pending work remains spent and only
    # observed elapsed time can be refunded by a matching completed reservation.
    from .budgets import vector, KEYS
    events=audit['budget_events']
    if len(events)>65536:
        raise ValueError('bounded exploratory budget prefix required')
    tip=digest(dict(campaign=campaign['identity'],caps=campaign['payload']['caps']))
    reservations, finished, charges = {}, set(), {}
    for index,e in enumerate(events):
        if seal(e)!=e or e['previous']!=tip or e['index']!=index:
            raise ValueError('exploratory budget chain differs')
        vector(e['cost']);op=e['operation'];tip=e['identity']
        if e['event']=='RESERVE':
            if op in reservations:raise ValueError('duplicate exploratory reservation')
            reservations[op]=e;charges[op]=dict(e['cost'])
        elif e['event']=='COMMIT' and op in reservations and op not in finished:
            original=reservations[op]
            if e['boot']!=original['boot'] or e['start_us']!=original['start_us']:
                raise ValueError('exploratory supervisor clock differs')
            if e['end_us'] is None:
                if e['cost']['elapsed_us']!=original['cost']['elapsed_us']:
                    raise ValueError('unproven exploratory refund')
            elif type(e['end_us']) is not int or e['end_us']<e['start_us'] or e['end_us']-e['start_us']!=e['cost']['elapsed_us']:
                raise ValueError('exploratory monotonic proof differs')
            if any(e['cost'][k]!=original['cost'][k] for k in KEYS if k!='elapsed_us') or e['cost']['elapsed_us']>original['cost']['elapsed_us']:
                raise ValueError('exploratory debit refund differs')
            charges[op]['elapsed_us']=e['cost']['elapsed_us'];finished.add(op)
        else:raise ValueError('invalid exploratory budget event')
    spent={}
    for task in campaign['payload']['tasks']:
        used={k:sum(v[k] for op,v in charges.items() if op.startswith(task['id']+'/')) for k in KEYS}
        if any(used[k]>task['caps'][k] for k in KEYS):raise ValueError('exploratory task allowance exceeded')
        spent[task['id']]=used
    nodes={n['id']:n for n in campaign['payload']['graph']['nodes']}
    chosen=[t for t in campaign['payload']['tasks'] if t['slot'] in branch['task_slots']]
    for key,cap in branch['reservation'].items():
        branch_spent=sum(spent[t['id']][key] for t in chosen)
        available=sum(t['caps'][key]-spent[t['id']][key] for t in chosen)
        pool_spent=sum(spent[t['id']][key] for t in campaign['payload']['tasks'] if nodes[t['slot']]['kind']=='diagnostic')
        if min(cap-branch_spent,available,campaign['payload']['graph']['diagnostic_pool'][key]-pool_spent)<0 or (key=='elapsed_us' and min(cap-branch_spent,available,campaign['payload']['graph']['diagnostic_pool'][key]-pool_spent)<=0):
            raise ValueError('exploratory original reservation exhausted')
    return []


def verify_cycle(comparison, authorities, campaign):
    binding=comparison['binding']
    plans=[a for a in authorities if a.get('identity')==binding['cycle'] and a.get('version')=='progressive-oeis-cycle/v1']
    if len(plans)!=1:return ['original_cycle_plan:'+comparison['id']]
    plan=plans[0];label=binding['baseline']
    if label not in {'starting','continued_synthetic'}:
        raise ValueError('both original OEIS baselines must remain distinct')
    expected_design=dict(claim='capability',bootstrap_seed=plan['design']['bootstrap_seed'],comparisons=plan['design']['comparisons'],links=[],costs=None,primary=['oeis'])
    if comparison['design']!=expected_design or comparison['action']!='RETAIN_BASELINE':
        raise ValueError('cycle design/action differs from frozen authority')
    panels={}
    for kind in ('transfer','retention'):
        ref=plan['panels'][kind]
        matches=[a for a in authorities if digest(a)==ref['sha256']]
        # Artifact hashes identify serialized bytes, while envelope identities
        # identify the sealed value; use the included original authority map.
        matches=[a for a in authorities if a.get('_artifact_sha256')==ref['sha256']] or matches
        if len(matches)!=1:return ['original_cycle_panel:'+kind]
        panels[kind]=matches[0]
    roster=[dict(task=r['id'],group=r['groups'][0],stratum='retention' if k=='retention' else 'oeis')
            for k,p in panels.items() for r in p['payload']['rows'] if k=='retention' or r['stratum']=='oeis']
    canonical=lambda rows:sorted(rows,key=lambda r:(r.get('seed',-1),r.get('arm',''),r['task']))
    if canonical(roster)!=canonical(comparison['roster']):
        raise ValueError('cycle comparison original roster differs')
    tasks={t['id']:t for t in campaign['payload']['tasks']}
    expected_seeds=sorted({tasks[e['task']]['seed'] for e in plan['training']})
    if expected_seeds!=comparison['seeds']:raise ValueError('cycle seed roster differs')
    derived=[];seen=set()
    for producer in binding['producers']:
        descriptor=producer['descriptor'];stage=descriptor['stage']
        if stage not in {label,'replay'}:raise ValueError('foreign cycle comparison stage')
        task=tasks.get(descriptor['task'])
        expected_arm='candidate' if stage=='replay' else 'baseline'
        if (task is None or descriptor['seed']!=task['seed'] or descriptor['arm']!=expected_arm
            or task['arm']!=('replay' if stage in {'starting','replay'} else 'continued_synthetic')):
            raise ValueError('cycle checkpoint arm/seed differs')
        kind=next((k for k in panels if plan['panels'][k]==descriptor['panel']),None)
        if kind is None:raise ValueError('foreign cycle panel')
        key=(stage,descriptor['seed'],kind)
        if key in seen:raise ValueError('duplicate cycle producer')
        seen.add(key)
        generated=checked_json(producer['candidates'],descriptor['candidates'])
        scored=checked_json(producer['scores'],descriptor['scores'])
        members={r['task']:r for r in roster if (r['stratum']=='retention')==(kind=='retention')}
        if (seal(generated)!=generated or seal(scored)!=scored or (not generated.get('checkpoint') and not (stage=='starting' and generated.get('model_control')=='random' and task['initializer']['mode']=='random'))
            or set(generated['records'])!=set(members) or scored['candidates']!=descriptor['candidates']['sha256']
            or scored['denominator']!=len(members)):
            raise ValueError('cycle original checkpoint/scorer binding differs')
        for task_id,outcome in scored['outcomes'].items():
            if task_id not in members:raise ValueError('foreign cycle outcome')
            primary=outcome['equal_budget']
            if type(primary.get('success')) is not bool or type(primary.get('complete')) is not bool:
                raise ValueError('exact cycle completion flags required')
            status='success' if primary['success'] else 'failure' if primary['complete'] else 'unresolved'
            if primary['outcome']!=status:raise ValueError('cycle outcome differs')
            derived.append(dict(members[task_id],seed=descriptor['seed'],arm=expected_arm,outcome=status))
    expected={(stage,seed,kind) for stage in (label,'replay') for seed in expected_seeds for kind in panels}
    if seen!=expected:return ['cycle_paired_producers:'+comparison['id']]
    supplied=[{k:r[k] for k in ('task','group','stratum','seed','arm','outcome')} for r in comparison['rows']]
    if canonical(derived)!=canonical(supplied):raise ValueError('cycle rows differ from original scores')
    return []



def partition(comparisons, observations, *, maximum_bytes):
    """Split original comparison bytes without losing a source or denominator."""
    from .report import encoded, MAX_PART
    result=[]
    for comparison in comparisons:
        raw=encoded(comparison).decode()
        if len(raw.encode())<=MAX_PART:
            result.append(comparison);continue
        if len(raw.encode())*2>maximum_bytes:
            result.append(dict(id=comparison['id'],omitted='original comparison exceeds frozen package reserve'))
            continue
        parts=[]
        for index,start in enumerate(range(0,len(raw),262144)):
            chunk=raw[start:start+262144]
            identity=digest(['comparison-part',comparison['id'],index,chunk])
            observations.append(dict(producer=identity,data=dict(comparison_part=comparison['id'],ordinal=index,raw=chunk)))
            parts.append(identity)
        result.append(dict(id=comparison['id'],parts=parts))
    return result


def assemble(comparisons, observations, missing):
    from .report import validate_rows, MAX_BYTES
    from .verify import _json
    parts={}
    for observation in observations:
        data=observation['data']
        if 'comparison_part' not in data:continue
        if set(data)!={'comparison_part','ordinal','raw'} or type(data['ordinal']) is not int or type(data['raw']) is not str:
            raise ValueError('closed comparison part required')
        identity=digest(['comparison-part',data['comparison_part'],data['ordinal'],data['raw']])
        if identity!=observation['producer'] or identity in parts:
            raise ValueError('comparison part identity/ownership differs')
        parts[identity]=data
    result=[];claimed=set()
    for comparison in comparisons:
        if 'omitted' in comparison:
            missing.append('comparison_omitted:'+comparison['id']);continue
        if 'parts' not in comparison:
            result.append(comparison);continue
        raw=[]
        for ordinal,identity in enumerate(comparison['parts']):
            if identity not in parts:
                missing.append('comparison_part:'+identity);break
            part=parts[identity]
            if identity in claimed or part['comparison_part']!=comparison['id'] or part['ordinal']!=ordinal:
                raise ValueError('comparison part reused or reordered')
            claimed.add(identity);raw.append(part['raw'])
        else:
            data=''.join(raw).encode()
            if len(data)>MAX_BYTES:raise ValueError('assembled comparison byte cap')
            value=_json(data)
            if value.get('id')!=comparison['id']:raise ValueError('assembled comparison identity differs')
            validate_rows('comparisons',[value],assembled=True)
            result.append(value)
    if set(parts)-claimed:
        raise ValueError('unclaimed comparison fragments')
    return result
