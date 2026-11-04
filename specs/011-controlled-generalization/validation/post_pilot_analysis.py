#!/usr/bin/env python3
"""Read-only, selection-only audit of a public fresh-pilot report (no engine/GPU)."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from oeis_learn.pipeline_validation.bundle import verify, pack
from oeis_learn.pipeline_validation.identity import file_hash, write
from oeis_learn.synthesis.admission import source_identity
from oeis_learn.synthesis.contracts import identity
from oeis_learn.synthesis.reference_parser import parse
from oeis_learn.synthesis.reference import evaluate
from oeis_learn.generalization.diagnostics import signature
from oeis_learn.generalization.targets import ground
from oeis_learn.decoder.bigint_codec import decode_body

CONTROLS = {'raw','rescue','equal_budget','shuffled','absent','template','bank'}
MODES = ('correct','shuffled','absent','template')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_input(path):
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        require(len(infos) <= 4097 and sum(i.file_size for i in infos) <= 512 << 20,
                'bounded public archive required')
        require(all(i.file_size <= 128 << 20 for i in infos), 'member exceeds128MiB')
    return verify(path)


def count_outcomes(rows):
    totals = defaultdict(Counter)
    for controls in rows.values():
        for control, r in controls.items():
            require(type(r['success']) is bool and type(r['complete']) is bool,
                    'boolean score/completion required')
            require(not r['success'] or r['complete'], 'success cannot be incomplete')
            c = totals[control]
            c['n'] += 1
            c['success'] += r['success']
            c['complete'] += r['complete']
            c['unresolved'] += not r['complete']
            c['reason:' + str(r.get('reason'))] += 1
    return dict(totals)


def compare(a, b):
    require(set(a) == set(b), 'paired task roster mismatch')
    result = defaultdict(lambda: dict(both_success=0, static_only=0,
                                     refreshed_only=0, neither_success=0))
    for task in sorted(a):
        require(set(a[task]) == set(b[task]), 'paired control roster mismatch')
        for control, x in a[task].items():
            y = b[task][control]
            key = ('both_success' if x['success'] and y['success'] else
                   'static_only' if x['success'] else 'refreshed_only' if y['success']
                   else 'neither_success')
            result[control][key] += 1
    return dict(result)


def paired_candidate_counts(a,b):
    require(set(a)==set(b),'paired candidate roster mismatch')
    counts=Counter()
    for task in a:
        x=a[task]['details']['correct']['details'][0].get('source')
        y=b[task]['details']['correct']['details'][0].get('source')
        counts['n']+=1
        counts['first_original_equal']+=bool(x and y and x==y)
        counts['first_structure_equal']+=bool(x and y and signature(x)['signature']==signature(y)['signature'])
        ex=a[task]['selections']['equal_budget'];ey=b[task]['selections']['equal_budget']
        counts['both_selected']+=bool(ex and ey)
        counts['selected_source_equal']+=bool(ex and ey and ex['selected']['source']==ey['selected']['source'])
    return dict(counts)


def check_envelope(e):
    s, v = e['selected'], e['verification']
    require(source_identity(e['original_source']) == e['original_source_sha256'],
            'original source hash mismatch')
    require(source_identity(s['source']) == s['source_sha256'] == v['source_sha256'],
            'selected source hash mismatch')
    require(s['seal_sha256'] == identity({k:x for k,x in s.items()
            if k not in {'seal_sha256','selection'}}), 'selected seal mismatch')
    require(v['scope'] == 'visible_only' and v['admitted'] and s['verified']
            and s['visible_count'] == 20 and v['indices'] == list(map(str,range(20)))
            and len(v['values']) == 20, 'visible-only verification required')
    require(parse(ground(e['transformed_source'], e['assignment'])) == parse(s['source']),
            'assignment grounding mismatch')
    if 'provenance' in s:
        p = s['provenance']
        require(p['original_source_sha256'] == e['original_source_sha256']
                and p['transformed_source_sha256'] == source_identity(e['transformed_source'])
                and p['assignment'] == e['assignment'] and p['verification'] == identity(v),
                'selected provenance mismatch')
    if 'prompt_sha256' in e:
        from oeis_learn.synthesis.admission import VisiblePrompt
        prompt = VisiblePrompt(tuple(v['indices']), tuple(v['values']))
        require(prompt.sha256 == e['prompt_sha256'] == v['prompt_sha256'],
                'visible prompt hash mismatch')


def bank_envelope(e, bank):
    """Reject defective legacy text; explicitly rejoin its hash to frozen bank bytes."""
    if source_identity(e['original_source']) == e['original_source_sha256']:
        return e, None
    matches = [p['source'] for p in bank['proposals'] if p['source'] and
               source_identity(p['source']) == e['original_source_sha256']]
    require(len(set(matches)) == 1, 'bank original unavailable or ambiguous')
    require(e['original_source'] == e['transformed_source'], 'unrecognized bank provenance defect')
    return dict(e, original_source=matches[0]), dict(kind='bank_original_source_mismatch',
        retained_text_sha256=source_identity(e['original_source']),
        recovered_original_sha256=e['original_source_sha256'],
        recovery='diagnostic-only join to archived proposal-bank; original envelope unchanged')


def prompt_for(record):
    prompts = [(e['verification']['indices'],e['verification']['values'])
               for e in record['selections'].values() if e]
    require(not prompts or all(p == prompts[0] for p in prompts),
            'conflicting visible prompts')
    return prompts[0] if prompts else None


def check_attempt(a):
    source = a.get('source')
    if not a['generation']['complete']:
        require(not source, 'incomplete generation has executable source')
        return dict(parse_status='unavailable', signature=None, parameters=None)
    require(source and source_identity(source) == a['source_sha256'], 'attempt source hash mismatch')
    require(decode_body(a['tokens']) == source, 'retained token/source mismatch')
    attrs = signature(source)
    require(attrs['parse_status'] == 'ok', 'completed candidate fails independent parser')
    return attrs


def execute(source, prompt, deadline):
    if not source:
        return dict(status='unavailable', reason='source_missing')
    if time.monotonic() >= deadline:
        raise TimeoutError('audit deadline reached')
    # A local audit timeout is not a new scientific failure or a HIP measurement.
    r = evaluate(source, prompt[0] if prompt else list(map(str,range(20))),
                 deadline=min(deadline,time.monotonic()+0.1))
    return dict(status=r.outcome, reason=r.reason, coverage=r.coverage,
                visible_match=list(r.values) == prompt[1] if prompt is not None and r.outcome == 'ok' else None,
                visible_values_available=prompt is not None,
                first_values=list(r.values[:5]))


def features(source):
    p = parse(source, allow_parameters=True)
    ints, seams, operators = [], [], Counter()
    def visit(n):
        if isinstance(n,str): return
        operators[n[0]] += 1
        if n[0]=='int': ints.append(int(n[1]))
        if n[0] in {'eq','lt','le'} and n[1]=='$index' and isinstance(n[2],tuple) and n[2][0]=='int':
            seams.append(dict(operator=n[0], threshold=int(n[2][1])))
        for child in n[1:]: visit(child)
    for n in (*p.statements,p.result): visit(n)
    return dict(literals=len(ints), max_literal_digits=max((len(str(abs(x))) for x in ints),default=0),
                index_comparisons=seams, operators=dict(operators))


def inspect_arm(records, outcomes, deadline, bank=None):
    require(len(records)==256,'frozen256-task pilot roster required')
    require(set(records) == set(outcomes), 'candidate/scoring roster mismatch')
    counts, shapes, sources = defaultdict(Counter), defaultdict(set), defaultdict(set)
    selected_checks, raw_checks, tasks = Counter(), Counter(), {}
    findings=[]
    for task, r in sorted(records.items()):
        if time.monotonic() >= deadline: raise TimeoutError('audit deadline reached')
        require(set(r['selections']) == set(outcomes[task]) == CONTROLS,
                'control roster mismatch')
        prompt = prompt_for(r)
        t = dict(prompt_available=prompt is not None, attempts={}, selected={})
        for control, e in r['selections'].items():
            score = outcomes[task][control]
            require((e['selected']['source_sha256'] if e else None) == score['candidate_sha256'],
                    'score/selection identity mismatch')
            if not e: continue
            if control == 'bank' and bank is not None:
                e, finding = bank_envelope(e, bank)
                if finding: findings.append(dict(task=task, control=control, **finding))
            check_envelope(e)
            check = execute(e['selected']['source'],prompt,deadline)
            selected_checks['total'] += 1
            selected_checks['status:'+check['status']] += 1
            selected_checks['visible_match:'+str(check.get('visible_match'))] += 1
            require(check.get('visible_match') is not False, 'selected visible reference mismatch')
            mode = {'raw':'correct','rescue':'correct','equal_budget':'correct'}.get(control,control)
            if mode in MODES:
                ordinal = e['selected']['attempt_ordinal']
                original = next((a for a in r['details'][mode]['details'] if a['position']==ordinal),None)
                require(original is not None and original['source_sha256']==e['original_source_sha256'],
                        'selected/original attempt mismatch')
            t['selected'][control] = dict(reference=check,features=features(e['selected']['source']))
        for mode in MODES:
            attempts = r['details'][mode]['details']
            require({a['position'] for a in attempts} == set(range(8)) and len(attempts)==8, 'eight distinct attempt ordinals required')
            c = counts[mode]
            for a in attempts:
                attrs = check_attempt(a)
                c['attempts'] += 1
                c['generation_complete'] += a['generation']['complete']
                c['parse:'+attrs['parse_status']] += 1
                if a.get('source'):
                    shapes[mode].add(attrs['signature']); sources[mode].add(a['source_sha256'])
                    c['explicit_parameters'] += attrs['parameters'] > 0
                    f = features(a['source'])
                    c['over_4_digit_literal'] += f['max_literal_digits'] > 4
                    c['index_comparison_at_20'] += any(x['threshold']==20 for x in f['index_comparisons'])
                fit = a.get('fitting',{})
                c['fit:'+str(fit.get('outcome'))] += 1
                c['phase:'+str(fit.get('phase'))] += 1
                c['reason:'+str(fit.get('reason') or a['generation'].get('reason'))] += 1
                c['fit_complete:'+str(fit.get('complete'))] += 1
                c['verified_but_search_incomplete'] += fit.get('outcome')=='verified_solution' and not fit.get('complete')
                for assignment in (a.get('native') or {}).get('assignments',[]):
                    require(assignment['original_source_sha256']==a['source_sha256'] and
                            source_identity(assignment['source'])==assignment['source_sha256'],
                            'native assignment source mismatch')
                    require(parse(ground(assignment['transformed_source'],assignment['assignment'])) ==
                            parse(assignment['source']), 'native assignment grounding mismatch')
                    c['native_assignments_checked'] += 1
                if mode=='correct':
                    check = execute(a.get('source'),prompt,deadline)
                    raw_checks['status:'+check['status']] += 1
                    raw_checks['visible_match:'+str(check.get('visible_match'))] += 1
                    t['attempts'][str(a['position'])] = check
        tasks[task] = t
    for mode in MODES:
        counts[mode]['unique_structures'] = len(shapes[mode])
        counts[mode]['unique_sources'] = len(sources[mode])
    return dict(outcomes=count_outcomes(outcomes), attempts=dict(counts),
                selected_reference=dict(selected_checks), original_reference=dict(raw_checks), tasks=tasks, findings=findings)


def incomplete_attempt(a):
    return not a['generation']['complete'] or not a.get('fitting',{}).get('complete',False)


def branch_visibility(source):
    """Conservative syntactic visibility for the frozen sampler's single D guard.

    No target coefficients or hidden values are reconstructed. Non-D forms are
    explicitly outside this diagnostic; it is not a general identifiability proof.
    """
    p=parse(source,allow_parameters=True)
    if len(p.statements)!=1 or p.statements[0][0]!='if':
        return dict(status='not_single_branch',unseen_parameters=[])
    _,guard,yes,no=p.statements[0]
    if guard[0] not in {'lt','le','eq'} or guard[1]!='$index' or guard[2][0]!='int':
        return dict(status='unsupported_guard',unseen_parameters=[])
    threshold=int(guard[2][1])
    def pred(i):return i<threshold if guard[0]=='lt' else i<=threshold if guard[0]=='le' else i==threshold
    def params(n):
        if isinstance(n,str):return set()
        if n[0]=='param.get':return {n[1]}
        return set().union(*(params(x) for x in n[1:]))
    seen=set();later=set();paths={True:yes,False:no}
    for truth in {pred(i) for i in range(20)}:seen.update(params(paths[truth]))
    for truth in {pred(i) for i in range(20,100)}:later.update(params(paths[truth]))
    seen.update(params(p.result))
    return dict(status='analyzed',guard=guard[0],threshold=threshold,
                visible_paths=sorted({pred(i) for i in range(20)}),
                unseen_parameters=sorted(later-seen))


def branch_witness(source, unseen, deadline):
    """Construct a family-level ambiguity witness, never an actual target row."""
    p=parse(source,allow_parameters=True); names=set()
    def visit(n):
        if isinstance(n,str):return
        if n[0]=='param.get':names.add(n[1])
        for x in n[1:]:visit(x)
    for n in (*p.statements,p.result):visit(n)
    base={name:'2' for name in sorted(names)}
    a=evaluate(ground(source,base),list(map(str,range(100))),deadline=deadline)
    require(a.outcome=='ok','ambiguity witness reference unavailable')
    for name in unseen:
        changed=dict(base,**{name:'3'})
        b=evaluate(ground(source,changed),list(map(str,range(100))),deadline=deadline)
        require(b.outcome=='ok','ambiguity witness reference unavailable')
        if a.values[:20]==b.values[:20] and a.values!=b.values:
            return dict(scope='constructed family ambiguity, not actual held-out coefficients',
                        baseline=base,alternative=changed,visible_equal=True,
                        first_difference_index=next(i for i,(x,y) in enumerate(zip(a.values,b.values)) if x!=y))
    return None


def family_audit(plan, scores, deadline):
    """Match public selection family identities to source-only sampler descriptors."""
    from oeis_learn.data.bigint_programs import BigintSampler, canonical_family
    from oeis_learn.pipeline_validation.identity import digest
    needed={plan['roster'][digest(['static','equal_budget',task])]['family'] for task in scores['static']}
    catalog={};sampler=BigintSampler(20261016)
    for ordinal in range(4096):
        if time.monotonic()>=deadline:raise TimeoutError('family audit deadline reached')
        d=sampler.descriptor(ordinal)
        if d['family'] in needed:
            require(canonical_family(d['parameter_source'])==d['family'],'descriptor family identity mismatch')
            catalog.setdefault(d['family'],d)
    require(set(catalog)==needed,'selection family missing from frozen source catalog')
    families={key:dict(stage=d['stage'],parameter_source=d['parameter_source'],
                      visibility=branch_visibility(d['parameter_source'])) for key,d in catalog.items()}
    for f in families.values():
        f['ambiguity_witness']=branch_witness(f['parameter_source'],f['visibility']['unseen_parameters'],deadline) if f['visibility']['unseen_parameters'] else None
    rows={};groups=defaultdict(Counter)
    for task in sorted(scores['static']):
        meta=plan['roster'][digest(['static','equal_budget',task])]
        f=families[meta['family']]
        unseen=bool(f['visibility']['unseen_parameters'])
        rows[task]=dict(family=meta['family'],stratum=meta['cell'].split('/')[-1],stage=f['stage'],
                        unseen_branch_parameters=unseen, family_ambiguity_witness=f['ambiguity_witness'] is not None)
        for arm in scores:
            other=plan['roster'][digest([arm,'equal_budget',task])]
            require(other['family']==meta['family'],'paired family mismatch')
            score=scores[arm][task]['equal_budget']
            for label in (f['stage'], 'unseen_branch_parameters' if unseen else 'no_unseen_branch_parameters',
                          'family_ambiguity_witness' if f['ambiguity_witness'] else 'no_family_ambiguity_witness'):
                c=groups[arm+'/'+label];c['n']+=1;c['success']+=score['success'];c['unresolved']+=not score['complete']
                c['continuation_failure']+=score.get('reason')=='continuation_mismatch'
    return dict(method='selection family SHA matched to frozen seed20261016 catalog0..4095; no target values reconstructed',
                families=families,rows=rows,groups=dict(groups))


def sample_cases(all_records, all_scores, audits):
    from oeis_learn.decoder.bigint_codec import ID_TO_TOKEN
    cases=[]
    for arm in ('static','refreshed'):
        other = 'refreshed' if arm=='static' else 'static'
        buckets=defaultdict(list)
        for task in sorted(all_scores[arm]):
            s=all_scores[arm][task]['equal_budget']; o=all_scores[other][task]['equal_budget']
            if s['success']: category='shared_success' if o['success'] else 'arm_only_success'
            elif not s['complete']: category='unresolved'
            elif s['candidate_sha256']: category='continuation_failure'
            else: category='complete_no_fit'
            buckets[category].append(task)
            if all_scores[arm][task]['template']['success'] and not s['success']:
                buckets['template_beats_model'].append(task)
            if any(not a['generation']['complete'] for a in all_records[arm][task]['details']['correct']['details']):
                buckets['decode_failure'].append(task)
        for category,ids in sorted(buckets.items()):
            for task in ids[:2]:
                r=all_records[arm][task]; selected=r['selections']['equal_budget']
                attempts=r['details']['correct']['details']
                ordinal=selected['selected']['attempt_ordinal'] if selected else attempts[0]['position']
                if category in {'unresolved','decode_failure'}:
                    relevant=[a for a in attempts if not (a['generation']['complete'] if category=='decode_failure' else not incomplete_attempt(a))]
                    if relevant: ordinal=relevant[0]['position']
                a=next(a for a in attempts if a['position']==ordinal)
                # The failed attempt and portfolio selection need not be the same candidate.
                cases.append(dict(arm=arm,category=category,task=task,attempt_ordinal=ordinal,
                    score=all_scores[arm][task]['equal_budget'],original_source=a.get('source'),
                    original_reference=audits[arm]['tasks'][task]['attempts'][str(ordinal)],
                    generation=a['generation'],token_count=len(a['tokens']),token_tail=[ID_TO_TOKEN.get(t,str(t)) for t in a['tokens'][-40:]],
                    fitting=a.get('fitting'),native=a.get('native'),
                    selected=selected,selected_reference=audits[arm]['tasks'][task]['selected'].get('equal_budget'),
                    template=r['selections']['template'] if category=='template_beats_model' else None))
    return cases


def case_markdown(cases):
    lines=['# Deterministic before/after case review','',
           'Two lexicographically first task IDs per category and arm where available. Categories overlap.',
           'Recorded100-term scores are retained; reference checks below cover only the visible20.',
           'A failed-attempt ordinal can differ from the selected portfolio ordinal. Null means no admitted final program.','']
    for i,c in enumerate(cases,1):
        lines += [f"## {i}. {c['arm']} / {c['category']}",'',f"Task `{c['task']}`; inspected attempt {c['attempt_ordinal']}.",'',
                  f"Recorded primary score: success={c['score']['success']}, complete={c['score']['complete']}, reason={c['score'].get('reason')}.",
                  f"Original reference: `{json.dumps(c['original_reference'],sort_keys=True)}`.",'', 'Original:', '```text',c['original_source'] or 'null','```']
        e=c['selected']
        if e:
            lines += [f"Portfolio selected attempt {e['selected']['attempt_ordinal']}; assignment `{json.dumps(e['assignment'],sort_keys=True)}`.",'',
                      'Transformed:','```text',e['transformed_source'],'```','Final:','```text',e['selected']['source'],'```',
                      f"Final reference/features: `{json.dumps(c['selected_reference'],sort_keys=True)}`."]
        else: lines += ['Final: null (no admitted portfolio selection).']
        lines += [f"Generation: `{json.dumps(c['generation'],sort_keys=True)}`; {c['token_count']} tokens; tail `{c['token_tail']}`.",f"Fit: `{json.dumps(c['fitting'],sort_keys=True)}`.",'']
        if c['template']:
            lines += ['Successful template final:','```text',c['template']['selected']['source'],'```','']
    return '\n'.join(lines)+'\n'


def run(bundle, output, *, seconds=600):
    output=Path(output)
    if output.exists() or output.with_suffix('.zip').exists(): raise FileExistsError('fresh audit output required')
    output.mkdir(parents=True)
    started=time.monotonic(); deadline=started+seconds
    summary=dict(version='post-pilot-audit/v1',selected_scope='FAIL',qualified=False,
                 scientific_status='UNCHANGED',source_bundle=None,helper_sha256=file_hash(__file__),
                 source_mutated=False,hidden_truth_available=False,seconds=seconds)
    try:
        summary['source_bundle']=file_hash(bundle)
        manifest=verify_input(bundle)
        summary['verified_members']=len(manifest['members'])
        records,scores,audits,lineage={},{},{},{}
        with zipfile.ZipFile(bundle) as z:
            selection=json.loads(z.read('payloads/fresh/selection.json'))
            require(selection['assessment']=='SEALED','sealed assessment required')
            for arm in ('static','refreshed'):
                records[arm]=json.loads(z.read(f'payloads/fresh/generation-{arm}/selections.json'))
                scored=json.loads(z.read(f'payloads/fresh/scoring-{arm}.json'))
                ref=next(r for r in manifest['members'] if r['path']==f'payloads/fresh/generation-{arm}/selections.json')
                require(scored['candidates']['sha256']==ref['sha256'] and scored['candidates']['bytes']==ref['bytes'],
                        'scoring/candidate file mismatch')
                scores[arm]=scored['result']['outcomes']
                bank=json.loads(z.read(f'payloads/fresh/generation-{arm}/proposal-bank.json'))
                evaluation=json.loads(z.read(f'payloads/fresh/generation-{arm}/evaluation.json'))
                checkpoint=selection['checkpoints'][arm]['checkpoint']
                require(evaluation['checkpoint']==bank['model']==checkpoint['weights']['sha256'],'model/checkpoint identity mismatch')
                lineage[arm]=dict(weights=checkpoint['weights']['sha256'],initializer=checkpoint['initialization'])
                audits[arm]=inspect_arm(records[arm],scores[arm],deadline,bank)
                print(arm+': candidate audit complete',flush=True)
        paired=compare(scores['static'],scores['refreshed'])
        for task in records['static']:
            a,b=prompt_for(records['static'][task]),prompt_for(records['refreshed'][task])
            require(a is None or b is None or a==b, 'paired visible prompt mismatch')
        with zipfile.ZipFile(bundle) as z:
            families=family_audit(json.loads(z.read('payloads/fresh/measurement-plan.json')),scores,deadline)
        cases=sample_cases(records,scores,audits)
        for c in cases:c['family']=families['rows'][c['task']]
        for arm in audits:
            audits[arm]['primary_selected_features']=dict(Counter(
                'comparison_at_20' if any(x['threshold']==20 for x in t['selected']['equal_budget']['features']['index_comparisons']) else 'no_comparison_at_20'
                for t in audits[arm]['tasks'].values() if 'equal_budget' in t['selected']))
            audits[arm]['tasks_without_visible_prompt']=sum(not t['prompt_available'] for t in audits[arm]['tasks'].values())
        write(output/'audit.json',dict(arms=audits,paired_outcomes=paired,paired_candidates=paired_candidate_counts(records['static'],records['refreshed']),lineage=lineage,family_analysis=families))
        write(output/'cases.json',cases)
        (output/'cases.md').write_text(case_markdown(cases))
        summary.update(selected_scope='PASS',cases=len(cases),integrity_findings=sum(len(a['findings']) for a in audits.values()))
    except Exception as exc:
        summary['error']=type(exc).__name__+': '+str(exc)
    finally:
        summary['elapsed_seconds']=time.monotonic()-started
        write(output/'summary.json',summary)
        pack(output,mandatory=[p.name for p in sorted(output.iterdir()) if p.is_file()],cap=64<<20)
    if summary['selected_scope']!='PASS': raise RuntimeError(summary['error'])
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seconds',type=int,default=600,choices=range(1,1801),metavar='1..1800')
    args=p.parse_args()
    print(json.dumps(run(args.bundle,args.output,seconds=args.seconds),indent=2))


if __name__=='__main__':main()
