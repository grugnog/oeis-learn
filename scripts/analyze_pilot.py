#!/usr/bin/env python3
"""Read-only, stdlib audit of the three OEIS Learn pilot public ZIPs.

Usage: python analyze_pilot.py --input-dir /path/to/zips --output-dir analysis
No model loading, generation, hidden-truth access, or repository imports.
"""
import argparse
import csv
import hashlib
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path


def sha(b):
    return 'sha256:' + hashlib.sha256(b).hexdigest()


class Bundle:
    def __init__(self, path):
        self.path = path
        self.z = zipfile.ZipFile(path)
        self.manifest = self.read('manifest.json')
        body = {k:v for k,v in self.manifest.items() if k!='identity'}
        assert sha(json.dumps(body, sort_keys=True, separators=(',',':'), allow_nan=False).encode()) == self.manifest['identity']
        members = self.manifest['members']
        assert len(self.z.namelist()) == len(set(self.z.namelist())), 'duplicate ZIP paths'
        assert set(self.z.namelist()) == {r['path'] for r in members} | {'manifest.json'}
        for r in members:
            b = self.z.read(r['path'])
            assert len(b) == r['bytes'] and sha(b) == r['sha256'], r['path']
        self.contract = self.read('contract.json')
        self.summary = self.read('summary.json')

    def read(self, name):
        return json.loads(self.z.read(name))

    def lines(self, name):
        with self.z.open(name) as f:
            yield from (json.loads(l) for l in f if l.strip())


def aggregate(rows):
    n = sum(r['tokens'] for r in rows)
    return dict(rows=len(rows), tokens=n,
                loss=sum(r['loss_sum'] for r in rows)/n,
                mean_example_loss=statistics.mean(r['loss_sum']/r['tokens'] for r in rows),
                token_accuracy=sum(r['correct_tokens'] for r in rows)/n,
                exact_programs=sum(r['exact_program'] for r in rows),
                entropy=sum(r['entropy_sum'] for r in rows)/n)


def quantile(values, q):
    if not values:
        return None
    a = sorted(values)
    i = (len(a)-1)*q
    lo, hi = math.floor(i), math.ceil(i)
    return a[lo] + (a[hi]-a[lo])*(i-lo)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    bundles = {}
    audits = []
    for p in sorted(args.input_dir.glob('009-*.zip')):
        b = Bundle(p)
        v = b.contract['version']
        assert v not in bundles
        bundles[v] = b
        audits.append(dict(filename=p.name, zip_sha256=sha(p.read_bytes()),
                           manifest_identity=b.manifest['identity'],
                           payload_members=len(b.manifest['members']),
                           omissions=b.manifest.get('omissions', []),
                           selected_scope=b.summary['selected_scope'],
                           elapsed_seconds=b.summary['elapsed_seconds'],
                           stages={k:v['state'] for k,v in b.summary['stages'].items()}))
    run = bundles['pipeline-pilot-execution/v1']
    diag = bundles['pipeline-pilot-diagnostics/v1']
    admission = bundles['pipeline-pilot-validation/v2']
    imported = diag.read('stages/diagnostics.import/import.json')
    assert imported['producer_manifests']['run'] == run.manifest['identity']
    assert imported['producer_manifests']['admission'] == admission.manifest['identity']
    assert run.contract['decision'] == admission.read('stages/study.report/decision.json')
    assert imported['study'] == run.contract['decision']['identity']
    for name in ['authority', 'compatibility', 'cost', 'qualification']:
        ref = run.contract['decision'][name]
        relative = ref['path'].split('/stages/',1)[1]
        assert sha(admission.z.read('stages/'+relative)) == ref['sha256'], name

    def save(name, rows):
        if not rows:
            return
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (args.output_dir/(name+'.csv')).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    tables = defaultdict(list)
    all_rows = {}
    all_outcomes = {}
    all_selections = {}
    summaries = []
    diag_report = diag.read('stages/study.report/diagnostics.json')
    hours = [0, 1.5, 3, 6, 12, 24]
    for n in range(6):
        prefix = f'stages/diagnostics.snapshot-{n}/snapshot-{n}/'
        rows = list(diag.lines(prefix+'per-example.jsonl'))
        indexed = {r['id']:r for r in rows}
        assert len(indexed) == len(rows) == 33024
        all_rows[n] = indexed
        s = diag.read(prefix+'summary.json')
        assert s['completed_rows'] == s['expected_rows'] == len(rows)
        assert not s['missing_rows'] and not s['training'] and not s['optimizer_loaded']
        assert s['parameter_digest_before'] == s['parameter_digest_after']
        snap = run.read(f'stages/pilot.evaluate-{n}/snapshot-{n}.json')
        assert s['parameter_digest_before'] == snap['parameter_digest']
        summaries.append(dict(ordinal=n, scheduled_hours=hours[n], status=s['status'],
                              rows=len(rows), seconds=s['seconds_through_role'],
                              tokens=snap['tokens'], updates=snap['updates'],
                              epochs=snap['effective_token_epochs'],
                              learner_hours=snap['learner_role_seconds']/3600))
        splits = sorted(set(r['split'] for r in rows))
        for split in splits:
            rr = [r for r in rows if r['split']==split]
            a = aggregate(rr)
            supplied = diag_report['points'][n]['groups']['split:'+split]
            for k in ['rows','tokens','exact_programs','loss','entropy','token_accuracy']:
                assert math.isclose(a[k], supplied[k], rel_tol=1e-9, abs_tol=1e-9), (n,split,k)
            if split != 'training':
                assert math.isclose(a['loss'], snap['teacher_forcing']['splits'][split]['loss'], rel_tol=1e-5)
            tables['teacher_forcing'].append(dict(ordinal=n,scheduled_hours=hours[n],split=split,**a))
            for key in ['stage','complexity','size','length','nonaffine','constant','offset_only']:
                grouped = defaultdict(list)
                for r in rr:
                    grouped[str(r.get(key,r['strata'].get(key)))].append(r)
                for value, gr in sorted(grouped.items()):
                    tables['strata'].append(dict(ordinal=n,split=split,dimension=key,value=value,**aggregate(gr)))
            if n==5:
                fc = Counter(r['family'] for r in rr)
                by_family = defaultdict(list)
                for r in rr:
                    by_family[r['family']].append(r)
                tables['cohort'].append(dict(split=split,rows=len(rr),families=len(fc),
                    unique_targets=len({r['target_identity'] for r in rr}),
                    mean_target_tokens=a['tokens']/len(rr), min_target_tokens=min(r['tokens'] for r in rr),
                    max_target_tokens=max(r['tokens'] for r in rr),
                    largest_family=max(fc.values()),
                    top_two_family_rows=sum(v for _,v in fc.most_common(2)),
                    top_five_family_rows=sum(v for _,v in fc.most_common(5)),
                    row_weight_effective_families=len(rr)**2/sum(v*v for v in fc.values())))
                for family,count in fc.most_common():
                    fr=by_family[family]
                    tables['families'].append(dict(split=split,family=family,rows=count,stage=fr[0]['stage'],**{k:v for k,v in aggregate(fr).items() if k!='rows'}))
                for errors,count in sorted(Counter(r['tokens']-r['correct_tokens'] for r in rr).items()):
                    tables['error_histogram'].append(dict(ordinal=n,split=split,errors=errors,rows=count))
        controls = diag.read(prefix+'matched-controls.json')['rows']
        ci = {(r['id'],r['mode']):r for r in controls}
        assert len(ci)==len(controls)==864
        for split in splits:
            ids = [pid for pid,mode in ci if mode=='correct' and indexed[pid]['split']==split]
            for mode in ['correct','shuffled','absent']:
                rr = [ci[pid,mode] for pid in ids]
                for r in rr:
                    assert r['target_identity']==indexed[r['id']]['target_identity']
                tables['matched_controls'].append(dict(ordinal=n,split=split,mode=mode,**aggregate(rr)))
                if mode!='correct':
                    base=[ci[pid,'correct'] for pid in ids]
                    tokens=sum(r['tokens'] for r in base)
                    diffs=[b['loss_sum']/b['tokens']-a['loss_sum']/a['tokens'] for a,b in zip(base,rr)]
                    tables['control_effects'].append(dict(ordinal=n,split=split,mode=mode,rows=len(ids),
                        token_weighted_loss_increase=sum(b['loss_sum']-a['loss_sum'] for a,b in zip(base,rr))/tokens,
                        token_accuracy_change=sum(b['correct_tokens']-a['correct_tokens'] for a,b in zip(base,rr))/tokens,
                        median_example_loss_increase=statistics.median(diffs),
                        examples_worse=sum(d>1e-6 for d in diffs),
                        examples_better=sum(d < -1e-6 for d in diffs)))
        outcome=run.read(f'stages/pilot.evaluate-{n}/score-{n}/outcomes.json')
        selected=run.read(f'stages/pilot.evaluate-{n}/generate-{n}/selections.json')
        assert set(outcome)==set(selected) and len(outcome)==296
        all_outcomes[n]=outcome
        all_selections[n]=selected
        for split in ['training_panel','parameter_development','structural_development','oeis_development']:
            ids=[k for k,v in selected.items() if v['strata']['split']==split]
            for arm in ['raw','rescue','equal_budget','shuffled','absent','template']:
                rr=[outcome[k][arm] for k in ids]
                assert all(not r['success'] or r['complete'] for r in rr)
                success=sum(r['success'] for r in rr)
                complete=sum(r['complete'] for r in rr)
                tables['semantic_outcomes'].append(dict(ordinal=n,split=split,arm=arm,
                    denominator=len(ids),success=success,complete=complete,unresolved=len(ids)-complete,
                    measured_failure=complete-success, lower=success/len(ids),
                    upper=(success+len(ids)-complete)/len(ids)))
                counts=Counter((r['complete'],r['reason']) for r in rr)
                for (complete,reason),count in sorted(counts.items()):
                    tables['outcome_reasons'].append(dict(ordinal=n,split=split,arm=arm,complete=complete,reason=reason,count=count))
        for pid in sorted(set(outcome)&set(indexed)):
            r=indexed[pid]
            tables['teacher_semantic_join'].append(dict(ordinal=n,id=pid,split=selected[pid]['strata']['split'],
                stage=r['stage'],family=r['family'],canonical_exact=r['exact_program'],
                mean_loss=r['loss_sum']/r['tokens'],tokens=r['tokens'],
                **{arm+'_'+k:outcome[pid][arm][k] for arm in ['raw','rescue','equal_budget'] for k in ['success','complete','reason']}))
        events=list(run.lines(f'stages/pilot.evaluate-{n}/generate-{n}/attempts.jsonl'))
        for mode in sorted(set(r['mode'] for r in events)):
            me=[r for r in events if r['mode']==mode]
            committed=[r for r in me if r['event']=='COMMITTED']
            for fit_result,count in Counter(r['payload']['fitting']['outcome'] for r in committed).items():
                tables['native_outcomes'].append(dict(ordinal=n,mode=mode,fit_outcome=fit_result,count=count))
            generated=[r for r in me if r['event']=='GENERATED']
            hashes=[r['payload'].get('source_sha256') for r in generated if r['payload'].get('source_sha256')]
            tables['generation'].append(dict(ordinal=n,mode=mode,started=sum(r['event']=='STARTED' for r in me),
                generated=len(generated),generated_complete=sum(r['payload'].get('complete',False) for r in generated),
                committed=len(committed),incomplete_events=sum(r['event']=='INCOMPLETE' for r in me),
                unique_generated_sources=len(set(hashes)),source_records=len(hashes),
                most_common_source_count=max(Counter(hashes).values(),default=0)))
            terminal=[r for r in me if r['event'] in ['COMMITTED','INCOMPLETE']]
            for event in ['COMMITTED','INCOMPLETE']:
                clocks=[r['payload'].get('fitting',{}).get('clock',{}) for r in terminal if r['event']==event]
                for label,start,end in [('queue','admitted','dispatched'),('service','dispatched','completed'),('owner_delay','completed','owner_observed')]:
                    values=[c[end]-c[start] for c in clocks if c.get(start) is not None and c.get(end) is not None]
                    tables['native_timing'].append(dict(ordinal=n,mode=mode,event=event,phase=label,
                        count=len(values),missing_clocks=len(clocks)-len(values),
                        p50_seconds=quantile(values,.5),p90_seconds=quantile(values,.9),max_seconds=max(values,default=None)))
        for v in run.read(f'stages/pilot.train-{n}/train-{n}/validation-history.json'):
            for split,a in v['splits'].items():
                tables['validation_history'].append(dict(ordinal=n,update=v['update'],tokens=v['tokens'],
                    split=split,loss=a['loss'],token_accuracy=a['token_accuracy'],exact_programs=a['exact_programs']))
        acts=diag.read(prefix+'activations.json')
        assert all(v==64 for v in acts['completed_rows'].values())
        for mode,r in acts['representations'].items():
            tables['representations'].append(dict(ordinal=n,mode=mode,**{k:v for k,v in r.items() if k!='interpretation'},
                memory_rms=acts['summaries'][mode+'|memory']['rms'],logits_rms=acts['summaries'][mode+'|logits']['rms']))
        tensors=diag.read(prefix+'tensor-comparisons.json')['tensors']
        for group in ['encoder','decoder']:
            ts=[r for r in tensors if r['name'].startswith(group+'.') and r['kind']=='floating_state' and not r['name'].endswith('.pe')]
            assert all(math.isfinite(r['summary']['norm']) for r in ts)
            tables['tensors'].append(dict(ordinal=n,module=group,states=len(ts),
                unchanged_from_initial=sum(r['versus_initial']['delta_norm']==0 for r in ts),
                state_l2=math.sqrt(sum(r['summary']['norm']**2 for r in ts)),
                delta_from_initial_l2=math.sqrt(sum(r['versus_initial']['delta_norm']**2 for r in ts)),
                delta_from_previous_l2=math.sqrt(sum(r['versus_previous']['delta_norm']**2 for r in ts))))
    for a,b in [(0,1),(1,2),(2,3),(3,4),(4,5),(2,5)]:
        ia,ib=all_rows[a],all_rows[b]
        assert set(ia)==set(ib)
        for pid in ia:
            assert (ia[pid]['target_identity'],ia[pid]['tokens'],ia[pid]['split']) == (ib[pid]['target_identity'],ib[pid]['tokens'],ib[pid]['split'])
        for split in ['training','parameter_development','structural_development']:
            ids=[pid for pid in ia if ia[pid]['split']==split]
            deltas=[ib[k]['loss_sum']/ib[k]['tokens']-ia[k]['loss_sum']/ia[k]['tokens'] for k in ids]
            loss_delta=sum(ib[k]['loss_sum']-ia[k]['loss_sum'] for k in ids)
            contributions=sorted((ib[k]['loss_sum']-ia[k]['loss_sum'] for k in ids),reverse=True)
            tables['paired_changes'].append(dict(from_ordinal=a,to_ordinal=b,split=split,rows=len(ids),
                loss_increased=sum(x>1e-6 for x in deltas),loss_decreased=sum(x < -1e-6 for x in deltas),
                median_loss_change=statistics.median(deltas),mean_loss_change=statistics.mean(deltas),
                token_weighted_loss_change=loss_delta/sum(ia[k]['tokens'] for k in ids),
                exact_gained=sum(not ia[k]['exact_program'] and ib[k]['exact_program'] for k in ids),
                exact_lost=sum(ia[k]['exact_program'] and not ib[k]['exact_program'] for k in ids),
                top_10_share_net_loss_increase=sum(contributions[:10])/loss_delta if loss_delta>0 else None))
        for split in ['training_panel','parameter_development','structural_development','oeis_development']:
            ids=[k for k,v in all_selections[a].items() if v['strata']['split']==split]
            for arm in ['raw','rescue','equal_budget','shuffled','absent','template']:
                pa=[(all_outcomes[a][k][arm],all_outcomes[b][k][arm]) for k in ids]
                complete=[(x,y) for x,y in pa if x['complete'] and y['complete']]
                tables['semantic_transitions'].append(dict(from_ordinal=a,to_ordinal=b,split=split,arm=arm,
                    denominator=len(ids),both_complete=len(complete),
                    gains=sum(not x['success'] and y['success'] for x,y in complete),
                    losses=sum(x['success'] and not y['success'] for x,y in complete),
                    unresolved_pairs=len(ids)-len(complete)))
    tables['milestones']=summaries
    for name,rows in tables.items():
        save(name,rows)
    result=dict(archives=audits,bindings_verified=True,diagnostic_rows_verified=6*33024,
                control_rows_verified=6*864,activation_rows_verified=6*3*64,
                semantic_outcomes=6*296*6,source_identity=imported['producer_source'],
                diagnostic_source_identity=imported['source'],tables={k:len(v) for k,v in tables.items()})
    (args.output_dir/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
