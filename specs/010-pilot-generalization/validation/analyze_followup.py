#!/usr/bin/env python3
"""Offline, stdlib audit of public follow-up archives; never loads model/truth files.

python3 specs/010-pilot-generalization/validation/analyze_followup.py \
  --input-dir /path/to/zips --output-dir /path/to/analysis

Bounds assign every unresolved outcome to failure/success. They are finite-cohort
identification bounds, not confidence intervals. Completed pairs are descriptive.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from analyze_pilot import Bundle

ARMS = ('raw', 'rescue', 'equal_budget', 'shuffled', 'absent', 'template')
SPLITS = dict(training_panel=32, parameter_development=128,
              structural_development=128, oeis_development=8)


def metrics(values):
    n = len(values)
    success = sum(v['success'] for v in values)
    complete = sum(v['complete'] for v in values)
    return dict(n=n, success=success, complete=complete, unresolved=n-complete,
                lower=success/n, upper=(success+n-complete)/n)


def pooled_loss(point):
    rows = [point['splits'][k] for k in ('parameter_development', 'structural_development')]
    return sum(r['loss_sum'] for r in rows)/sum(r['tokens'] for r in rows)


def compare(cases, left, right, arm_left='equal_budget', arm_right='equal_budget'):
    a, b = cases[left], cases[right]
    rows = []
    for split in SPLITS:
        ids = [k for k, r in a['selected'].items() if r['strata']['split'] == split]
        pairs = [(a['outcomes'][k][arm_left], b['outcomes'][k][arm_right]) for k in ids]
        common = [(x, y) for x, y in pairs if x['complete'] and y['complete']]
        wins = sum(x['success'] and not y['success'] for x, y in common)
        losses = sum(y['success'] and not x['success'] for x, y in common)
        ma, mb = metrics([x for x, _ in pairs]), metrics([y for _, y in pairs])
        rows.append(dict(left=left, right=right, arm_left=arm_left, arm_right=arm_right,
            split=split, n=len(ids), completed_pairs=len(common), wins=wins, losses=losses,
            difference_lower=ma['lower']-mb['upper'], difference_upper=ma['upper']-mb['lower']))
    return rows


def save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cases, audit, groups, families, training = {}, [], [], [], []
    source_work = set()
    roster = None
    for name in ('matched', 'matched-drain', 'study-20261016'):
        path = args.input_dir/(name+'.zip')
        bundle = Bundle(path)
        assert bundle.summary['selected_scope'] == 'PASS'
        assert all(r['state'] == 'PASS' for r in bundle.summary['stages'].values())
        imported = bundle.read('stages/diagnostics.import/import.json')
        source_work.add((imported['source'], imported['work']))
        audit.append(dict(archive=name+'.zip', sha256='sha256:'+hashlib.sha256(path.read_bytes()).hexdigest(),
            manifest=bundle.manifest['identity'], payload_members=len(bundle.manifest['members']),
            elapsed_seconds=bundle.summary['elapsed_seconds'], source=imported['source'], work=imported['work'],
            omissions=bundle.manifest['omissions'], stages={k:r['state'] for k,r in bundle.summary['stages'].items()}))
        plans = [(str(i), 'diagnostics.snapshot-'+str(i), i) for i in (2,5,6)] if name != 'study-20261016' else [
            (arm, 'followup.'+arm, i) for arm,i in (('uniform',7),('family_balanced',8),('decay',9))]
        for label, stage, ordinal in plans:
            key = name+'/'+label
            prefix = 'stages/'+stage+'/'
            generated = prefix+'generate-'+str(ordinal)+'/'
            scored = prefix+'score-'+str(ordinal)+'/'
            selected = bundle.read(generated+'selections.json')
            outcomes = bundle.read(scored+'outcomes.json')
            assert set(selected) == set(outcomes) and len(selected) == 296
            current = {k:(r['family'],r['strata']) for k,r in selected.items()}
            if roster is None: roster = current
            assert current == roster, 'cohort/strata changed'
            assert Counter(r['strata']['split'] for r in selected.values()) == Counter(SPLITS)
            for row in outcomes.values():
                assert set(row) == set(ARMS)
                for v in row.values():
                    assert type(v['success']) is bool and type(v['complete']) is bool
                    assert not v['success'] or v['complete']
            clocks = bundle.read(generated+'portfolio-clocks.json')
            assert clocks['cost_complete'] and clocks['expected_portfolios'] == 1184 and not clocks['missing_portfolios']
            gen = bundle.read(generated+'summary.json')
            score = bundle.read(scored+'summary.json')
            assert gen['source'] == score['source'] == imported['source']
            assert gen['work'] == score['work'] == imported['work']
            assert gen['status'] == score['status'] == 'COMPLETE'
            case = dict(selected=selected, outcomes=outcomes, generated={
                (r['prompt'],r['mode'],r['attempt']):r['payload']
                for r in bundle.lines(generated+'attempts.jsonl') if r['event']=='GENERATED'})
            cases[key] = case
            for split,n in SPLITS.items():
                ids = [k for k,r in selected.items() if r['strata']['split'] == split]
                by_family = defaultdict(list)
                for k in ids: by_family[selected[k]['family']].append(k)
                for arm in ARMS:
                    m = metrics([outcomes[k][arm] for k in ids])
                    assert m['n'] == n
                    fm = [metrics([outcomes[k][arm] for k in ks]) for ks in by_family.values()]
                    groups.append(dict(case=key, split=split, arm=arm, **m,
                        families=len(fm), macro_lower=statistics.mean(x['lower'] for x in fm),
                        macro_upper=statistics.mean(x['upper'] for x in fm),
                        generation_seconds=gen['seconds_through_role'], scoring_seconds=score['seconds_through_role']))
                    if split == 'structural_development' and arm == 'equal_budget':
                        for family,ks in sorted(by_family.items()):
                            families.append(dict(case=key, family=family, **metrics([outcomes[k][arm] for k in ks])))
            if name != 'study-20261016':
                report = bundle.read('stages/study.report/evaluation.json')
                point = next(p for p in report['points'] if p['ordinal']==ordinal)
                for row in groups[-len(SPLITS)*len(ARMS):]:
                    reported = point['groups'][row['split']+'|'+row['arm']]
                    assert all(reported[k]==row[k] for k in ('success','complete','unresolved','lower','upper'))
            else:
                history = bundle.read(prefix+'train/validation-history.json')
                weights = next(r for r in bundle.read('stages/study.report/study.json')['arms'] if r['profile']['arm']==label)
                result = bundle.read(prefix+'train/summary.json')
                assert all(r['complete'] for h in history for r in h['splits'].values())
                assert history[-1]['tokens'] == weights['tokens'] == result['tokens']
                assert 0 < 25_000_000-weights['tokens'] < 4096
                best = min(history, key=pooled_loss)
                case['history'] = history
                for point in history:
                    for split,v in point['splits'].items():
                        training.append(dict(arm=label,tokens=point['tokens'],updates=point['updates'],
                            selected=point is best,split=split,loss=v['loss'],token_accuracy=v['token_accuracy'],
                            exact_programs=v['exact_programs'],programs=v['programs'],
                            pooled_development_loss=pooled_loss(point),train_seconds=result['seconds_through_role']))
    assert len(source_work) == 1
    assert cases['study-20261016/uniform']['history'][0]['splits'] == cases['study-20261016/decay']['history'][0]['splits'] == cases['study-20261016/family_balanced']['history'][0]['splits']
    assert [(h['tokens'],h['updates']) for h in cases['study-20261016/uniform']['history']] == [(h['tokens'],h['updates']) for h in cases['study-20261016/decay']['history']]
    paired = []
    for left,right in [('matched-drain/2','matched-drain/5'),('matched-drain/6','matched-drain/5'),
                       ('matched-drain/2','matched-drain/6'),('study-20261016/uniform','study-20261016/decay'),
                       ('study-20261016/uniform','study-20261016/family_balanced')]:
        paired.extend(compare(cases,left,right))
    for key in cases:
        if key.startswith('matched-drain/') or key.startswith('study-'):
            for arm in ('raw','rescue','absent','shuffled','template'):
                paired.extend(compare(cases,key,key,arm_right=arm))
    generation_pairs = []
    for ordinal in ('2','5','6'):
        left,right = 'matched/'+ordinal,'matched-drain/'+ordinal
        paired.extend(compare(cases,left,right))
        a,b = cases[left]['generated'],cases[right]['generated']
        for mode in ('correct','shuffled','absent','template'):
            keys = [k for k in a.keys() & b.keys() if k[1]==mode and a[k]['complete'] and b[k]['complete']]
            generation_pairs.append(dict(snapshot=int(ordinal),mode=mode,both_decoded=len(keys),
                identical_source=sum(a[k]['source_sha256']==b[k]['source_sha256'] for k in keys)))
    for filename,rows in [('groups.csv',groups),('families.csv',families),('paired.csv',paired),('training.csv',training)]:
        save_csv(output/filename,rows)
    (output/'audit.json').write_text(json.dumps(dict(archives=audit,source_work=list(source_work),
        cases=len(cases),groups=len(groups),same_roster=True,generation_pairs=generation_pairs,
        bounds='finite-cohort identification bounds; not confidence intervals',
        final_access='SEALED',seed_confirmation=False),sort_keys=True,indent=2)+'\n')
    print(f'Verified {sum(r["payload_members"] for r in audit)} payloads; {len(cases)} cases, {len(groups)} groups; wrote {output}')


if __name__ == '__main__':
    main()
