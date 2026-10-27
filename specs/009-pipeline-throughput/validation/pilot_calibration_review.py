"""Reproduce public calibration integrity, clock, cost and capability audits."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from oeis_learn.pipeline_validation.bundle import verify
from oeis_learn.pipeline_validation.identity import seal,file_hash
from oeis_learn.pipeline_validation.pilot_clocks import clock_record,ATTEMPTS,POLICY
from oeis_learn.pipeline_validation.pilot_calibration_cost import forecast
from oeis_learn.pipeline_validation.pilot_contract import ARMS,SPLITS,SPEC
from oeis_learn.pipeline_validation.model_sizes import parameters,PROFILES,EXECUTION


def require(value,message):
    if not value:raise ValueError(message)


def clocks_audit(clocks,attempts,records):
    expected={(p,mode,i) for p in records for mode,n in ATTEMPTS.items() for i in range(n)}
    jobs={(j['prompt'],j['mode'],j['attempt']):j['state'] for j in attempts['jobs']}
    require(seal(clocks)==clocks and seal(attempts)==attempts,'changed clock/attempt seal')
    require(len(records)==296 and set(jobs)==expected and len(attempts['jobs'])==5328,'changed registered job roster')
    require(attempts['expected']==5328 and dict(Counter(jobs.values()))==attempts['states'],'changed attempt counts')
    require(set(jobs.values())<={'COMMITTED','INCOMPLETE','UNSTARTED'},'active/unaccounted attempt')
    require(clocks['attempts']==attempts['identity'] and clocks['policy']==POLICY,'clock/attempt binding mismatch')
    require(len(clocks['clocks'])==1184 and clocks['missing_portfolios']==0 and clocks['expected_portfolios']==1184,'missing portfolios')
    seen=set();snapshots=set()
    for r in clocks['clocks']:
        key=(r['prompt'],r['mode']);require(key not in seen and (*key,0) in jobs,'duplicate/foreign portfolio')
        seen.add(key);snapshots.add(r['snapshot'])
        actual=clock_record(**{k:r[k] for k in ('prompt','mode','snapshot','start','scope_deadline','finish','expected','started','terminal')})
        require(r==actual and r['cost_complete'],'cut or altered portfolio clock')
        states=[jobs[(*key,i)] for i in range(ATTEMPTS[key[1]])]
        require(r['started']==sum(s!='UNSTARTED' for s in states) and r['terminal']==sum(s in ('COMMITTED','INCOMPLETE') for s in states),'clock/ledger count mismatch')
    require(len(snapshots)==1 and clocks['cost_complete'],'changed model within snapshot')
    require(dict(Counter(r['termination'] for r in clocks['clocks']))==clocks['terminations'],'changed termination counts')
    return dict(clock_identity=clocks['identity'],snapshot=next(iter(snapshots)),portfolios=1184,
        terminations=clocks['terminations'],attempts=attempts['states'])


def review(path):
    path=Path(path);manifest=verify(path)
    with zipfile.ZipFile(path) as z:
        read=lambda name:json.loads(z.read(name))
        inv=read('invocation.json');summary=read('summary.json');source=inv['binding']['source']
        inventory={m['path']:m for m in manifest['members']}
        for key,r in summary['stages'].items():
            require(seal(r)==r and r==read(f'stages/{key}/receipt.json'),'changed receipt/summary')
            require(r['state']=='PASS' and r['binding']['source']==source,'incomplete or foreign-source stage')
            for name,sha in r['files'].items():
                require(inventory[f'stages/{key}/{name}']['sha256']==sha,'unbound stage file')
            for parent,sha in r.get('prerequisites',{}).items():
                require(sha==summary['stages'][parent]['identity'],'broken prerequisite link')
        cost=read('stages/calibration.report/candidate-cost.json');imp=read('stages/calibration.import/import.json')
        require(cost==forecast(cost['observations'],source=source,work=imp['work'],archives=cost['archives']),'cost forecast differs from observations')
        snapshots=[]
        for i in range(3):
            root=f'stages/calibration.session/'
            gen=root+f'generate-{i}/';score=root+f'score-{i}/'
            records=read(gen+'selections.json');outcomes=read(score+'outcomes.json')
            require(set(outcomes)==set(records),'scoring roster differs from generation')
            expected={split:SPEC[split] for split in SPLITS}
            require(dict(Counter(r['strata']['split'] for r in records.values()))==expected,'changed full split denominators')
            g=read(gen+'summary.json');s=read(score+'summary.json');b=read(gen+'boundary.json')
            require(g['source']==s['source']==source and g['work']==s['work']==imp['work'],'role source/work binding changed')
            require(g['counts']==parameters(PROFILES['m100']) and g['execution']==EXECUTION,'changed actual model/numerics')
            require(b['active']==b['reserved']==0,'native boundary not drained')
            for role in (g,s):require(role['resources']['qualified'] and role['resources']['breach'] is None,'resource qualification failed')
            for kind in ('generate','score'):require(read(root+f'{kind}-{i}/cleanup.json')['within_bound'],'unacknowledged role cleanup')
            c=read(gen+'portfolio-clocks.json');a=read(gen+'attempt-index.json')
            audit=clocks_audit(c,a,records)
            observation=cost['observations'][i]
            require(observation['clock_receipt']==c['identity'] and observation['complete'] and observation['clock_complete'],'cost/clock mismatch')
            pointer=imp['snapshots'][i]['pointer']
            require(observation['checkpoint']==g['checkpoint']==pointer,'different original checkpoint evaluated')
            sums=defaultdict(Counter);reasons=defaultdict(Counter)
            for prompt,arms in outcomes.items():
                require(set(arms)==set(ARMS),'missing scientific arm')
                for arm,r in arms.items():
                    require(not r['success'] or r['complete'],'success without complete scoring')
                    if r['success']:
                        for method in ('production','reference'):
                            require(r['scoring'][method]['outcome']=='ok' and r['scoring'][method]['coverage']==100,'success lacks exact100 coverage')
                    key=records[prompt]['strata']['split']+':'+arm
                    sums[key]['denominator']+=1;sums[key]['complete']+=r['complete'];sums[key]['success']+=r['success']
                    if not r['complete']:reasons[key][r['reason']]+=1
            metrics={}
            for key,row in sums.items():
                n=row['denominator'];unknown=n-row['complete']
                metrics[key]=dict(row,unresolved=unknown,coverage=row['complete']/n,lower=row['success']/n,upper=(row['success']+unknown)/n,
                    interpretation='fixed-checkpoint incomplete-measurement bounds, not future-learning bounds')
            snapshots.append(dict(audit,ordinal=i,checkpoint=pointer,generation_seconds=observation['generation_seconds'],
                scoring_seconds=observation['scoring_seconds'],metrics=metrics,unresolved_reasons=dict(reasons),
                generation_resources=g['resources'],scoring_resources=s['resources']))
        require(len({s['snapshot'] for s in snapshots})==3,'three distinct saved models required')
        return dict(version='pipeline-pilot-calibration-review/v1',archive_sha256=file_hash(path),source=source,
            inventory=dict(total_members=len(manifest['members'])+1,omissions=manifest['omissions'],hash_mismatches=[]),
            elapsed_seconds=summary['elapsed_seconds'],stages={k:r['state'] for k,r in summary['stages'].items()},
            cost=cost,snapshots=snapshots,primary_plumbing=summary['engineering'],portability=summary['portability'],
            next='Review versioned full-clock long profile and authority migration; no repeat calibration needed.',
            limitations=['No new training in calibration.','Scientific incomplete outcomes are not converted to negatives.',
                'Structural development is A-stage only; parameter development has no C-stage rows.',
                'Fitted8-attempt equal-budget differs from1-attempt shuffled/absent; no sole conditioning attribution.',
                'Costs reflect early checkpoints; later execution retains finite caps.'],long_run_admitted=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    result=review(args.archive);args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(result['primary_plumbing'],result['cost']['status'],result['cost']['projected_seconds'])


if __name__=='__main__':main()
