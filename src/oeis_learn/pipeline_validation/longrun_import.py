"""Explicit compatible-source import; retains producer identities and paid costs."""
import ast
from collections import Counter
import json
from pathlib import Path
from .identity import confined,digest,file_hash,seal,validate
from .pilot_clocks import ATTEMPTS,POLICY,clock_record

CALIBRATION_SOURCE='sha256:3b08430642a0965c4843fac9086c3655d69d0afcc739d82c08c1e860a21aace5'
# Reviewed control-plane migration only. All model/decoder/native/data/build/lock
# bytes remain identical. Unknown additions/changes invalidate reuse.
MIGRATION_PATHS={
 'scripts/foundation_local.py',
 'src/oeis_learn/fresh/pilot_control.py',
 *('src/oeis_learn/pipeline_validation/'+n+'.py' for n in (
     'pilot','pilot_runtime','pilot_study','pilot_ledger','cli','adapters',
     'longrun_contract','longrun_import','longrun_admission')),
 'src/oeis_learn/container_validation.py',
 'tests/unit/test_pipeline_pilot_longrun.py','tests/unit/test_pipeline_pilot_runtime.py',
 'specs/009-pipeline-throughput/contracts/long-run-pilot.md',
}


def changes(old,current):
    changed={p:dict(producer=old.get(p),current=current.get(p)) for p in sorted(set(old)|set(current)) if old.get(p)!=current.get(p)}
    if set(changed)-MIGRATION_PATHS:raise ValueError('unmapped changed calibration dependency: '+', '.join(sorted(set(changed)-MIGRATION_PATHS)))
    if any(current.get(p) is None for p in old):raise ValueError('removed calibration dependency')
    return changed


def validate_compatibility(record,*,source):
    validate(record,{'version','source','calibration_source','producer_source','work','engine','cpus','resources',
        'calibration_manifest','producer_manifest','old_snapshot','current_snapshot','changes','environment','authority'},'pipeline-pilot-compatibility/v1')
    if (record['source']!=source or digest(record['current_snapshot'])!=source
        or record['calibration_source']!=CALIBRATION_SOURCE or digest(record['old_snapshot'])!=CALIBRATION_SOURCE
        or record['changes']!=changes(record['old_snapshot'],record['current_snapshot'])
        or record['authority']!='retained data/native/cost only; fresh learner; current qualification required'):
        raise ValueError('compatible source proof changed')
    from .pilot_calibration_import import PRODUCER_SOURCE
    from oeis_learn.synthesis.state import require_hash
    if record['producer_source']!=PRODUCER_SOURCE:raise ValueError('known corpus producer required')
    if record['engine'] not in ('podman','docker') or record['cpus']!=16:raise ValueError('original calibrated engine/CPU required')
    for key in ('work','resources','calibration_manifest','producer_manifest'):require_hash(record[key])
    if set(record['environment'])!={'engine','hardware','gpu','cpu','artifacts'}:raise ValueError('closed runtime environment required')
    return record


def clocks_audit(clocks,attempts,records):
    expected={(p,mode,i) for p in records for mode,n in ATTEMPTS.items() for i in range(n)}
    jobs={(j['prompt'],j['mode'],j['attempt']):j['state'] for j in attempts['jobs']}
    if (seal(clocks)!=clocks or seal(attempts)!=attempts or len(records)!=296 or set(jobs)!=expected
        or len(attempts['jobs'])!=5328 or attempts['expected']!=5328 or dict(Counter(jobs.values()))!=attempts['states']
        or set(jobs.values())-{'COMMITTED','INCOMPLETE','UNSTARTED'} or clocks['attempts']!=attempts['identity']
        or clocks['policy']!=POLICY or len(clocks['clocks'])!=1184 or clocks['missing_portfolios']!=0
        or clocks['expected_portfolios']!=1184 or not clocks['cost_complete']):raise ValueError('complete clock/attempt roster required')
    seen=set();snapshots=set()
    for row in clocks['clocks']:
        key=(row['prompt'],row['mode'])
        if key in seen or (*key,0) not in jobs:raise ValueError('duplicate/foreign portfolio')
        seen.add(key);snapshots.add(row['snapshot'])
        check=clock_record(**{k:row[k] for k in ('prompt','mode','snapshot','start','scope_deadline','finish','expected','started','terminal')})
        states=[jobs[(*key,i)] for i in range(ATTEMPTS[key[1]])]
        if (row!=check or not row['cost_complete'] or row['started']!=sum(s!='UNSTARTED' for s in states)
            or row['terminal']!=sum(s in ('COMMITTED','INCOMPLETE') for s in states)):
            raise ValueError('cut/altered clock differs from attempt ledger')
    if len(snapshots)!=1 or dict(Counter(r['termination'] for r in clocks['clocks']))!=clocks['terminations']:
        raise ValueError('snapshot/termination binding changed')
    return next(iter(snapshots))


def environment(root,engine):
    root=Path(root)
    read=lambda n:json.loads(confined(root,'stages/'+n).read_text())
    def text(n):
        p=confined(root,'stages/'+n)
        if p.stat().st_size>65536:raise ValueError('bounded environment observation required')
        return ast.literal_eval(p.read_text())
    # Native build manifest's producer hash changes with the package; the actual
    # binaries/guest/toolchain/locks/policy must still be byte-identical.
    artifacts={}
    for target in ('gpu','qualification'):
        a=read('image.'+engine+'.'+target+'/inspection/logs/stdout.log')['artifacts']
        artifacts[target]={k:v for k,v in a.items() if k!='build_sha256'}
    return dict(engine=read('engine.'+engine+'/engine.json')['facts'],
        hardware=read('cpu.'+engine+'.semantic/receipt.json')['binding']['hardware'],
        gpu=text('hip.preflight/environment/stdout.log'),cpu=text('cpu.'+engine+'.semantic/environment/stdout.log'),artifacts=artifacts)


def inspect(calibration,producer,repo,*,engine):
    from .reuse import validate_import
    from .pilot_calibration_import import inspect as inspect_producer
    from .pilot_calibration_cost import forecast
    from .model_sizes import EXECUTION,PROFILES,parameters
    from .pilot_contract import SPEC,SPLITS
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    calibration=Path(calibration);producer=Path(producer);validate_import(calibration)
    read=lambda name:json.loads(confined(calibration,name).read_text())
    inv=read('invocation.json');old=inv['binding']['dependency_closure']['files'];current=collect_snapshot(repo)
    if inv['binding']['source']!=CALIBRATION_SOURCE or digest(old)!=CALIBRATION_SOURCE:raise ValueError('accepted calibration source required')
    migration=changes(old,current)
    summary=read('summary.json')
    from .pilot_calibration import registry
    required={s.id for s in registry([engine],engine)}
    if any(summary['stages'].get(k,{}).get('state')!='PASS' for k in required):raise ValueError('complete primary calibration prerequisites required')
    data=inspect_producer(producer,repo,engine=engine)
    imported=read('stages/calibration.import/import.json')
    if imported!=data['public']:raise ValueError('calibration corpus/snapshots differ from actual producer')
    cost=read('stages/calibration.report/candidate-cost.json');work=data['manifest']['identity']
    if cost!=forecast(cost['observations'],source=CALIBRATION_SOURCE,work=work,archives=cost['archives']) or not cost['within_budget']:
        raise ValueError('complete full-clock cost required')
    models=set()
    for i in range(3):
        prefix='stages/calibration.session/'
        gen=prefix+'generate-'+str(i)+'/'
        score=prefix+'score-'+str(i)+'/'
        records=read(gen+'selections.json');clocks=read(gen+'portfolio-clocks.json');attempts=read(gen+'attempt-index.json')
        models.add(clocks_audit(clocks,attempts,records))
        if dict(Counter(r['strata']['split'] for r in records.values()))!={k:SPEC[k] for k in SPLITS}:raise ValueError('frozen full split roster required')
        g=read(gen+'summary.json');s=read(score+'summary.json');boundary=read(gen+'boundary.json');row=cost['observations'][i]
        if (g['source']!=CALIBRATION_SOURCE or s['source']!=CALIBRATION_SOURCE or g['work']!=work or s['work']!=work
            or g['execution']!=EXECUTION or g['counts']!=parameters(PROFILES['m100']) or boundary['active'] or boundary['reserved']
            or row!=read(prefix+'observation-'+str(i)+'.json') or row['clock_receipt']!=clocks['identity']
            or row['checkpoint']!=g['checkpoint'] or g['checkpoint']!=data['snapshots'][i]['pointer']):raise ValueError('calibration actual model/work/clock mismatch')
        for role,path in ((g,gen),(s,score)):
            if role.get('fatal') or not role['resources']['qualified'] or role['resources']['breach'] is not None or read(path+'cleanup.json')['within_bound'] is not True:
                raise ValueError('calibration resources/cleanup failed')
    if len(models)!=3:raise ValueError('three distinct actual snapshots required')
    from .resources import execution_identity
    resources=execution_identity(read('stages/engine.'+engine+'/engine.json')['resources'])
    proof=seal(dict(version='pipeline-pilot-compatibility/v1',source=digest(current),calibration_source=CALIBRATION_SOURCE,
        producer_source=data['manifest']['source'],work=work,engine=engine,cpus=16,resources=resources,
        calibration_manifest=file_hash(calibration/'manifest.json'),producer_manifest=file_hash(producer/'manifest.json'),
        old_snapshot=old,current_snapshot=current,changes=migration,environment=environment(calibration,engine),
        authority='retained data/native/cost only; fresh learner; current qualification required'))
    validate_compatibility(proof,source=digest(current))
    return dict(prepared=data['prepared'],manifest=data['manifest'],compatibility=proof,candidate=cost)
