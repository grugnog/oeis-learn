"""Internal isolated engineering adapters; no study/training execution authority."""
import argparse
from dataclasses import asdict,replace
import json
import os
from pathlib import Path
import statistics
import time

from .identity import digest,seal,write,number


def native_corpus():
    """Independent exact values. Large intermediates cancel to bounded zero outputs."""
    from oeis_learn.synthesis.contracts import decimal_integer
    rows=[]
    for bits in (64,256,512,1024,4096):
        indices=[decimal_integer((1<<(bits-1))+i) for i in range(20)]
        rows.append(dict(id=f'identity-{bits}',stratum=f'bits{bits}',
            source='(program (return (param.get p0)))',indices=[str(i) for i in range(20)],
            values=[indices[0]]*20))
        for degree in (2,4):
            term='$index'
            for _ in range(degree.bit_length()-1):term=f'(mul {term} {term})'
            rows.append(dict(id=f'cancel-{bits}-{degree}',stratum=f'bits{bits}-degree{degree}',
                source=f'(program (return (sub {term} {term})))',indices=indices,values=['0']*20))
    rows.extend([
        dict(id='affine',stratum='small-affine',source='(program (return (add (mul (param.get p0) $index) (param.get p1))))',indices=[str(i) for i in range(20)],values=[str(3*i+7) for i in range(20)]),
        dict(id='nonlinear',stratum='nonlinear',source='(program (return (mul (param.get p0) (param.get p1))))',indices=[str(i) for i in range(20)],values=['15']*20),
        dict(id='no-fit',stratum='negative',source='(program (return (param.get p0)))',indices=[str(i) for i in range(20)],values=[str(i) for i in range(20)]),
    ])
    return seal(dict(version='pipeline-native-corpus/v1',rows=rows,exclusions=[]))


def native_trial(output,*,workers,corpus,contract,cap,admission="queued8/v1",on_client=None):
    if admission not in {"queued8/v1","slot-waves/v1"} or type(workers) is not int or workers not in (1,2,4):
        raise ValueError("registered native admission/worker policy required")
    from oeis_learn.synthesis.client import SynthesisClient
    from oeis_learn.synthesis.service_policy import ServiceLimits
    from oeis_learn.synthesis.admission import VisiblePrompt
    from oeis_learn.fresh.pipeline_jobs import NativeRequest,logical_job
    from oeis_learn.fresh.pipeline_scheduler import drive
    from oeis_learn.fresh.pipeline_jobs import JobLedger
    ledger=JobLedger();committed={};timing={}
    output.mkdir(parents=True,exist_ok=True)
    limits=ServiceLimits(max_inflight_batches=workers)
    start=time.monotonic();deadline=start+cap
    from contextlib import nullcontext
    from .native_capacity import NativeEvidence
    evidence=NativeEvidence(output,corpus,ledger,timing) if admission=='slot-waves/v1' else nullcontext()
    with evidence, SynthesisClient(contract,service_limits=limits,cache_root=output/'private/native-cache') as client:
        if on_client is not None:on_client(client)
        def job(i,row):
            logical=logical_job(contract=contract,snapshot='fixed-native-corpus',mode='native',arm='fit',
                prompt_id=row['id'],attempt_ordinal=0,method='fixture',seed=20261016)
            admitted=time.monotonic();service_deadline=min(deadline,admitted+4)
            if admission=="slot-waves/v1":
                timing[row["id"]]=dict(admitted=admitted,deadline=service_deadline)
                evidence.event("ADMITTED",row["id"],job=logical,deadline=service_deadline,at=admitted)
            def operation():
                from contextlib import nullcontext
                affinity=getattr(client.worker,'affinity',None)
                with affinity(i) if affinity else nullcontext():
                    began=time.monotonic()
                    if admission=="slot-waves/v1":
                        timing[row["id"]]["dispatched"]=began
                        evidence.event("DISPATCHED",row["id"],at=began)
                    result=client.fit(row['source'],VisiblePrompt(tuple(row['indices']),tuple(row['values'])),
                        deadline=min(service_deadline,time.monotonic()+4))
                    if admission=="slot-waves/v1":
                        timing[row["id"]]["native_completed"]=time.monotonic()
                        evidence.event("NATIVE_COMPLETED",row["id"],at=timing[row["id"]]["native_completed"])
                    return dict(id=row['id'],stratum=row['stratum'],seconds=time.monotonic()-began,
                        complete=result.get('complete',False),outcome=result['outcome'],
                        admitted=[dict(source_sha256=a['source_sha256'],assignment=a['assignment'],verification=a['verification']) for a in result['admitted']])
            value=yield NativeRequest(logical,i,service_deadline,operation,cancel=client.worker.cancel_all)
            if admission=='slot-waves/v1':evidence.event('OWNER_OBSERVED',row['id'],record=ledger.records[logical['job_id']])
            committed[row['id']]=value
            write(output/'jobs'/(row['id']+'.json'),value)
            return value
        policies={'selection':'visible-only/v2','corpus':corpus['identity']}
        failure=None
        try:
            if admission=='queued8/v1':
                results,boundary=drive(None,[job(i,row) for i,row in enumerate(corpus['rows'])],workers=workers,
                    policies=policies,ledger=ledger)
            else:
                policies=dict(policies,admission=admission)
                results=[]
                for offset in range(0,len(corpus['rows']),workers):
                    if time.monotonic()>=deadline:raise TimeoutError('native parent deadline before wave admission')
                    group=[job(i,corpus['rows'][i]) for i in range(offset,min(offset+workers,len(corpus['rows'])))]
                    values,boundary=drive(None,group,workers=workers,policies=policies,ledger=ledger)
                    results.extend(values)
        except TimeoutError as exc:
            failure=str(exc)
            for key in ledger.records:ledger.cancel(key)
            boundary=ledger.boundary(policies)
            results=[committed.get(row['id'],dict(id=row['id'],stratum=row['stratum'],complete=False,
                outcome='unstarted_or_cancelled',admitted=[],seconds=None)) for row in corpus['rows']]
        result=seal(dict(version='pipeline-native-trial/v1',workers=workers,corpus=corpus['identity'],
            seconds=time.monotonic()-start,results=results,complete=all(r['complete'] for r in results),
            boundary=boundary,metrics=client.worker.stats,failure=failure))
    if admission=='slot-waves/v1':
        result=seal(dict(result,version='pipeline-native-wave-trial/v1',admission=admission,
            jobs=ledger.records,timing=timing,started=start,parent_deadline=deadline,metrics=client.worker.stats))
    write(output/'trial.json',result)
    if admission=='slot-waves/v1':
        result=seal(dict(result,primary_seconds=time.monotonic()-start))
        write(output/'measurement.json',result)
    return result


def native_compare(inputs,output,config):
    corpus=native_corpus();write(output/'workload.json',corpus)
    from .comparisons import calibrate
    cap=config['options'].get('seconds',3000)
    began=time.monotonic()
    cold=native_trial(output/'cold',workers=1,corpus=corpus,contract=config['contract'],cap=min(180,cap/10))
    complete={r['id'] for r in cold['results'] if r['complete']}
    subset=seal(dict(version='pipeline-native-corpus/v1',rows=[r for r in corpus['rows'] if r['id'] in complete],
        exclusions=[r['id'] for r in corpus['rows'] if r['id'] not in complete]))
    write(output/'complete-subset.json',subset)
    if not complete:
        return dict(status='INCONCLUSIVE',selected_workers=1,reason='cold baseline has no complete work; candidates unstarted')
    native_trial(output/'warmup',workers=1,corpus=subset,contract=config['contract'],cap=min(180,cap/10))
    calibrations=[]
    for ordinal in range(3):
        trial=native_trial(output/f'calibration-{ordinal}',workers=1,corpus=subset,contract=config['contract'],cap=min(180,cap/10))
        calibrations.append(dict(complete=trial['complete'],seconds=trial['seconds'],identity=trial['identity']))
    try:
        calibration=calibrate(calibrations,minimum=30,cap=300,parent=config['contract'],workload=subset['identity'])
    except ValueError as exc:
        return dict(status='INCONCLUSIVE',selected_workers=1,reason=str(exc),candidate_runs=0)
    reservation=calibration['reservation_seconds']
    write(output/'comparison-contract.json',seal(dict(version='pipeline-native-comparison-contract/v1',
        calibration=calibration,subset=subset['identity'],candidates=[2,4],orders=['AB','BA'],
        selection='fastest eligible median; fewer workers within five percent',remaining_reservation=10*reservation)))
    if 10*reservation>cap-(time.monotonic()-began):
        return dict(status='INCONCLUSIVE',selected_workers=1,reason='finite native candidate reservations cannot fit',candidate_runs=0)
    records=[]
    for workers in (2,4):
        native_trial(output/f'w{workers}-warmup',workers=workers,corpus=subset,contract=config['contract'],cap=reservation)
        for pair,order in enumerate(('AB','BA')):
            row={}
            for arm in order:
                count=1 if arm=='A' else workers
                row[arm]=native_trial(output/f'w{workers}-pair{pair}-{arm}',workers=count,
                    corpus=subset,contract=config['contract'],cap=reservation)
            parity=[r['admitted'] for r in row['A']['results']]==[r['admitted'] for r in row['B']['results']]
            records.append(dict(workers=workers,pair=pair,order=order,arms=row,parity=parity))
    eligible={}
    for workers in (2,4):
        pairs=[r for r in records if r['workers']==workers]
        if all(r['parity'] and all(a['complete'] for a in r['arms'].values()) for r in pairs):
            eligible[workers]=statistics.median(r['arms']['B']['seconds'] for r in pairs)
    if eligible:
        eligible[1]=statistics.median(r['arms']['A']['seconds'] for r in records if all(a['complete'] for a in r['arms'].values()))
        fastest=min(eligible.values())
        selected=min(w for w,t in eligible.items() if t<=fastest*1.05)
    else:
        selected=1
    report=seal(dict(version='pipeline-native-comparison/v1',records=records,selected_workers=selected,
        status='COMPLETE' if eligible else 'INCONCLUSIVE',adoption='EXPERIMENTAL_ONLY',
        reason='joint end-to-end confirmation required; no worker expansion'))
    write(output/'comparison.json',report)
    return report


def cpu_contracts(inputs,output,config):
    import subprocess
    import sys
    command=[sys.executable,'-m','pytest','-q','tests/unit/test_pipeline_policy.py',
        'tests/unit/test_pipeline_native_pool.py','tests/unit/test_pipeline_overlap.py',
        'tests/integration/test_bigint_decoding.py','--junitxml='+str(output/'results.xml')]
    subprocess.run(command,check=True,timeout=600)
    from .engines import junit
    from .policies import fixture_report
    comparison=fixture_report()
    write(output/'policy-comparison.json',comparison)
    return dict(status='COMPLETE',counts=junit(output/'results.xml'),comparison=comparison)


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['native','contracts','neural','score','replay-reference','replay','replay-compare','replay-batching-reference','replay-batching','replay-batching-compare','hardware-reference','hardware-replay','native-capacity','coschedule-reference','coschedule-gpu','coschedule-study','sizing-reference','sizing-trial','soak-reference','soak-trial']);p.add_argument('--input',default='/work/input');p.add_argument('--output',default='/work/output')
    args=p.parse_args(argv);inputs=Path(args.input);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    result={'status':'FAIL'}
    role_started=time.monotonic()
    try:
        config=json.loads((inputs/'expected.json').read_text())
        if set(config)!={'version','source','contract','mode','options'} or config['version']!='pipeline-role/v1' or config['mode']!=args.mode:
            raise ValueError('unknown internal role contract')
        from oeis_learn.experiments.source_snapshot import source_snapshot
        if digest(source_snapshot())!=config['source']:
            raise ValueError('current installed source mismatch')
        from oeis_learn.tracking.pipeline_profile import device_view,audit_device_view
        view=device_view();write(output/'devices.json',view)
        try:
            audit_device_view([],view,json.loads(os.environ['OEIS_EXPECTED_PIPELINE_DEVICES']))
        except (ValueError,RuntimeError) as exc:
            from .runner import ResourceBreach
            raise ResourceBreach('role device isolation audit failed: '+str(exc)) from exc
        from .neural import gpu,score
        from .replay import reference_role,replay_role
        from .replay_comparison import comparison_role
        from .replay_batching import reference_role as batching_reference,batch_role,comparison_role as batching_compare
        from .hardware_ladder import reference_role as hardware_reference,hardware_role
        from .native_capacity import investigate
        from .coschedule import reference_role as coschedule_reference,gpu_role as coschedule_gpu,study as coschedule_study
        from .model_sizing import reference_role as sizing_reference,gpu_role as sizing_trial
        from .soak_runtime import reference as soak_reference,trial as soak_trial
        from .memory import MemoryWatchdog
        ceiling=config['options'].get('memory_ceiling')
        if ceiling is None:
            from .resources import observe,resolve
            ceiling=resolve(**observe())['ceiling_bytes']
        with MemoryWatchdog(output,ceiling=ceiling,gpu=args.mode in {'neural','replay','replay-compare','replay-batching','replay-batching-compare','hardware-replay','coschedule-gpu','coschedule-study','sizing-trial','soak-trial'},
                render_node=config['options'].get('render_node'),audit_samples=args.mode.startswith('soak-')) as guard:
            if args.mode in {'native-capacity','coschedule-reference','coschedule-gpu','coschedule-study','sizing-reference','sizing-trial','soak-reference','soak-trial'} and not guard.report()['qualified']:
                raise RuntimeError('known aggregate memory accounting required before '+args.mode)
            result={'native':native_compare,'contracts':cpu_contracts,'neural':gpu,'score':score,'replay-reference':reference_role,'replay':replay_role,'replay-compare':comparison_role,'replay-batching-reference':batching_reference,'replay-batching':batch_role,'replay-batching-compare':batching_compare,'hardware-reference':hardware_reference,'hardware-replay':hardware_role,'native-capacity':investigate,'coschedule-reference':coschedule_reference,'coschedule-gpu':coschedule_gpu,'coschedule-study':coschedule_study,'sizing-reference':sizing_reference,'sizing-trial':sizing_trial,'soak-reference':soak_reference,'soak-trial':soak_trial}[args.mode](inputs,output,config)
        result['resources']=guard.report()
    except BaseException as exc:
        from .runner import ResourceBreach
        from oeis_learn.synthesis.worker_pool import NonQuiescentWorker
        result=dict(status='FAIL',error=f'{type(exc).__name__}: {exc}',
            fatal='RESOURCE_BREACH' if isinstance(exc,(MemoryError,ResourceBreach,NonQuiescentWorker)) else None)
    finally:
        result['seconds_through_role']=time.monotonic()-role_started
        result.setdefault('version','pipeline-role-result/v1')
        result=seal(result)
        write(output/'summary.json',result)
    return 1 if result['status']=='FAIL' else 0


if __name__=='__main__':
    raise SystemExit(main())
