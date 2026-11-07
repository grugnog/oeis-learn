"""CPU-only adapters around the unchanged native client; no learner or full-horizon scorer."""
import copy
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .identity import seal,write,digest
from .generalization_preparation import reference
from oeis_learn.generalization.contracts import verify_artifact
from oeis_learn.generalization.selector_fit_replay import extract,validate_packet
from oeis_learn.synthesis.admission import VisiblePrompt


class ObservedWorker:
    """Observe the existing transport without changing its requests or responses."""
    def __init__(self,worker):self.worker=worker;self.local=threading.local()
    def __getattr__(self,name):return getattr(self.worker,name)
    def call(self,request,*,deadline):
        slot=getattr(getattr(self.worker,'local',None),'prompt',0)%len(getattr(self.worker,'slots',[None]))
        before=copy.deepcopy(self.worker.stats)
        previous=(before.get('slots') or [before])[slot].get('queue_wait_seconds')
        began=time.monotonic()
        response=self.worker.call(request,deadline=deadline)
        if getattr(self.local,'calls',None) is not None:
            metrics=copy.deepcopy(response.get('metrics') or {})
            current=(metrics.get('slots') or [metrics])[slot if metrics.get('slots') else 0].get('queue_wait_seconds')
            self.local.calls.append(dict(operation=request['operation'],wall_seconds=time.monotonic()-began,
                allowance_seconds=max(0,deadline-began),complete=response.get('complete'),
                queue_wait_seconds=max(0,current-previous) if current is not None and previous is not None else None,
                outcome=response.get('outcome'),reason=response.get('reason'),metrics=metrics))
        return response


def resource_snapshot():
    """Optional cgroup-v2 counters; unavailability is explicit, never zero usage."""
    result={}
    for name in ('cpu.stat','cpu.max','memory.current','memory.peak','memory.max'):
        try:
            value=(Path('/sys/fs/cgroup')/name).read_text().strip()
            result[name]=dict(line.split() for line in value.splitlines()) if name=='cpu.stat' else value
        except (OSError,ValueError):result[name]=None
    return result


def prepare(inputs,output,settings,*,seconds):
    from oeis_learn.generalization.selector_validation import probe_views
    if set(settings)!={'generation','probes','inventory','targets'}:raise ValueError('closed fit preparation request')
    root=inputs['mechanical_sources']
    def load(ref):return json.loads(verify_artifact(root,ref,max_bytes=512<<20))
    prompts,_,_=probe_views(load(settings['probes']),load(settings['inventory']))
    packet=extract(load(settings['generation']),prompts,settings['targets'])
    # Candidate and visible-prompt channels remain physically disjoint.
    write(output/'visible/prompts.json',packet['prompts'])
    candidates=seal(dict(version=packet['version'],attempts=packet['attempts']))
    write(output/'candidates/attempts.json',candidates)
    return dict(status='PASS',packet=packet['identity'],cases=len(packet['prompts']),attempts=len(packet['attempts']),
        prompts=reference(output/'visible/prompts.json',output),candidates=reference(output/'candidates/attempts.json',output))


def replay(packet,recipe,output,*,concurrency,repetitions,client_factory=None):
    from oeis_learn.fresh.bigint_worker import client
    from oeis_learn.decoder.bigint_codec import fitting_policy
    if concurrency not in (1,4) or type(repetitions) is not int or not 1<=repetitions<=5:
        raise ValueError('bounded fixed replay matrix required')
    validate_packet(packet)
    prompts={p['id']:VisiblePrompt(tuple(p['indices']),tuple(p['values'])) for p in packet['prompts']}
    runs=[];output=Path(output)
    for repeat in range(repetitions):
        factory=client_factory or client
        with factory(dict(recipe=recipe,contract_sha256=packet['identity']),output/('repeat-'+str(repeat))) as native:
            native.worker=ObservedWorker(native.worker)
            runtime=native.runtime_identity('bigint-body/v1');reference_id=native.reference_sha256
            warm=[]
            # Warm each existing worker before measurements; cache roots are fresh
            # per repetition/configuration. Warmup is recorded separately.
            for slot in range(4):
                began=time.monotonic()
                with native.worker.affinity(slot):
                    value=native.execute('(program (return (int 0)))',list(map(str,range(20))),deadline=began+4)
                warm.append(dict(slot=slot,seconds=time.monotonic()-began,outcome=value.get('outcome')))
                if value.get('outcome')!='ok':raise ValueError('native warmup incomplete')
            def fit_one(index):
                row=packet['attempts'][index];prompt=prompts[row['case']]
                if len(prompt.indices)!=20 or len(prompt.values)!=20 or prompt.sha256!=row['prompt_sha256']:
                    raise ValueError('visible-only replay prompt mismatch')
                expected=row.get('original_resource')
                if expected is not None and expected!=native.limits.identity:raise ValueError('original fitting resources changed')
                native.worker.local.calls=[];began=time.monotonic()
                with native.worker.affinity(index%4):
                    result=native.fit(row['source'],prompt,policy=fitting_policy(row['source']),
                        seed=row['seed'],deadline=began+row['service_seconds'])
                elapsed=time.monotonic()-began;n=result.get('native') or {}
                return dict(case=row['case'],position=row['position'],seed=row['seed'],source_sha256=row['source_sha256'],
                    original_outcome=row['original_outcome'],service_seconds=row['service_seconds'],wall_seconds=elapsed,
                    complete=result.get('complete',False),outcome=result['outcome'],reason=result.get('reason'),
                    admitted=[r['source_sha256'] for r in result.get('admitted',[])],
                    native={k:n.get(k) for k in ('outcome','reason','phase','elapsed_seconds','timing','searches','completed_bounds')},
                    transport=native.worker.local.calls)
            resources_before=resource_snapshot();began=time.monotonic();rows=[]
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                # Stable waves assign each concurrent fit a distinct existing slot.
                for offset in range(0,len(packet['attempts']),concurrency):
                    rows.extend(executor.map(fit_one,range(offset,min(offset+concurrency,len(packet['attempts'])))))
            wall=time.monotonic()-began
            resources_after=resource_snapshot()
            cases={k:dict(fit_search_complete=all(r['complete'] for r in rows if r['case']==k),
                admitted_candidates=sum(len(r['admitted']) for r in rows if r['case']==k)) for k in prompts}
            run=dict(repeat=repeat,wall_seconds=wall,attempts=rows,cases=cases,warmup=warm,
                     worker_stats=copy.deepcopy(native.worker.stats),runtime=runtime,reference=reference_id,
                     resources_before=resources_before,resources_after=resources_after)
            write(output/('repeat-'+str(repeat)+'.json'),seal(run));runs.append(run)
    result=seal(dict(version='pac-fit-replay-result/v1',status='PASS',scientific_status='DIAGNOSTIC_ONLY',
        freeze_eligible=False,packet=packet['identity'],concurrency=concurrency,repetitions=repetitions,runs=runs,
        assessment='NOT_EXECUTED',training_tokens=0,scope='fit-only replay; no GPU generation, scoring, or replacement of original outcomes'))
    write(output/'replay.json',result);return result


def run(inputs,output,settings,*,seconds):
    if set(settings)!={'prompts','candidates','recipe','concurrency','repetitions'}:
        raise ValueError('closed native fit diagnostic request')
    prompts=json.loads(verify_artifact(inputs['visible_prompts'],settings['prompts'],max_bytes=1<<20))
    candidates=json.loads(verify_artifact(inputs['generated_candidates'],settings['candidates'],max_bytes=32<<20))
    if seal(candidates)!=candidates:raise ValueError('candidate packet changed')
    packet=seal(dict(version=candidates['version'],prompts=prompts,attempts=candidates['attempts']))
    return replay(packet,settings['recipe'],output,concurrency=settings['concurrency'],repetitions=settings['repetitions'])
