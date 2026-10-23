"""Shared preparation and finite neural trials, with separate isolated CPU scoring."""
from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import time
import zlib

from .identity import seal,write,digest,file_hash
from .comparisons import calibrate,ORDER,decide
from .runner import CapabilityError

BASE_POLICY=dict(version='pipeline-policy/v1',selection='visible-only/v2',archive='structure-pareto/v2',
    workers=1,overlap=False,lookahead=1,batch_size=8,bookkeeping='baseline',measurement='ordinary')


def legacy(root):
    spec=importlib.util.spec_from_file_location('pipeline_profile_adapter',Path(root)/'scripts/profile_bigint_pipeline.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Benchmarks:
    def __init__(self,adapters):
        self.adapters=adapters;self.prepared=None;self.calibration=None;self.subset=None
        self.inference_eligible=False;self.selected_workers=1;self.prewarm=[];self.baseline_peak=0

    def prepare(self,output):
        if self.prepared is not None:return self.prepared
        a=self.adapters
        if not a.args.run or not a.args.inputs:
            raise CapabilityError('neural diagnostics require a completed --run and compatible --inputs; CPU checks remain independent')
        host=a.host(a.args.engine);host.select_gpu()
        module=legacy(a.root)
        root=output/'prepared';root.mkdir()
        from oeis_learn.experiments.source_snapshot import collect_snapshot
        manifest=module.prepare(a.root/a.args.run,a.root/a.args.inputs,root,
            image=a.images.images[(a.args.engine,'gpu')],source=digest(collect_snapshot(a.root)),render_node=host.policy['render_node'])
        self.prepared=(root,manifest)
        return self.prepared

    def trial(self,output,*,policy,assessment='fixed-work',bookkeeping='baseline',prompt_ids=None,seconds=600,telemetry=False,cache_state='cold'):
        a=self.adapters;prepared,manifest=self.prepare(output.parent)
        host=a.host(a.args.engine)
        from .resources import observe,resolve,reserve
        observations=observe();observations['limits'].append(host.policy['memory_bytes'])
        try:
            resource=resolve(**observations,profile=(policy or {}).get('memory_profile',a.contract['resource_policy']))
            model=manifest.get('model',{})
            weight_bytes=(prepared/'input/weights.pt').stat().st_size
            peak=max(self.baseline_peak,12*weight_bytes)+((policy or {}).get('workers',1)+2)*(1<<30)
            peak+=(5<<30) if (policy or {}).get('cache_capacity')=='expanded/v1' else (640<<20)
            reservation=reserve(resource,peak)
        except ValueError as exc:
            raise CapabilityError(str(exc)) from exc
        output.mkdir(parents=True)
        write(output/'resource-reservation.json',dict(policy=resource,reservation=reservation))
        inputs=output/'input';inputs.mkdir()
        os.link(prepared/'input/weights.pt',inputs/'weights.pt')
        work=json.loads((prepared/'input/work.json').read_text());write(inputs/'work.json',work)
        options=dict(policy=policy,assessment=assessment,bookkeeping=bookkeeping,prompt_ids=prompt_ids,
                     memory_ceiling=resource['ceiling_bytes'],render_node=host.policy['render_node'],
                     seconds=seconds,telemetry=telemetry,cache_state=cache_state,prewarm=self.prewarm if cache_state=='warm' else [])
        config=dict(version='pipeline-role/v1',source=manifest['source_sha256'],contract=a.contract['identity'],mode='neural',options=options)
        write(inputs/'expected.json',config)
        host=__import__('copy').copy(host);host.policy=dict(host.policy,memory_bytes=resource['ceiling_bytes'],cpu_memory_bytes=resource['ceiling_bytes'])
        image=a.images.images[(a.args.engine,'gpu')];module=legacy(a.root)
        started=time.monotonic()
        with host.gpu_lock():
            module.container_stage(host,image,inputs,output/'gpu',mode='neural',seconds=seconds,gpu=True,
                argv=['python','-m','oeis_learn.pipeline_validation.role','neural'])
        gpu=json.loads((output/'gpu/summary.json').read_text())
        if gpu['status']=='FAIL':raise RuntimeError(gpu.get('error','neural trial failed'))
        selections=json.loads((output/'gpu/private/selections.json').read_text())
        truth=json.loads((prepared/'private/truth.json').read_text())
        truth={key:truth[key] for key in selections}
        score_input=output/'private/scorer-input';score_input.mkdir(parents=True)
        write(score_input/'truth.json',truth);write(score_input/'selections.json',selections)
        write(score_input/'score.json',dict(truth=digest(truth),selections=digest(selections),resources=work['recipe']['execution']['resources']))
        write(score_input/'expected.json',dict(config,mode='score',options={'seconds':seconds,'memory_ceiling':host.policy['memory_bytes']}))
        module.container_stage(host,image,score_input,output/'scorer',mode='score',seconds=seconds,gpu=False,
            argv=['python','-m','oeis_learn.pipeline_validation.role','score'])
        scored=json.loads((output/'scorer/summary.json').read_text())
        if scored['status']=='FAIL':raise RuntimeError(scored.get('error','independent scorer failed'))
        # Include required compact export serialization, compression and disk write
        # in the repeated interval; no nested archive is retained in the bundle.
        export=output/'private/export.zlib'
        export.write_bytes(zlib.compress(json.dumps({'gpu':gpu,'score':scored},sort_keys=True).encode()))
        complete=gpu['status']=='COMPLETE' and scored['status']=='COMPLETE'
        result=seal(dict(version='pipeline-complete-trial/v1',complete=complete,
            seconds=time.monotonic()-started-gpu.get('prewarm_seconds',0),host_seconds=time.monotonic()-started,gpu=gpu,score=scored,options=options,
            work=digest(dict(prompts=gpu['prompt_ids'],attempts=gpu['scheduled_attempts'],updates=500,validation=256)),
            timing_interpretation='GPU and isolated independent scorer startup/work/cleanup plus compact diagnostic export; image build excluded'))
        write(output/'trial.json',result)
        return result

    def baseline(self,output,*,cap):
        # Freeze the complete subset from the one cold baseline before any candidate.
        cold=self.trial(output/'cold',policy=BASE_POLICY,seconds=min(600,cap/12))
        self.subset=cold['gpu']['complete_ids']
        self.baseline_peak=max(cold['gpu'].get('resources',{}).get('role_peak_upper_bytes',0),cold['score'].get('resources',{}).get('role_peak_upper_bytes',0))
        self.prewarm=json.loads((output/'cold/gpu/prewarm.json').read_text())
        if not self.subset:
            return dict(stage_state='INCONCLUSIVE',reason='no complete cold baseline subset; candidates unstarted')
        self.trial(output/'warmup',policy=BASE_POLICY,prompt_ids=self.subset,seconds=min(600,cap/12),cache_state='warm')
        rows=[]
        for i in range(3):
            row=self.trial(output/f'calibration-{i}',policy=BASE_POLICY,prompt_ids=self.subset,seconds=min(600,cap/12))
            rows.append(dict(complete=row['complete'],seconds=row['host_seconds'],identity=row['identity']))
        try:
            calibration=calibrate(rows,minimum=60,cap=1800,parent=self.adapters.contract['identity'],workload=digest(self.subset))
        except ValueError as exc:
            return dict(stage_state='INCONCLUSIVE',reason=str(exc),candidate_runs=0)
        self.calibration=calibration
        write(output/'calibration.json',seal(dict(calibration=calibration,subset=self.subset,original_denominator=32,
            exclusions=[p for p in cold['gpu']['prompt_ids'] if p not in self.subset])))
        return calibration

    def inference(self,output,*,cap):
        if self.calibration is None:raise CapabilityError('complete calibrated baseline required')
        reservation=self.calibration['reservation_seconds']
        if 5*reservation>cap:raise CapabilityError('frozen inference work cannot fit phase cap; no candidate started')
        frozen=seal(dict(version='pipeline-inference-comparison/v1',calibration=self.calibration['identity'],
            subset=self.subset,order=['AB','BA'],baseline='baseline',candidate='hoisted/v1',batch=8,selection='visible-only/v1'))
        write(output/'comparison-contract.json',frozen)
        self.trial(output/'warmup',policy=None,prompt_ids=self.subset,seconds=reservation/2)
        rows=[]
        for i,order in enumerate(['AB','BA']):
            arms={}
            for arm in order:
                arms[arm]=self.trial(output/f'pair{i}-{arm}',policy=None,prompt_ids=self.subset,
                    bookkeeping='baseline' if arm=='A' else 'hoisted/v1',seconds=reservation/2)
            parity=arms['A']['gpu']['tokens']==arms['B']['gpu']['tokens'] and arms['A']['gpu']['semantic']==arms['B']['gpu']['semantic']
            rows.append(dict(order=order,arms=arms,parity=parity))
        self.inference_eligible=all(r['parity'] and all(a['complete'] for a in r['arms'].values()) for r in rows)
        result=seal(dict(version='pipeline-inference-result/v1',rows=rows,
            eligible=self.inference_eligible,adoption='EXPERIMENTAL_ONLY' if self.inference_eligible else 'ROLLBACK',
            reason='isolated exploration; common-v2 joint confirmation required'))
        write(output/'comparison.json',result)
        if not self.inference_eligible:return dict(result,stage_state='INCONCLUSIVE')
        return result

    def joint(self,output,*,cap):
        if self.calibration is None:raise CapabilityError('complete calibrated baseline required')
        reservation=self.calibration['reservation_seconds']
        # Every branch is fixed before observations. Do not borrow another phase.
        candidates=[dict(BASE_POLICY,workers=self.selected_workers,overlap=True,lookahead=lookahead,
            bookkeeping='hoisted/v1' if self.inference_eligible else 'baseline') for lookahead in (1,2)]
        candidates += [dict(candidates[0],batch_size=16),dict(candidates[0],batch_size=1)]
        candidates.append(dict(candidates[0],cache_capacity='expanded/v1'))
        pressure=self.baseline_peak>=.8*(16<<30)
        if pressure:
            candidates.extend(dict(candidates[0],memory_profile=p) for p in ('historical16','capacity64','workstation90'))
        maximum_candidates=len(candidates)+1  # One conditional batch32 branch, reserved now.
        required=(2+5*maximum_candidates+4+20+20)*reservation
        if required>cap:raise CapabilityError(f'preregistered joint scope needs {required}s reservation, exceeds phase cap {cap}s; no exploratory/confirmation candidates started')
        contract=seal(dict(version='pipeline-joint-comparison/v1',calibration=self.calibration['identity'],
            candidates=candidates,subset=self.subset,confirmation_order=ORDER,cold_warm=['cold','warm'],
            thresholds=dict(gain=.10,stratum_ratio=1.05),selection_rule='fastest eligible median, smaller policy on <=5% tie',pressure_profiles='included' if pressure else 'not indicated by cold baseline peak',conditional_batch32='only positive batch16 exploration plus measured ten percent capacity; reservation retained'))
        write(output/'comparison-contract.json',contract)
        # Ordinary-vs-telemetry overhead is observational; primary runs stay ordinary.
        ordinary=self.trial(output/'telemetry-ordinary',policy=BASE_POLICY,prompt_ids=self.subset,seconds=reservation/2)
        telemetry=self.trial(output/'telemetry-observed',policy=BASE_POLICY,prompt_ids=self.subset,seconds=reservation/2,telemetry=True)
        write(output/'telemetry.json',dict(overhead=telemetry['seconds']/ordinary['seconds']-1,primary='ordinary'))
        exploratory=[]
        for n,candidate in enumerate(candidates):
            try:
                self.trial(output/f'candidate{n}-warmup',policy=candidate,prompt_ids=self.subset,
                    seconds=reservation/2,bookkeeping=candidate['bookkeeping'])
                pairs=[]
                for i,order in enumerate(['AB','BA']):
                    arms={}
                    for arm in order:
                        arms[arm]=self.trial(output/f'candidate{n}-pair{i}-{arm}',policy=BASE_POLICY if arm=='A' else candidate,prompt_ids=self.subset,seconds=reservation/2,bookkeeping=(BASE_POLICY if arm=='A' else candidate)['bookkeeping'])
                    pairs.append(arms)
            except CapabilityError as exc:
                exploratory.append(dict(candidate=candidate,eligible=False,state='UNAVAILABLE',reason=str(exc)))
                continue
            eligible=all(all(a['complete'] for a in r.values()) and r['A']['gpu']['semantic']==r['B']['gpu']['semantic'] for r in pairs)
            exploratory.append(dict(candidate=candidate,eligible=eligible,median=__import__('statistics').median(r['B']['seconds'] for r in pairs),pairs=pairs))
            if candidate['batch_size']==16 and eligible and all(r['B']['seconds']<r['A']['seconds'] for r in pairs):
                peak=max(r['B']['gpu'].get('resources',{}).get('role_peak_upper_bytes',0) for r in pairs)
                ceiling=min(r['B']['gpu'].get('resources',{}).get('ceiling_bytes',0) for r in pairs)
                predicted=2*peak+(4<<30)
                if peak>0 and predicted<=.9*ceiling:
                    proof=seal(dict(version='pipeline-capacity/v1',batch16_positive=True,
                        predicted_peak_bytes=predicted,ceiling_bytes=ceiling,
                        observations=[r['B']['identity'] for r in pairs]))
                    candidates.append(dict(candidate,batch_size=32,capacity_qualification=proof))
                    write(output/'batch32-admission.json',proof)
        eligible=[r for r in exploratory if r['eligible']]
        write(output/'exploratory.json',exploratory)
        if not eligible:return dict(stage_state='INCONCLUSIVE',adoption='NO_ADOPTION',reason='no complete equivalent exploratory candidate')
        fastest=min(r['median'] for r in eligible)
        selected=next(r['candidate'] for r in eligible if r['median']<=1.05*fastest)
        write(output/'selected-candidate.json',seal(dict(version='pipeline-selected/v1',parent=contract['identity'],policy=selected)))
        confirmations={}
        for scope in ('cold','warm'):
            for arm,policy in [('A',BASE_POLICY),('B',selected)]:
                self.trial(output/f'confirm-{scope}-warmup-{arm}',policy=policy,prompt_ids=self.subset,
                    seconds=reservation/2,bookkeeping=policy['bookkeeping'],cache_state=scope)
            pairs=[]
            for i,order in enumerate(ORDER):
                arms={};budget={}
                for arm in order:
                    policy=BASE_POLICY if arm=='A' else selected
                    arms[arm]=self.trial(output/f'confirm-{scope}-{i}-{arm}',policy=policy,
                        prompt_ids=self.subset,seconds=reservation/2,bookkeeping=policy['bookkeeping'],cache_state=scope)
                    budget[arm]=self.trial(output/f'budget-{scope}-{i}-{arm}',policy=policy,
                        assessment='fixed-budget',seconds=reservation/2,bookkeeping=policy['bookkeeping'],cache_state=scope)
                a,b=arms['A'],arms['B']
                fixed_a=budget['A']['score']['result']['splits']['equal_budget']
                fixed_b=budget['B']['score']['result']['splits']['equal_budget']
                strata={key:b['gpu']['timings'][key]/max(1e-9,a['gpu']['timings'][key]) for key in a['gpu']['timings']}
                pairs.append(dict(order=order,baseline=a['seconds'],candidate=b['seconds'],
                    complete=a['complete'] and b['complete'],work=a['work'],candidate_work=b['work'],
                    strata=strata,correct=a['gpu']['semantic']==b['gpu']['semantic'] and a['gpu']['tokens']==b['gpu']['tokens'],
                    resources=all(t['gpu'].get('resources',{}).get('qualified') is True and t['score'].get('resources',{}).get('qualified') is True for t in [a,b,*budget.values()]),baseline_success=fixed_a['success'],candidate_success=fixed_b['success'],
                    baseline_complete=fixed_a['complete'],candidate_complete=fixed_b['complete'],reset=True,scope=scope,kind='confirmation'))
            confirmations[scope]=decide(pairs,required_strata={'updates','validation','generation_native','checkpoint'})
            write(output/f'confirmation-{scope}.json',dict(pairs=pairs,decision=confirmations[scope]))
        result=seal(dict(version='pipeline-joint-result/v1',selected=selected,confirmations=confirmations,
            adoption='ADOPT' if all(r['adoption']=='ADOPT' for r in confirmations.values()) else 'NO_ADOPTION' if any(r['adoption']=='NO_ADOPTION' for r in confirmations.values()) else 'INCONCLUSIVE',
            default='serial baseline retained',coverage_scope=dict(subset=self.subset,original_denominator=32)))
        write(output/'comparison.json',result)
        return dict(result,stage_state='INCONCLUSIVE' if result['adoption']=='INCONCLUSIVE' else 'PASS')
