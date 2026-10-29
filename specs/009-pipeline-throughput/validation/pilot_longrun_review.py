"""Reproduce public long-pilot integrity, training, clock, science and resource review."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from oeis_learn.pipeline_validation.bundle import verify
from oeis_learn.pipeline_validation.identity import seal,file_hash,digest
from oeis_learn.pipeline_validation.longrun_contract import PROFILE,SPEC,PURPOSE
from oeis_learn.pipeline_validation.longrun_import import clocks_audit
from oeis_learn.pipeline_validation.pilot_contract import ARMS,SPLITS
from oeis_learn.pipeline_validation.pilot import learning_report
from oeis_learn.pipeline_validation.model_sizes import EXECUTION,parameters,PROFILES
from oeis_learn.fresh.synthesis_policy import validate_selection_envelope


def require(value,reason):
    if not value:raise ValueError(reason)


def resources(samples):
    cpu=[r for r in samples if type(r.get('cpu_usage_usec')) in (int,float)]
    cores=(cpu[-1]['cpu_usage_usec']-cpu[0]['cpu_usage_usec'])/1e6/(cpu[-1]['time']-cpu[0]['time']) if len(cpu)>1 else None
    gpu=[r['gpu_busy_percent'] for r in samples if type(r.get('gpu_busy_percent')) in (int,float)]
    weighted=weight=0.
    for a,b in zip(samples,samples[1:]):
        dt=b['time']-a['time'];require(dt>0,'nonmonotone resource samples')
        if type(a.get('gpu_busy_percent')) in (int,float):weighted+=dt*a['gpu_busy_percent'];weight+=dt
    return dict(samples=len(samples),cpu_average_cores=cores,gpu_observation_mean=statistics.mean(gpu) if gpu else None,
        gpu_time_weighted_mean=weighted/weight if weight else None,gpu_observation_median=statistics.median(gpu) if gpu else None,
        sampled_seconds=samples[-1]['time']-samples[0]['time'] if len(samples)>1 else 0,
        gpu_known_sample_fraction=len(gpu)/len(samples) if samples else 0)


def review(path,producer):
    path=Path(path);producer=Path(producer);manifest=verify(path);verify(producer)
    with zipfile.ZipFile(producer) as parent:
        prepared=json.loads(parent.read('stages/pilot.prepare/prepared-index.json'))
        original_init=json.loads(parent.read('stages/pilot.session/train-0/trainer.json'))
        prepared_bytes=parent.read('stages/pilot.prepare/prepared-index.json')
    with zipfile.ZipFile(path) as z:
        read=lambda n:json.loads(z.read(n))
        summary=read('summary.json');inv=read('invocation.json');wrapper=read('contract.json');decision=wrapper['decision']
        source=inv['binding']['source'];work=prepared['identity']
        require(seal(inv)==inv and seal(wrapper)==wrapper and seal(decision)==decision,'changed invocation/decision')
        require(decision['profile']==PROFILE and decision['state']=='ADMITTED' and decision['version']=='pipeline-pilot-study/v2','changed full-clock study')
        require(inv['binding']==dict(source=source,study=decision['identity']) and decision['source']==source,'source/study binding mismatch')
        require(decision['prepared']['sha256']=='sha256:'+hashlib.sha256(prepared_bytes).hexdigest(),'prepared producer bytes changed')
        require(not manifest['omissions'] and summary['selected_scope']=='PASS','incomplete operational archive')
        inventory={r['path']:r for r in manifest['members']};receipts=summary['stages']
        expected={'pilot.current-images','study.report'}|{f'pilot.{role}-{i}' for role in ('train','evaluate') for i in range(6)}
        require(set(receipts)==expected,'changed stage roster')
        for key,row in receipts.items():
            require(row==read('stages/'+key+'/receipt.json') and seal(row)==row and row['state']=='PASS','changed/failed receipt')
            require(row['binding']==inv['binding'],'changed receipt source/study')
            for name,sha in row['files'].items():require(inventory['stages/'+key+'/'+name]['sha256']==sha,'unbound stage artifact')
            for parent,sha in row.get('prerequisites',{}).items():require(sha==receipts[parent]['identity'],'broken prerequisite')
        ledger=read('stages/study.report/debits.json')
        require(seal(ledger)==ledger and ledger['decision']==decision and ledger['expires']-ledger['started']==172800,'changed original allowance')
        require(len(ledger['invocations'])==1 and summary['elapsed_seconds']<=172800,'unexpected recovery or parent overrun')
        charged=defaultdict(float);actual=defaultdict(float)
        for key,row in ledger['slots'].items():
            require(row['key']==key and row['state']=='COMPLETE' and row['started']>=ledger['started'] and row['ended']<=ledger['expires'],'invalid slot/lifetime')
            require(0<=row['ended']-row['started']<=row['seconds'],'late/negative slot')
            charged[row['category']]+=row['seconds'];actual[row['category']]+=row['ended']-row['started']
        for category,seconds in charged.items():require(seconds<=SPEC['reservations'][category],'category overrun')
        require(charged==dict(qualification=1200,setup=600,learner=86400,evaluation=31380),'changed prospective allocation')
        points=[];snapshots=[];training=[];prompts=None;all_role_peaks=[];all_machine_peaks=[];metadata=None
        previous=None;known_contract=None;seen_binding=None;all_successes=0
        for i in range(6):
            train=f'stages/pilot.train-{i}/train-{i}/';evaluation=f'stages/pilot.evaluate-{i}/'
            trainer=read(train+'trainer.json');audit=read(train+'checkpoint-audit.json');identity=read(train+'checkpoint-identity.json')
            role=read(train+'summary.json');pilot=read(train+'pilot.json');state=pilot['state'];point=read(evaluation+f'snapshot-{i}.json')
            require(trainer==role['trainer'] and trainer['pilot']==pilot and seal(audit)==audit,'changed trainer/audit')
            require(role['source']==audit['source']==source and role['work']==audit['work']==work and role['status']=='COMPLETE','trainer source/work mismatch')
            require(role['execution']==trainer['execution']==EXECUTION and trainer['counts']['actual']==parameters(PROFILES['m100']) and not trainer['fixture'],'actual model/numerics mismatch')
            require(trainer['torch_threads']==dict(interop=1,intraop=2),'thread policy changed')
            require(audit['checkpoint']==trainer['checkpoint']==identity['latest']==point['checkpoint'],'latest milestone substituted')
            require(audit['ledger']==dict(budget=86400,records={'learning-pilot':{'debit':86400,'status':'active'}}),'worker allowance renewed')
            require(set(audit['states'])=={'curriculum','data_order','optimizer','replay','rng','selection','weights'},'incomplete checkpoint state')
            require(audit['boundary']['active']==audit['boundary']['reserved']==0,'checkpoint not drained')
            if known_contract is None:known_contract=audit['contract'];seen_binding=pilot['binding']
            require(audit['contract']==known_contract and pilot['binding']==seen_binding and state['interrupted_ordinals']==[],'changed contract/budget/gap')
            require(audit['data_order']['completed']==trainer['update']==state['updates']==point['updates'],'update mismatch')
            require(state['completed_ordinal']==i and state['role_ordinal']==i and pilot['complete'],'missing complete milestone')
            require(point==ledger['slots'][f'pilot.evaluate-{i}']['result']['snapshot'],'ledger/snapshot mismatch')
            require(ledger['slots'][f'pilot.train-{i}']['result']['trainer']==trainer,'ledger/trainer mismatch')
            if i==0:
                require(trainer['update']==state['tokens']==0 and trainer['initial_parameter_digest']==original_init['initial_parameter_digest'],'fresh seeded baseline changed')
            else:require(trainer['initial_parameter_digest']==previous['final_parameter_digest'],'full-state weight continuity failed')
            previous=trainer
            require(point['parameter_digest']==trainer['final_parameter_digest'] and state['tokens']==point['tokens'],'actual exposure/weights mismatch')
            require(state['unique_examples']<=(32768 if i else 0) and state['updates']<=SPEC['max_updates'] and state['tokens']<=SPEC['max_tokens'],'work cap exceeded')
            bins=read(train+'training-bins.json')
            require(len(bins)<=SPEC['max_training_bins'] and all(r['key'][0]==i for r in bins),'bin capacity/role mismatch')
            require(sum(r['tokens'] for r in bins)==pilot['tokens'] and sum(r['updates'] for r in bins)==pilot['updates'],'minute bins/exposure mismatch')
            require(read(train+'pilot-validation.json')==point['teacher_forcing'],'teacher-forcing snapshot changed')
            training.append(dict(ordinal=i,learner_role_seconds=ledger['slots'][f'pilot.train-{i}']['result']['learner_role_seconds'],
                tokens=pilot['tokens'],updates=pilot['updates'],cumulative_tokens=state['tokens'],cumulative_updates=state['updates'],
                unique_examples=state['unique_examples'],effective_token_epochs=point['effective_token_epochs'],
                components=state['components'],teacher_forcing=point['teacher_forcing']['splits']))
            gen=evaluation+f'generate-{i}/';score=evaluation+f'score-{i}/'
            records=read(gen+'selections.json');outcomes=read(score+'outcomes.json');clocks=read(gen+'portfolio-clocks.json');attempts=read(gen+'attempt-index.json')
            require(set(records)==set(outcomes)=={p for ids in prepared['roster'].values() for p in ids},'frozen prompt roster changed')
            meta={k:dict(id=k,family=r['family'],strata=r['strata']) for k,r in records.items()}
            if metadata is None:metadata=meta;prompts=list(meta.values())
            require(meta==metadata,'prompt family/strata changed')
            model=clocks_audit(clocks,attempts,records)
            require(clocks['source']==source and clocks['work']==work,'clock source/work changed')
            g=read(gen+'summary.json');s=read(score+'summary.json');boundary=read(gen+'boundary.json')
            require(g['counts']==parameters(PROFILES['m100']) and g['execution']==EXECUTION and g['checkpoint']==point['checkpoint'],'different model/checkpoint evaluated')
            require(g['clock_complete'] and point['evaluation_cost']['cost_complete'] and point['evaluation_cost']['clock_receipt']==clocks['identity'],'incomplete clock receipt')
            require(boundary['active']==boundary['reserved']==0 and s['cpu_only'] and s['rows']==296,'undrained native or scoring scope')
            require(point['outcomes']==outcomes and all(set(v)==set(ARMS) for v in outcomes.values()),'outcome/arm record changed')
            reasons=defaultdict(Counter)
            for prompt,arms in outcomes.items():
                for arm,outcome in arms.items():
                    require(outcome['upstream']==records[prompt]['arm_evidence'][arm],'upstream/scorer mismatch')
                    selected=records[prompt]['selections'][arm]
                    if selected is not None:
                        candidate=validate_selection_envelope(selected)
                        require(outcome['candidate_sha256']==candidate['source_sha256'] and candidate['provenance']['snapshot']==model,'selection/scoring/model mismatch')
                    if outcome['success']:
                        require(outcome['complete'] and selected is not None and outcome['scoring']['outcome']=='success','unverified success')
                        for method in ('production','reference'):require(outcome['scoring'][method]['outcome']=='ok' and outcome['scoring'][method]['coverage']==100,'success without exact100 coverage')
                        all_successes+=1
                    if not outcome['complete']:reasons[records[prompt]['strata']['split']+':'+arm][outcome['reason']]+=1
            role_audits={}
            for kind,prefix,summary_role in (('train',train,role),('generate',gen,g),('score',score,s)):
                require(summary_role['source']==source and summary_role['work']==work and not summary_role.get('fatal'),'role lineage failed')
                memory=read(prefix+'memory.json');reservation=read(prefix+'resource-reservation.json')
                require(memory==summary_role['resources'] and memory['qualified'] and memory['breach'] is None,'resource qualification failed')
                require(memory['role_peak_upper_bytes']<=reservation['reservation']['peak_bytes'] and memory['machine_peak_upper_bytes']<=memory['ceiling_bytes'],'memory cap exceeded')
                require(read(prefix+'cleanup.json')['within_bound'] is True and read(prefix+'isolation-audit.json')['status']=='PASS','unacknowledged cleanup/isolation')
                samples=[json.loads(row) for row in z.read(prefix+'resource-samples.jsonl').splitlines()]
                role_audits[kind]=dict(resources(samples),role_peak_upper_bytes=memory['role_peak_upper_bytes'],machine_peak_upper_bytes=memory['machine_peak_upper_bytes'])
                all_role_peaks.append(memory['role_peak_upper_bytes']);all_machine_peaks.append(memory['machine_peak_upper_bytes'])
            training[-1]['resources']=role_audits['train']
            points.append(point);snapshots.append(dict(ordinal=i,model=model,cost_complete=clocks['cost_complete'],attempts=attempts['states'],
                terminations=clocks['terminations'],unresolved_reasons=dict(reasons),resources=role_audits))
        learning=read('stages/study.report/learning.json')
        require(learning==learning_report(prepared,prompts,points,purpose=PURPOSE),'learning report differs from source outcomes')
        disks=[read(n) for n in z.namelist() if n.endswith('/disk.json')]
        require(all(d['breach'] is None and d['peak_bytes']<=d['cap_bytes'] for d in disks),'disk breach')
        def paired_counts(a,b,split,arm):
            ids=prepared['roster'][split];aa=points[a]['outcomes'];bb=points[b]['outcomes']
            common=[k for k in ids if aa[k][arm]['complete'] and bb[k][arm]['complete']]
            return dict(known_pairs=len(common),unresolved_pairs=len(ids)-len(common),
                gained=sum(not aa[k][arm]['success'] and bb[k][arm]['success'] for k in common),
                lost=sum(aa[k][arm]['success'] and not bb[k][arm]['success'] for k in common))
        return dict(version='pipeline-longrun-public-review/v1',archive_sha256=file_hash(path),source=source,study=decision['identity'],work=work,
            inventory=dict(members=len(manifest['members'])+1,payloads=len(manifest['members']),omissions=manifest['omissions'],hash_mismatches=[]),
            elapsed_seconds=summary['elapsed_seconds'],stages={k:r['state'] for k,r in receipts.items()},
            original_lifetime_seconds=172800,charged_seconds=dict(charged),observed_slot_seconds=dict(actual),
            training=training,snapshots=snapshots,total_exact100_success_outcomes=all_successes,
            metrics={k:dict(points=[{n:x[n] for n in ('ordinal','tokens','updates','learner_role_seconds','measurement')} for x in v['points']],
                extrapolation=v['extrapolation'],further_compute=v['further_compute']) for k,v in learning['series'].items()},
            paired_late_change={split:paired_counts(2 if split=='structural_development' else 3,5,split,'equal_budget') for split in SPLITS},
            storage=dict(transient_peak_bytes=max(d['peak_bytes'] for d in disks),minimum_free_bytes=min(d['minimum_free_bytes'] for d in disks),
                archive_bytes=path.stat().st_size,durable_peak_bytes=None,durable_audit='Outer watcher not included in this public archive; no independent cumulative peak claim'),
            resource_maxima=dict(role_peak_upper_bytes=max(all_role_peaks),machine_peak_upper_bytes=max(all_machine_peaks)),
            public_worker_identity='Original full-state worker identity predates public seen/bin compaction; public bytes are authenticated by sealed role/stage/archive hashes, not by resealing that trimmed object.',
            admission_evidence=dict(decision=decision,independent_bundle_verification='PENDING: separately referenced admission archive not attached'),
            operational='COMPLETE_ALL_SIX_MILESTONES',scientific='INCONCLUSIVE_FOR_FUTURE_LEARNING_CAPACITY',
            disposition='Do not extend this same fixed corpus unchanged; review fresh generic diversity/curriculum in a separate prospective decision.',
            automatic_run_authorized=False)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('--producer',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    result=review(a.archive,a.producer);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(result['operational'],result['scientific'],result['elapsed_seconds'],result['inventory'])

if __name__=='__main__':main()
