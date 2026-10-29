"""Verified read-only diagnostic weights/data; no learning-state reconstruction."""
import hashlib
import json
from pathlib import Path
from .identity import seal,digest,file_hash,confined

PRODUCER_SOURCE='sha256:fcfbb37256bdc75f7c85ac13be29195ac315f77d72e2fff846640c2f521143c4'
CHANGES={
 'scripts/analyze_pilot.py','scripts/pilot_hip_serial.py',
 'src/oeis_learn/decoder/bigint_codec.py','src/oeis_learn/fresh/bigint_diagnostics.py',
 'src/oeis_learn/fresh/pipeline_scheduler.py','src/oeis_learn/pipeline_validation/pilot_runtime.py',
 'src/oeis_learn/pipeline_validation/pilot_clocks.py','src/oeis_learn/pipeline_validation/pilot_evidence.py',
 *('src/oeis_learn/pipeline_validation/'+n+'.py' for n in ('followup_evaluation','followup_training','followup_study')),
 'tests/contract/test_native_fitting.py',
 'tests/unit/test_pilot_followup.py','tests/unit/test_pilot_followup_host.py',

 'scripts/foundation_local.py','src/oeis_learn/pipeline_validation/role.py',
 'src/oeis_learn/pipeline_validation/pilot.py','src/oeis_learn/pipeline_validation/adapters.py',
 'src/oeis_learn/container_validation.py','tests/unit/test_pipeline_diagnostics.py',
 'tests/unit/test_pipeline_diagnostics_runtime.py','specs/009-pipeline-throughput/contracts/pilot-diagnostics.md',
 *('src/oeis_learn/pipeline_validation/'+n+'.py' for n in ('diagnostics','diagnostics_contract','diagnostics_collect','diagnostics_import','diagnostics_runtime')),
}


def compatible_source(old,current):
    changes={k:dict(producer=old.get(k),current=current.get(k)) for k in sorted(set(old)|set(current)) if old.get(k)!=current.get(k)}
    if set(changes)-CHANGES or any(k not in current for k in old):raise ValueError('computational or unmapped diagnostic dependency changed')
    return changes


def snapshot(root,ordinal,trainer,audit):
    root=Path(root);d=json.loads(confined(root,'checkpoint.json').read_text())
    if set(d)!={'pointer','record','authority'} or d['authority']!='weights-only diagnostic; nonresumable':raise ValueError('nonresumable diagnostic descriptor required')
    pointer=d['pointer'];record=d['record']
    raw=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
    p=confined(root,'weights.pt')
    expected=record['files']['weights']['sha256']
    if (pointer!=trainer['checkpoint'] or pointer!=audit['checkpoint'] or pointer['sha256']!='sha256:'+hashlib.sha256(raw).hexdigest()
        or p.stat().st_size>512<<20 or file_hash(p)!=expected or expected!=audit['payload_hashes']['weights']
        or record['binding']['weights']!=expected or record['binding']['contract']!=audit['contract']):
        raise ValueError('snapshot bytes/pointer/audit/contract changed')
    return dict(ordinal=ordinal,path=p,descriptor=d,weights_sha256=expected,parameter_digest=trainer['final_parameter_digest'])


def selected_snapshot(run,trainer,audit):
    """Verify the public selected pointer against retained checkpoint/weight bytes."""
    checkpoint_root=confined(run,'stages/pilot.train-5/train-5/private/trainer/checkpoints')
    pointer=trainer['selected'];record_path=confined(checkpoint_root,pointer['path'])
    if file_hash(record_path)!=pointer['sha256']:raise ValueError('selected checkpoint pointer changed')
    record=json.loads(record_path.read_text());weights=record['files']['weights']
    path=confined(checkpoint_root,weights['path'])
    if (record['binding']['contract']!=audit['contract'] or record['binding']['weights']!=weights['sha256']
        or path.stat().st_size>512<<20 or file_hash(path)!=weights['sha256']):
        raise ValueError('selected weight/contract binding changed')
    return dict(ordinal=6,path=path,weights_sha256=weights['sha256'],parameter_digest=None,
        pointer=pointer,selection='producer retained selected checkpoint; parameter digest measured during probe')


def inspect(run,admission,producer,repo,plan):
    from .diagnostics_collect import Evidence
    from .longrun_contract import requirements,valid_cost,PROFILE
    from .pilot_contract import decision_payload
    from .longrun_import import validate_compatibility
    from .pilot_data import check,learner_view
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    run=Path(run);producer=Path(producer)
    if not run.is_dir() or not producer.is_dir():raise ValueError('models require full retained run/producer directories, not public ZIPs')
    with Evidence(run) as r,Evidence(admission) as a,Evidence(producer) as p:
        d=r.read('contract.json')['decision'];q=a.read('stages/study.report/qualification.json')
        proof=a.read('stages/pilot.compatibility/compatibility.json');auth=a.read('stages/study.report/authority.json');cost=a.read('stages/study.report/cost.json')
        if (d!=a.read('stages/study.report/decision.json') or d['source']!=PRODUCER_SOURCE or d['profile']!=PROFILE
            or d['state']!='ADMITTED' or q['source']!=PRODUCER_SOURCE or q['state']!='PASS'
            or auth['approved'] is not True or auth['decision_payload']!=decision_payload(d) or not auth['reviewed_by']):
            raise ValueError('exact approved completed pilot required')
        if seal(d)!=d or seal(q)!=q or seal(auth)!=auth:raise ValueError('changed diagnostic producer authority')
        validate_compatibility(proof,source=PRODUCER_SOURCE)
        if not valid_cost(cost,source=PRODUCER_SOURCE,compatibility=proof['identity']):raise ValueError('original exact cost authority changed')
        for field,name in [('authority','stages/study.report/authority.json'),('qualification','stages/study.report/qualification.json'),('cost','stages/study.report/cost.json'),('compatibility','stages/pilot.compatibility/compatibility.json')]:
            if d[field]['sha256']!='sha256:'+hashlib.sha256(a.bytes(name)).hexdigest():raise ValueError('original decision artifact changed')
        for key in requirements(q['engine']):
            receipt=a.summary['stages'].get(key,{})
            if receipt.get('state')!='PASS' or receipt.get('identity')!=q['receipts'][key]['identity']:raise ValueError('producer qualification missing')
        current=collect_snapshot(repo);changes=compatible_source(proof['current_snapshot'],current)
        if digest(proof['current_snapshot'])!=PRODUCER_SOURCE:raise ValueError('original source closure changed')
        prepared=confined(producer,'stages/pilot.prepare/prepare/private/prepared')
        for name in ('prepared.json','recipe.json','examples.json','prompts.json'):confined(prepared,name)
        m=check(prepared)
        if (m!=p.read('stages/pilot.prepare/prepared-index.json') or m['identity']!=q['work']
            or file_hash(prepared/'prepared.json')!=d['prepared']['sha256']):raise ValueError('original prepared corpus changed')
        examples=learner_view(json.loads((prepared/'examples.json').read_text()))
        from .diagnostics_contract import populations
        population=populations(examples,m['roster'],training_rows=plan['training_rows'])
        selected=set(plan['snapshots'])|{0};snapshots={}
        for i in sorted(selected-{6}):
            key='pilot.train-'+str(i);role='stages/'+key+'/train-'+str(i)+'/'
            if r.summary['stages'].get(key,{}).get('state')!='PASS':raise ValueError('requested complete snapshot unavailable')
            trainer=r.read(role+'trainer.json');audit=r.read(role+'checkpoint-audit.json')
            if (audit['source']!=PRODUCER_SOURCE or audit['work']!=m['identity'] or audit['boundary']['active'] or audit['boundary']['reserved']):raise ValueError('snapshot lineage/quiescence mismatch')
            s=snapshot(confined(run,'stages/'+key+'/private/snapshots/'+str(i)),i,trainer,audit)
            snapshots[i]=s
        if 6 in selected:
            role='stages/pilot.train-5/train-5/'
            snapshots[6]=selected_snapshot(run,r.read(role+'trainer.json'),r.read(role+'checkpoint-audit.json'))
        public=seal(dict(version='pipeline-pilot-diagnostics-import/v1',source=digest(current),producer_source=PRODUCER_SOURCE,
            work=m['identity'],study=d['identity'],producer_manifests={k:v.manifest['identity'] for k,v in [('run',r),('admission',a),('data',p)]},
            changed_diagnostic_dependencies=changes,populations=population,
            snapshots=[{k:v for k,v in snapshots[i].items() if k not in ('path','descriptor')} for i in sorted(selected)],
            authority='diagnostics only; no optimizer, resume, training or hidden truth'))
        return dict(prepared=prepared,manifest=m,contract=audit['contract'],snapshots=snapshots,populations=population,public=public)
