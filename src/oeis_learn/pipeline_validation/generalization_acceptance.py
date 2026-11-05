"""Small acceptance cases using the same isolated production workflow adapters."""
import json
from pathlib import Path
from .identity import seal,write,confined
from .generalization_preparation import reference
from .generalization_roles import project_and_invoke
from oeis_learn.generalization.contracts import verify_artifact


def check_training(inputs,output,settings,*,seconds):
    """Installed learner role: inspect exact full state without exporting tensors."""
    import torch
    import numpy as np
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    if set(settings)!={'runs','compare'} or not 1<=len(settings['runs'])<=2:raise ValueError('closed training proof request required')
    states=[];proofs=[]
    for run in settings['runs']:
        if set(run)!={'directory','compatibility','cap','checkpoint','roster'}:raise ValueError('closed training proof run required')
        root=confined(inputs['verified_training_replay'],run['directory'])
        roster=json.loads(verify_artifact(root,run['roster']))
        if seal(roster)!=roster:raise ValueError('retained roster seal')
        state=GeneralizationCheckpointStore(root,run['compatibility'],run['cap']).load(run['checkpoint'])
        latest=next(c for c in roster['checkpoints'] if c['state']==run['checkpoint'])
        initial=next(c for c in roster['checkpoints'] if 'initial' in c['pins'])
        import io
        origin=torch.load(io.BytesIO(verify_artifact(root,initial['weights'])),map_location='cpu',weights_only=True)
        weights=state['states']['weights'];changed=False
        for member,values in weights.items():
            for key,value in values.items():
                if not torch.isfinite(value).all():raise ValueError('nonfinite learner parameter')
                changed=changed or not torch.equal(value,origin[member][key])
        if state['ledger'].updates<1 or not changed:raise ValueError('actual finite parameter update required')
        states.append(state)
        proofs.append(dict(checkpoint=latest['identity'],weights=latest['weights']['sha256'],initial=initial['weights']['sha256'],
            updates=state['ledger'].updates,tokens=state['ledger'].consumed,finite=True,changed=True))
    equal=None
    if settings['compare']:
        if len(states)!=2:raise ValueError('two full states required for resume comparison')
        def same(a,b):
            if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
            if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
            if isinstance(a,dict):return isinstance(b,dict) and set(a)==set(b) and all(same(v,b[k]) for k,v in a.items())
            if isinstance(a,(list,tuple)):return type(a)==type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
            return a==b
        equal=states[0]['ledger']==states[1]['ledger'] and all(same(states[0]['states'][k],states[1]['states'][k]) for k in ('weights','optimizer','rng','data','replay'))
        if not equal:raise ValueError('resumed state differs from uninterrupted state')
    extra={}
    if any('pac' in state['states']['scheduler'] for state in states):
        if not all('pac' in state['states']['scheduler'] for state in states):raise ValueError('mixed PAC/legacy proof')
        if settings['compare']:
            a,b=(state['states']['scheduler']['pac'] for state in states)
            keys=set(a)-{'observed_seconds'}
            if keys!=set(b)-{'observed_seconds'} or any(not same(a[k],b[k]) for k in keys):raise ValueError('resumed PAC policy/observer differs')
            extra['pac_equal']=True
    result=seal(dict(status='PASS',runs=proofs,resume_equal=equal,**extra))
    write(output/'training-proof.json',result);return result


def training_proof(host,workspace,output,runtime,study,runs,*,compare=False):
    import time
    from oeis_learn.generalization.budgets import remaining_seconds
    from oeis_learn.generalization.contracts import artifact_refs
    from oeis_learn.generalization.resources import PROOF_SECONDS
    envelope=runtime['envelope']['proof'] if runtime.get('version')=='pac-runtime/v1' else None
    deadline=time.monotonic()+(envelope['outer'] if envelope else PROOF_SECONDS)
    sources=[];settings=[];origins={}
    for ordinal,(root,result) in enumerate(runs):
        parent=root/'checkpoints';pointer=result['checkpoint']
        record=json.loads(verify_artifact(parent,pointer))
        roster_ref=reference(parent/'retained.json',parent);roster=json.loads(verify_artifact(parent,roster_ref))
        initial=next(c for c in roster['checkpoints'] if 'initial' in c['pins'])
        refs={r['path']:r for r in [pointer,roster_ref,initial['weights'],*artifact_refs(record)]}
        name=str(ordinal)
        for ref in refs.values():
            projected=dict(ref,path=name+'/'+ref['path'])
            sources.append(projected);origins[projected['path']]=(parent,ref)
        settings.append(dict(directory=name,compatibility=result['compatibility'],cap=record['ledger']['cap'] if envelope else study['budgets']['target_tokens'],checkpoint=pointer,roster=roster_ref))
    return project_and_invoke(host,runtime['images'][0],workspace,'learner',
        {'training_targets':[],'verified_training_replay':sources},
        artifact_sources={'verified_training_replay':origins},
        output=str((output/'proof').relative_to(workspace)),operation='check-training',settings=dict(runs=settings,compare=compare),seconds=remaining_seconds(deadline),**({'envelope':envelope} if envelope else {}))



def _training_runs(output,result):
    return [(output/run['root'],run['result']) for run in result['runs']]


def run_case(name,host,workspace,output,base,study,runtime,plan,*,qualification,evidence_root=None):
    from .generalization_commands import train,_copy
    from .generalization_workflow import evaluate,_checkpoints,replay
    from oeis_learn.generalization.feature_acceptance import OBLIGATIONS
    from .receipts import Ledger
    if runtime['purpose']!='acceptance':raise ValueError('reduced acceptance runtime required')
    from oeis_learn.generalization.execution import validate_policy
    validate_policy(study,runtime,case='refresh' if name=='failure-packaging' else name)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    evidence_root=Path(evidence_root) if evidence_root is not None else output.parent
    ledger=Ledger(output,'acceptance.'+name,dict(case=name,qualification=qualification))
    ledger.transition('READY');ledger.transition('RUNNING')
    artifacts={};checks={};initializer=None
    try:
        if name=='retained-eval':
            view=runtime['evaluation']
            if view is None or view['role']!='selection':raise ValueError('acceptance uses selection fixtures only')
            checkpoints=[c['identity'] for c,_ in _checkpoints(base,view['checkpoints'])]
            result=evaluate(host,workspace,output/'private/evaluation',base,study,runtime,plan,checkpoints=checkpoints)
            if result['status']!='PASS':raise ValueError('retained evaluation incomplete')
            if (result.get('initializer') or {}).get('status')=='PASS':
                initializer=reference(output/'private/evaluation/initializer/publication/role'/result['initializer']['artifact']['path'],workspace)
            artifacts=dict(checkpoint_identity=dict(checkpoints=checkpoints),evaluation=result)
            checks=dict(weights_loaded=True,visible_only=True,matched_budgets=True)
        elif name in ('refresh','parameter-targets'):
            arms=['static','refreshed'] if name=='refresh' else ['literal','parameterized']
            if set(study['arms'])!=set(arms):raise ValueError('acceptance arm pair mismatch')
            if name=='parameter-targets' and not plan.get('target_pairs_parameterized'):
                raise ValueError('acceptance needs an actual supported parameterized learning target')
            trained=train(host,workspace,output/'private/training',base,study,runtime,plan,arms=arms,seeds=study['seeds'])
            if trained['status']!='PASS':raise ValueError('paired training incomplete')
            proofs=[training_proof(host,workspace,output/'private'/('proof-'+str(i)),runtime,study,[run]) for i,run in enumerate(_training_runs(output/'private/training',trained))]
            if any(p['status']!='PASS' for p in proofs):raise ValueError('finite update proof failed')
            if name=='refresh':
                if runtime['training']['refreshed']['data_mode']!='refresh' or runtime['training']['static']['data_mode']!='static':raise ValueError('static/refresh modes required')
                static={r['id'] for ref in runtime['training']['static']['shards'] for r in json.loads(verify_artifact(base,ref))}
                refreshed=[r['id'] for ref in runtime['training']['refreshed']['shards'] for r in json.loads(verify_artifact(base,ref))]
                consumed=trained['runs'][1]['result']['updates']*runtime['training']['refreshed']['batch_size']
                if not set(refreshed[:consumed])-static:raise ValueError('acceptance did not update on any fresh variant')
                # The custodian already compared every refreshed row to the frozen
                # descriptor stream. Retain that binding, and actual trained bytes.
                artifacts=dict(preparation=dict(study=study['identity'],generator=study['data']['generator']['sha256'],descriptors=study['data']['descriptors']['sha256']),training=dict(runs=proofs))
                checks=dict(fixed_support=True,fresh_shards=True,finite_update=True)
            else:
                if study['progression']['stage']!='P3' or not plan.get('target_pairs_verified'):raise ValueError('verified paired target study required')
                artifacts=dict(literal_training=proofs[0],parameterized_training=proofs[1],target_audit=dict(study=study['identity'],verified_pairs=plan['target_pairs_verified']))
                checks=dict(literal_update=True,parameterized_update=True,grounding_verified=True)
        elif name=='replay-resume':
            result=replay(host,workspace,output/'private/replay',base,study,runtime,plan,acceptance_resume=True)
            if result['status']!='PASS' or not result['runs'] or not all(r.get('resume',{}).get('status')=='PASS' and (r.get('paired_proof') or {}).get('status')=='PASS' and r['cumulative_discoveries']>0 for r in result['runs']):raise ValueError('paired finite updates, replay/resume or actual admission incomplete')
            artifacts=dict(admission=dict(rounds=result['rounds'],search=[r['search'] for r in result['runs']]),
                continued_sft=dict(runs=[r['paired']['runs'][0]['result'] for r in result['runs']],finite_updates=[r['paired_proof'] for r in result['runs']]),
                verified_replay=dict(runs=[r['paired']['runs'][1]['result'] for r in result['runs']]),resume=dict(runs=[r['resume'] for r in result['runs']]))
            if (output.parent/'bootstrap.json').is_file():
                artifacts['admission']['initializer_dependency']=json.loads((output.parent/'bootstrap.json').read_text())
            checks=dict(training_only_admission=True,paired_control=True,full_state_resume=True,finite_update=True)
        elif name=='failure-packaging':
            from .bundle import pack,verify
            failed=output/'private/injected';failed.mkdir(parents=True)
            write(failed/'summary.json',dict(selected_scope='FAIL',scientific_status='UNMEASURED',reason='injected acceptance failure'))
            write(failed/'failure.json',seal(dict(stage='injected',status='FAIL')))
            write(failed/'private/hidden.json',dict(secret='must not be shared'))
            archive=pack(failed,mandatory=['summary.json','failure.json']);verified=verify(archive)
            if verified['summary_state']!='FAIL' or any('private' in Path(p['path']).parts for p in verified['members']):raise ValueError('failure recovery integrity')
            artifacts=dict(failure_ledger=json.loads((failed/'failure.json').read_text()),archive_verification=verified)
            checks=dict(failure_preserved=True,archive_verified=True,private_payloads_excluded=True)
        else:raise ValueError('unknown acceptance case')
        wanted,labels=OBLIGATIONS[name]
        if set(checks)!=set(wanted) or set(artifacts)!=set(labels):raise ValueError('acceptance obligations incomplete')
        # Public proofs retain identities, measurements and outcomes, never full
        # state/truth references. Detailed raw work remains private in the run.
        def public(value):
            if isinstance(value,dict):
                if set(value)=={'path','sha256','bytes'}:return dict(artifact_identity=value['sha256'],artifact_bytes=value['bytes'])
                return {k:public(v) for k,v in value.items() if k not in {'source','final_source','original_source','transformed_source','assignment','values','indices'} and not (k=='tokens' and isinstance(v,list))}
            if isinstance(value,list):return [public(v) for v in value]
            return value
        refs={}
        for label,value in artifacts.items():
            path=output/(label+'.json');write(path,public(value));refs[label]=reference(path,evidence_root)
        result=seal(dict(version='generalization-acceptance-result/v1',case=name,qualification=qualification,status='PASS',checks=checks,artifacts=refs))
        write(output/'case-result.json',result)
        files={p.name:reference(p,output)['sha256'] for p in output.glob('*.json') if p.name!='receipt.json'}
        ledger.transition('PASS',files=files)
        return dict(status='PASS',case=name,receipt=reference(output/'receipt.json',evidence_root),**({'initializer':initializer} if initializer is not None else {}))
    except BaseException as exc:
        ledger.transition('PARTIAL' if isinstance(exc,KeyboardInterrupt) else 'FAIL',reason=type(exc).__name__)
        if isinstance(exc,KeyboardInterrupt):raise
        return dict(status='FAIL',case=name,reason=str(exc),receipt=reference(output/'receipt.json',evidence_root))


def pac_acceptance_fixture(output,*,execute,device,recipe=None,source=None,environment=None):
    """Custodian-only mechanical data; native callback is mandatory outside CPU tests."""
    from oeis_learn.data.bigint_programs import canonical_family
    from oeis_learn.pipeline_validation.generalization_preparation import prepare_pac_training
    from oeis_learn.pipeline_validation.generalization_training import PacPreparedBatches
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.budgets import charge_tokens
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from .identity import digest
    rows=[];probes=[]
    patterns=('(program (return (add $index (param.get p0))))',
              '(program (return (select (lt $index (param.get p0)) (int 1) (int 0))))')
    for f,pattern in enumerate(patterns):
        for i in range(12):
            v=str(i+2)
            row=dict(id=f'acceptance-{f}-{i}',role='training',family=canonical_family(pattern),parameter_source=pattern,
                source=pattern.replace('(param.get p0)',f'(int {v})'),assignment={'p0':v},indices=list(map(str,range(100))))
            (rows if i<4 else probes).append(row)
    bundle=prepare_pac_training(rows,probes,output=output,execute=execute,final_exclusions=[],seed=7,batch_size=2,purpose='acceptance')
    batches=PacPreparedBatches(output,bundle,arm='C',batch_size=2,updates=100000)
    # A conservative encoded cap guarantees sixteen whole batches even when the
    # actual family/row choices have different target lengths. Stop by updates.
    tokens=16*2*max(charge_tokens([r['tokens']],bos=1,pad=0) for r in batches.rows.values())
    return dict(study=digest('pac-mechanical-acceptance/v1'),arm='C',seed=7,recipe=recipe or resolve(dict(version='bigint-bootstrap/v1')),
        shards=bundle['parameterized'],data_mode='static',batch_size=2,target_tokens=tokens,milestones=[],seconds=900,device=device,
        checkpoint_policy='hourly/v1',pac=dict(bundle=bundle,purpose='acceptance',cadence='pac-acceptance/v1',
        source=source or digest(source_snapshot()),environment=environment or digest(['acceptance',device]),initializer=digest('fresh'),
        observation_seconds=600,publication_seconds=60))


def pac_acceptance(inputs,output,settings,*,seconds):
    """Actual installed learner, complete-state proof and real failure exporter."""
    import time
    from .generalization_role import learner
    from .generalization_training import PacPreparedBatches
    from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
    from oeis_learn.generalization.budgets import remaining_seconds,charge_tokens
    from .bundle import pack,verify
    if set(settings)!={'config'}:raise ValueError('closed PAC acceptance settings')
    config=settings['config'];pac=config.get('pac',{})
    if config['arm']!='C' or pac.get('purpose')!='acceptance' or pac.get('cadence')!='pac-acceptance/v1':
        raise ValueError('explicit reduced PAC acceptance required')
    deadline=time.monotonic()+seconds;output=Path(output);output.mkdir(parents=True,exist_ok=False)
    batches=PacPreparedBatches(inputs['training_targets'],pac['bundle'],arm='C',batch_size=config['batch_size'],updates=100000)
    if batches.inventory['version']!='pac-acceptance-inventory/v1':raise ValueError('mechanical acceptance inventory required')
    minimum=16*config['batch_size']*max(charge_tokens([r['tokens']],bos=1,pad=0) for r in batches.rows.values())
    if config['target_tokens']<minimum:raise ValueError('encoded token cap cannot reach acceptance boundaries')
    if {r['target_stratum'] for r in batches.inventory['rows']}!={'supported','fallback'}:raise ValueError('parameterized and whole-row fallback required')
    def run(name,stop,prior=None):
        root=output/'private'/name
        value=learner(dict(training_targets=inputs['training_targets'],verified_training_replay=prior[0] if prior else inputs['training_targets']),
            root,dict(config=config,resume=prior[1]['checkpoint'] if prior else None,stop_after_updates=stop),
            seconds=min(config['seconds'],remaining_seconds(deadline)))
        if value['updates']!=stop or value['device']!=config['device']:raise ValueError('acceptance did not reach actual update/device boundary')
        folder=root/'private/learner/checkpoints'
        state=GeneralizationCheckpointStore(folder,value['compatibility'],config['target_tokens']).load(value['checkpoint'])
        return folder,value,state
    full=run('full',16);proofs={}
    def proof_run(run):
        folder,value,_=run
        return dict(directory=str(folder.relative_to(output)),compatibility=value['compatibility'],cap=config['target_tokens'],
                    checkpoint=value['checkpoint'],roster=reference(folder/'retained.json',folder))
    for boundary in (1,2):
        paused=run('paused-'+str(boundary),boundary)
        policy=paused[2]['states']['scheduler']['pac']['policy']
        if policy['block']['phase']!=('updates' if boundary==1 else 'post'):raise ValueError('acceptance pause missed block boundary')
        resumed=run('resumed-'+str(boundary),16,paused)
        where=output/('proof-'+str(boundary))
        proof=check_training({'verified_training_replay':output},where,dict(runs=[proof_run(full),proof_run(resumed)],compare=True),seconds=remaining_seconds(deadline))
        proofs[str(boundary)]=reference(where/'training-proof.json',output)
    model_state=full[2]['states'];controller=model_state['scheduler']['pac'];policy=controller['policy']
    grades=sum(a['last_grade'] is not None for a in policy['review']['anchors'].values())
    exposed={key for key,a in policy['review']['anchors'].items() if a['acquired']}
    strata={r['target_stratum'] for r in batches.inventory['rows'] if r['id'] in exposed}
    event=policy['last_observation']
    if policy['kernel']['valid_updates']<6 or event['status'] in {'warmup','degenerate','censored'} or not grades or strata!={'supported','fallback'}:
        raise ValueError('acceptance lacks actual non-warmup reward/graded review/paired target exposure')
    if config['device']=='cuda' and not model_state['rng']['device']:raise ValueError('actual device RNG missing')
    # Evaluate the published final weights after a fresh weights-only load.
    # This is a training-probe forward check, never a held-out research score.
    from .generalization_evaluation import load_retained
    from .generalization_curriculum import observe_targets
    roster=json.loads((full[0]/'retained.json').read_text())
    descriptor=next(c for c in roster['checkpoints'] if c['state']==full[1]['checkpoint'])
    encoder,decoder,_=load_retained(full[0],descriptor['weights'],config['recipe'],device=config['device'])
    evaluation_rows=[batches.probe_rows[ids[0]] for ids in batches.probes.values()]
    remaining_seconds(deadline)
    observation=observe_targets(encoder,decoder,evaluation_rows,device=config['device'])
    remaining_seconds(deadline)
    evaluated=seal(dict(version='pac-checkpoint-forward/v1',status='PASS',checkpoint=descriptor['identity'],weights=descriptor['weights']['sha256'],
        device='hip' if config['device']=='cuda' else 'cpu',scope='training probes only',rows=len(evaluation_rows),**observation))
    write(output/'checkpoint-evaluation.json',evaluated)
    del encoder,decoder
    failed=output/'failure';failed.mkdir()
    write(failed/'summary.json',dict(selected_scope='FAIL',scientific_status='UNMEASURED',reason='injected acceptance failure'))
    write(failed/'failure.json',dict(primary='injected_after_actual_updates',status='FAIL'))
    write(failed/'private/hidden.json',dict(secret='excluded'))
    archive=pack(failed,mandatory=['summary.json','failure.json']);checked=verify(archive)
    if checked['summary_state']!='FAIL' or any('private' in Path(r['path']).parts for r in checked['members']):raise ValueError('acceptance failure export not verified')
    result=seal(dict(version='pac-feature-acceptance/v2',status='PASS',source=pac['source'],environment=pac['environment'],
        device='hip' if config['device']=='cuda' else 'cpu',research_eligible=config['device']=='cuda',
        cadence=dict(block_updates=2,cycle_updates=8,review_interval=1,common_updates=8),updates=16,
        valid_rewards=policy['kernel']['valid_updates'],graded_anchors=grades,resume_boundaries=[1,2],encoded_token_minimum=minimum,
        proofs=proofs,checkpoint_evaluation=reference(output/'checkpoint-evaluation.json',output),failure_archive=reference(archive,output)))
    from oeis_learn.generalization.feature_acceptance import verify_pac_acceptance
    verify_pac_acceptance(result,root=output,source=pac['source'],environment=pac['environment'],device=result['device'])
    write(output/'pac-acceptance.json',result)
    return result
