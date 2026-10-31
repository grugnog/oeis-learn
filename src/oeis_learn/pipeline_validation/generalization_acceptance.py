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
    result=seal(dict(status='PASS',runs=proofs,resume_equal=equal))
    write(output/'training-proof.json',result);return result


def training_proof(host,workspace,output,runtime,study,runs,*,compare=False):
    from .generalization_commands import _resume_inputs,_copy
    sources=[];settings=[]
    for ordinal,(root,result) in enumerate(runs):
        parent,pointer,refs=_resume_inputs(workspace,str((root/'checkpoints'/result['checkpoint']['path']).relative_to(workspace)))
        roster_ref=reference(parent/'retained.json',parent);roster=json.loads(verify_artifact(parent,roster_ref))
        refs+=[roster_ref,*[c['weights'] for c in roster['checkpoints']]]
        name=str(ordinal);target=output/'private/input'/name
        _copy(target,[(parent,r) for r in refs])
        sources.extend(dict(r,path=name+'/'+r['path']) for r in refs)
        settings.append(dict(directory=name,compatibility=result['compatibility'],cap=study['budgets']['target_tokens'],checkpoint=pointer,roster=roster_ref))
    return project_and_invoke(host,runtime['images'][0],workspace,'learner',
        {'training_targets':[],'verified_training_replay':sources},source_roots={'training_targets':output/'private/input','verified_training_replay':output/'private/input'},
        output=str((output/'proof').relative_to(workspace)),operation='check-training',settings=dict(runs=settings,compare=compare),seconds=60)


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
    artifacts={};checks={}
    try:
        if name=='retained-eval':
            view=runtime['evaluation']
            if view is None or view['role']!='selection':raise ValueError('acceptance uses selection fixtures only')
            checkpoints=[c['identity'] for c,_ in _checkpoints(base,view['checkpoints'])]
            result=evaluate(host,workspace,output/'private/evaluation',base,study,runtime,plan,checkpoints=checkpoints)
            if result['status']!='PASS':raise ValueError('retained evaluation incomplete')
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
        return dict(status='PASS',case=name,receipt=reference(output/'receipt.json',evidence_root))
    except BaseException as exc:
        ledger.transition('PARTIAL' if isinstance(exc,KeyboardInterrupt) else 'FAIL',reason=type(exc).__name__)
        if isinstance(exc,KeyboardInterrupt):raise
        return dict(status='FAIL',case=name,reason=str(exc),receipt=reference(output/'receipt.json',evidence_root))
