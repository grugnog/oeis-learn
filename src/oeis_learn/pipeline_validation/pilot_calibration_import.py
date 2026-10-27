"""Read-only import of one known successful producer; never reconstructs state."""
import hashlib
import json
from pathlib import Path
from .identity import confined,file_hash,digest,seal

PRODUCER_SOURCE='sha256:2b57e82976f9aaa322c105e0180edffcb38bda3141e32749652e554dcffbb043'


def snapshot(root,ordinal,trainer,audit,settings):
    """Verify all three independent public bindings before exposing weight bytes."""
    root=Path(root);desc=json.loads(confined(root,'checkpoint.json').read_text())
    if set(desc)!={'pointer','record','authority'} or desc['authority']!='weights-only diagnostic; nonresumable':
        raise ValueError('nonresumable diagnostic snapshot required')
    pointer=desc['pointer'];record=desc['record']
    canonical=json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
    if (pointer!=trainer['checkpoint'] or pointer!=audit['checkpoint'] or pointer!=settings['checkpoint']
        or pointer['sha256']!='sha256:'+hashlib.sha256(canonical).hexdigest()):
        raise ValueError('snapshot checkpoint pointer/record changed')
    weights=record['files']['weights']
    path=confined(root,'weights.pt')
    if path.stat().st_size>512<<20 or file_hash(path)!=weights['sha256'] or weights['sha256']!=audit['payload_hashes']['weights']:
        raise ValueError('snapshot weights changed')
    if record['binding']['weights']!=weights['sha256'] or record['binding']['contract']!=settings['contract_sha256']:
        raise ValueError('snapshot contract/weights binding changed')
    return dict(ordinal=ordinal,path=path,pointer=pointer,weights_sha256=weights['sha256'],original_source=PRODUCER_SOURCE)


def inspect(root,repo,*,engine):
    from .reuse import validate_import
    from .pilot_data import check
    from .pilot_contract import PROFILE
    from .pilot_recipe import validate_effective
    from .model_sizes import EXECUTION,PROFILES,parameters
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    root=Path(root);validate_import(root)
    inv=json.loads(confined(root,'invocation.json').read_text())
    if inv['binding']['source']!=PRODUCER_SOURCE:raise ValueError('known full-corpus producer required')
    old=inv['binding']['dependency_closure']['files'];current=collect_snapshot(repo)
    owners=[p for p in old if p.startswith(('crates/','src/oeis_learn/data/','src/oeis_learn/synthesis/'))]
    owners += ['src/oeis_learn/pipeline_validation/'+p+'.py' for p in ('pilot_data','pilot_selection','pilot_contract','pilot_recipe')]
    owners += ['src/oeis_learn/decoder/bigint_codec.py','src/oeis_learn/fresh/bigint_config.py','src/oeis_learn/fresh/bigint_splits.py']
    if any(old.get(p)!=current.get(p) for p in owners):raise ValueError('prepared-data/native mathematical dependency changed')
    summary=json.loads(confined(root,'summary.json').read_text());receipts=summary['stages']
    required=('pilot.prepare','pilot.session','hip.preflight','cpu.'+engine+'.semantic','pilot.operations.'+engine)
    if any(receipts.get(k,{}).get('state')!='PASS' for k in required):raise ValueError('successful primary producer prerequisites required')
    qualification=json.loads(confined(root,'stages/pilot.report/qualification.json').read_text())
    if (seal(qualification)!=qualification or qualification['state']!='PASS' or qualification['source']!=PRODUCER_SOURCE
        or qualification['engine']!=engine or qualification['profile']!=PROFILE['identity']):
        raise ValueError('producer qualification/source/profile changed')
    prepared=confined(root,'stages/pilot.prepare/prepare/private/prepared')
    m=check(prepared,hidden=True)
    if (m!=json.loads(confined(root,'stages/pilot.prepare/prepared-index.json').read_text())
        or m['source']!=PRODUCER_SOURCE or qualification['work']!=m['identity']):
        raise ValueError('private prepared input differs from public admission')
    validate_effective(json.loads((prepared/'recipe.json').read_text()))
    session=confined(root,'stages/pilot.session');snapshots=[]
    for i in range(3):
        print('AUDIT retained snapshot',i,flush=True)
        role=confined(session,'train-'+str(i))
        cleanup=json.loads(confined(role,'cleanup.json').read_text())
        result=json.loads(confined(role,'summary.json').read_text())
        if cleanup.get('within_bound') is not True or result.get('status')!='COMPLETE' or result.get('fatal') or result.get('source')!=PRODUCER_SOURCE or result.get('execution')!=EXECUTION:
            raise ValueError('producer training cleanup/numerical/source qualification missing')
        trainer=json.loads(confined(role,'trainer.json').read_text())
        if trainer!=result['trainer']:raise ValueError('public trainer/result mismatch')
        audit=json.loads(confined(role,'checkpoint-audit.json').read_text())
        if audit['source']!=PRODUCER_SOURCE or audit['work']!=m['identity'] or audit['boundary']['active'] or audit['boundary']['reserved']:
            raise ValueError('checkpoint data/source/quiescence changed')
        settings_path=confined(session,'private/evaluation-settings-'+str(i)+'.json')
        settings=json.loads(settings_path.read_text())
        if (digest(settings['recipe'])!=m['recipe'] or settings['pilot']['work']!=m['identity']
            or settings['pilot_prompts_sha256']!=m['files']['prompts.json']['sha256']
            or settings['pilot_truth_sha256']!=m['files']['private/truth.json']['sha256']):
            raise ValueError('saved evaluation settings/data/recipe changed')
        from oeis_learn.synthesis.state import ExperimentContract
        contract=ExperimentContract(**settings['contract'])
        if (contract.sha256!=settings['contract_sha256'] or contract.sha256!=audit['contract']
            or contract.data!=m['identity'] or contract.runtime!=digest(dict(source=PRODUCER_SOURCE,image=qualification['image']['gpu']))):
            raise ValueError('snapshot original source/image/data contract changed')
        for kind in ('generate','score'):
            prior=confined(session,kind+'-'+str(i))
            if json.loads(confined(prior,'cleanup.json').read_text()).get('within_bound') is not True:
                raise ValueError('producer evaluation cleanup unacknowledged')
        generation=json.loads(confined(session,'generate-'+str(i)+'/summary.json').read_text())
        if (generation.get('source')!=PRODUCER_SOURCE or generation.get('execution')!=EXECUTION
            or generation.get('counts')!=parameters(PROFILES['m100']) or generation.get('checkpoint')!=trainer['checkpoint']):
            raise ValueError('snapshot original actual-model evaluation changed')
        r=snapshot(confined(session,'private/snapshots/'+str(i)),i,trainer,audit,settings)
        r['settings']=settings;snapshots.append(r)
    public=seal(dict(version='pipeline-pilot-calibration-import/v1',producer_source=PRODUCER_SOURCE,work=m['identity'],
        producer_manifest=file_hash(root/'manifest.json'),qualification=qualification,mathematical_owners=owners,
        snapshots=[{k:v for k,v in r.items() if k not in ('path','settings')} for r in snapshots],
        authority='weights-only evaluation; never learner resume',prepared_native_readmission=False))
    return dict(prepared=prepared,manifest=m,snapshots=snapshots,public=public)
