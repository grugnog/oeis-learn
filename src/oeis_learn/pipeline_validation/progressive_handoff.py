"""Stdlib host qualification and acceptance; immutable plans, finite owned stages."""
from oeis_learn.generalization.progressive.contracts import CAMPAIGN_DOCUMENT_BYTES

import json
import runpy
import time
from pathlib import Path
from oeis_learn.generalization.progressive.contracts import (
    atomic_write, load, seal, digest, reference, verify_refs, validate,
)
from .identity import confined
from .progressive_evidence import publish


def named_plan(root, document, name, version, fields, optional=()):
    refs=[r['artifact'] for r in document['payload']['references'] if r['name']==name]
    if not refs:
        return None
    if len(refs)!=1:
        raise ValueError('one immutable '+name+' required')
    value=json.loads(verify_refs(root,refs,max_bytes=32<<20)[0])
    if not isinstance(value,dict) or not set(fields)|{'version','identity'}<=set(value) or set(value)-set(fields)-{'version','identity'}-set(optional) or value['version']!=version or seal(value)!=value:
        raise ValueError('closed original '+name+' required')
    maximum_seconds = 86400 if version == 'progressive-calibration-plan/v1' else 3600
    if type(value['seconds']) is not int or not 1<=value['seconds']<=maximum_seconds:
        raise ValueError('finite handoff stage allowance required')
    return value


def stage(host, root, image, output, argv, *, gpu, seconds, readonly_inputs=()):
    # This is the existing socket-free engine launcher, with its GPU lease and
    # inspected confinement/cleanup. No host Torch or proposer dependencies.
    if gpu:
        from .generalization_execution import startup_environment
        argv=['env',*(k+'='+v for k,v in startup_environment().items()),*argv]
    runner=runpy.run_path(str(Path(root)/'scripts/qualify_bigint_phase5.py'))
    return runner['container_stage'](host,image,output,argv,gpu=gpu,seconds=seconds,
                                     **({'readonly_inputs':readonly_inputs} if readonly_inputs else {}))


def original_authority(root, ref):
    from oeis_learn.generalization.qualification import consume_public, FACTS
    q=json.loads(verify_refs(root,[ref],max_bytes=32<<20)[0])
    if not isinstance(q,dict) or not FACTS<=set(q):
        raise ValueError('original closed qualification facts required')
    base=confined(root,ref['path']).parent
    consume_public(q,{k:q[k] for k in FACTS},root=base,
                   required_suites=['bigint-cpu','bigint-gpu'],
                   required_capabilities=['weight-load','finite-update','native-exact'])
    return q,base


def qualify_command(args, root, host_factory):
    from .progressive_commands import _output
    from .progressive_acceptance import hardware_plan
    from .generalization_qualification_io import current_facts, public_closure
    from oeis_learn.generalization.qualification import consume_public, FACTS
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    began=time.monotonic()
    blueprint=validate('blueprint',load(confined(root,args.blueprint)))
    output=_output(root,args.output)
    plan=named_plan(root,blueprint,'qualification_plan','progressive-qualification-plan/v1',
                    {'matrix','base_qualification','seconds'},optional={'original_cases'})
    if plan is None:
        return publish(output,operation='qualify',mechanical_status='BLOCKED',
                       reason='QUALIFICATION_PLAN_DEFICIT',missing=['qualification_plan']),2
    request=json.loads(verify_refs(root,[plan['matrix']])[0])
    originals=plan.get('original_cases',[])
    if not isinstance(originals,list) or len(originals)>48:
        raise ValueError('bounded original case inventory required')
    reuse=hardware_plan(root,request,originals)
    if plan['seconds']*1000000 > blueprint['payload']['proposed_caps']['elapsed_us']:
        raise ValueError('qualification exceeds prospective allowance')
    base,base_root=original_authority(root,plan['base_qualification'])
    host=host_factory(args)
    if request['source']!=digest(collect_snapshot(root)) or request['engine']!=host.engine.name or request['image']!=base['images'][1]:
        raise ValueError('current source/selected engine/immutable GPU image required')
    facts=current_facts(host,root,output/'inspection',images=base['images'],
                        source_ref=base['source_closure'],build_refs=base['build_dependencies'],producer_root=base_root,pac_seconds=max(1,min(120,plan['seconds']//6)))
    consume_public(base,facts,root=base_root,required_suites=['bigint-cpu','bigint-gpu'],
                   required_capabilities=['weight-load','finite-update','native-exact'])
    device=output/'device';device.mkdir()
    atomic_write(device/'request.json',request)
    copied=[]
    for i,(ref,raw) in enumerate(zip(originals,verify_refs(root,originals,max_bytes=4<<20))):
        path=device/'originals'/(str(i)+'.json');path.parent.mkdir(exist_ok=True)
        with path.open('xb') as stream: stream.write(raw)
        copied.append(reference(device,path))
    atomic_write(device/'originals.json',copied)
    # Every independent case has its own process, including the first batch.
    code=("from oeis_learn.generalization.progressive.contracts import load;"
          "from oeis_learn.pipeline_validation.progressive_acceptance import run_hardware_delta;"
          "r=load('/work/output/request.json');p=load('/work/output/originals.json');"
          "v=run_hardware_delta('/work/output',r,p,'/work/output/cases',seconds="+str(plan['seconds'])+");"
          "raise SystemExit(0 if v['status']=='PASS' else 1)")
    remaining=plan['seconds']-(time.monotonic()-began)
    if remaining<=0:
        raise TimeoutError('qualification allowance exhausted during inspection')
    stage(host,root,request['image'],device,['python','-c',code],gpu=True,seconds=remaining)
    if time.monotonic()-began>plan['seconds']:
        raise TimeoutError('qualification aggregate allowance exhausted')
    refs=[reference(output,p) for p in sorted((device/'cases').rglob('producer.json'))]
    if hardware_plan(output,request,refs)['delta']:
        raise ValueError('original selected device producers missing')
    # Keep every original byte/path from the base authority. New feature facts
    # are additional original artifacts, never a rewritten base PASS receipt.
    for ref in public_closure(base_root,base):
        raw=verify_refs(base_root,[ref],max_bytes=512<<20)[0]
        target=confined(output,ref['path'],exists=False)
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream: stream.write(raw)
    original=load(confined(output,base['producer']['receipt']['path']))
    caps=sorted(set(base['covered_capabilities'])|{'language:bigint-body/v1'}|
                {'model:'+digest({k:v for k,v in c['profile']['model'].items() if k!='seed'}) for c in request['cases']})
    atomic_write(output/'qualification-run.json',seal(dict(version='progressive-qualification-run/v1',
        matrix=request['identity'],original_execution_us=reuse['original_execution_us'],
        actual_elapsed_us=int((time.monotonic()-began)*1e6),
        reused=[v['case'] for v in reuse['reused']],executed=[c['identity'] for c in reuse['delta']])))
    artifacts=[*original['artifacts'],reference(output,device/'request.json'),reference(output,'qualification-run.json'),*refs]
    producer=seal(dict(original,capabilities=caps,artifacts=artifacts,
                       dependencies=[*original['dependencies'],base['producer']['receipt']]))
    atomic_write(output/'progressive-producer.json',producer)
    q=seal(dict(base,covered_capabilities=caps,
                producer=dict(receipt=reference(output,'progressive-producer.json'),status='PASS',artifacts=artifacts),
                artifact_inventory=public_closure(output,producer)))
    consume_public(q,facts,root=output,required_suites=['bigint-cpu','bigint-gpu'],required_capabilities=caps)
    validate_device_proofs(output,q,source=request['source'],image=request['image'],engine=request['engine'])
    atomic_write(output/'qualification.json',q)
    return publish(output,operation='qualify',mechanical_status='COMPLETE',qualification_status='QUALIFIED',
                   reason='ORIGINAL_SELECTED_DEVICE_QUALIFICATION',metrics={'qualification':reference(output,'qualification.json')}),0


def validate_device_proofs(root, authority, *, source, image, engine):
    """Match original model capabilities to the full exact selected case roster."""
    from .progressive_acceptance import hardware_plan
    payloads=verify_refs(root,authority['producer']['artifacts'],max_bytes=512<<20)
    documents=[]
    for ref,raw in zip(authority['producer']['artifacts'],payloads):
        try: value=json.loads(raw)
        except (ValueError,UnicodeError): continue
        if isinstance(value,dict):
            documents.append((ref,value))
    matrices=[v for _,v in documents if v.get('version')=='progressive-hip-matrix/v1']
    if len(matrices)!=1:
        raise ValueError('one selected original device matrix required')
    request=matrices[0]
    if (request['source'],request['image'],request['engine'])!=(source,image,engine):
        raise ValueError('device capability scope changed')
    refs=[r for r,v in documents if v.get('version')=='progressive-hip-case/v1']
    plan=hardware_plan(root,request,refs)
    if plan['delta']:
        raise ValueError('incomplete original device matrix')
    expected={'model:'+digest({k:v for k,v in c['profile']['model'].items() if k!='seed'}) for c in request['cases']}
    claimed={v for v in authority['covered_capabilities'] if v.startswith('model:')}
    if not claimed<=expected:
        raise ValueError('model capability absent from original device cases')
    return request, refs


def acceptance_command(args, root, host_factory):
    from .progressive_commands import _output
    from oeis_learn.generalization.progressive.campaign import validate_campaign
    began=time.monotonic()
    campaign=validate_campaign(load(confined(root,args.campaign),maximum=CAMPAIGN_DOCUMENT_BYTES))
    output=_output(root,args.output)
    plan=named_plan(root,campaign,'acceptance_plan','progressive-acceptance-plan/v1',
                    {'qualification','seconds'})
    if plan is None:
        return publish(output,operation='acceptance',mechanical_status='BLOCKED',
                       campaign=campaign['identity'],reason='ACCEPTANCE_PLAN_DEFICIT',missing=['acceptance_plan']),2
    if plan['seconds']<=30 or plan['seconds']*1000000>campaign['payload']['caps']['elapsed_us']:
        raise ValueError('acceptance exceeds selected campaign allowance')
    q,_=original_authority(root,plan['qualification'])
    # CPU/native execution precedes any device execution. The original runtime
    # and qualification are inspected before the device batch can be admitted.
    host=host_factory(args)
    cpu=output/'software'
    code="from pathlib import Path;from oeis_learn.pipeline_validation.progressive_acceptance import run_suite; r=run_suite(Path('/opt/oeis-build'),Path('/work/output/suite')); raise SystemExit(0 if r['status']=='COMPLETE' else 1)"
    stage(host,root,q['images'][0],cpu,['python','-c',code],gpu=False,seconds=min(1200,plan['seconds']-30))
    receipt=load(cpu/'suite/acceptance.json')
    if receipt['status']!='COMPLETE' or receipt['qualified'] is not False:
        raise ValueError('original diagnostic CPU/native acceptance required')
    # The feature's device campaign is a separate original producer, never
    # inferred from Q's model-shape coverage or a CPU suite PASS.
    from .progressive_device_acceptance import run_campaign
    return run_campaign(args,root,host,campaign,q,plan,output,receipt,began=began)
