"""Installed custodian assembly from reviewed originals; no learning occurs here."""
import json,time
from pathlib import Path
from .identity import seal,digest,write
from .generalization_preparation import reference
from oeis_learn.generalization.contracts import verify_artifact


def envelope(*,purpose,arms):
    """Conservative initial operational reservations; calibration may only raise."""
    from .generalization_execution import POLICY_ID,PAC_STAGES
    from oeis_learn.generalization.budgets import pac_work_allowance
    work={name:7200 for name in PAC_STAGES}
    for kind in ('preparation','learner','generation'):
        work[kind]=pac_work_allowance(kind,forecast=None,startup=1800,publication=1800,existing=0)['seconds']
    work.update(resume=work['learner'],acceptance=36000,calibration=18000,export=7200,validation=7200,proof=7200)
    if purpose=='acceptance':work['learner']=work['resume']=1800
    stages={name:dict(work=work[name],startup=1800,publication=1800 if name not in {'learner','resume'} or purpose=='research' else 300,
        projection=1800,cleanup=30) for name in PAC_STAGES}
    for stage in stages.values():stage['outer']=sum(stage.values())
    total=sum(s['outer'] for s in stages.values())+(len(arms)-1)*sum(stages[k]['outer'] for k in ('learner','generation','resume','proof','export'))+3600
    return dict(version='pac-envelope/v1',**stages,campaign=total,diagnostics_seconds=7200 if purpose=='research' else 1200,
        batch_size=32 if purpose=='research' else 2,acceptance_tokens=1000000,backend=POLICY_ID,
        storage=dict(data_bytes=16<<30,data_copies=3,scratch_bytes=8<<30,logs_bytes=2<<30,package_bytes=512<<20,reserved_bytes=128<<30))


def assemble(inputs,output,settings,*,seconds):
    from oeis_learn.generalization import structural
    from oeis_learn.generalization.refresh import RefreshPlan
    from oeis_learn.generalization.experiments import pac_recipe
    from oeis_learn.generalization.execution import pac_control
    from oeis_learn.synthesis.client import SynthesisClient
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from oeis_learn.fresh.bigint_worker import models
    from .generalization_curriculum import initial_weights_identity
    from .generalization_execution import POLICY_ID,bind_pac,pac_budget_projection
    from .generalization_preparation import prepare_pac_training,prepare_pac
    if set(settings)!={'context','images','engine','research','arms','seed','source'}:raise ValueError('closed PAC assembly request')
    context=settings['context'];root=inputs['mechanical_sources'];output=Path(output)
    if settings['source']!=digest(source_snapshot()):raise ValueError('installed assembly source mismatch')
    originals=context['originals']
    if sorted(({k:v for k,v in x.items() if k!='ref'} for x in originals),key=lambda x:x['name'])!=sorted(structural.original_cells(),key=lambda x:x['name']):raise ValueError('all original cells required')
    receipt=json.loads(verify_artifact(root,context['preparation'],max_bytes=16<<20))
    if receipt['identity']!='sha256:56955fda4ffba7eda116fbcfc26b174e3b0d381130db1185d1d8149b13073b7f' or receipt['status']!='PASS' or seal(receipt)!=receipt:raise ValueError('original reviewed preparation required')
    parent=Path(context['preparation']['path']).parent
    indexed={str(parent/r['path']):dict(r,path=str(parent/r['path'])) for r in receipt['inventory']}
    for item in originals:
        if indexed.get(item['ref']['path'])!=item['ref']:raise ValueError('original cell differs from reviewed producer inventory')
    recipe=json.loads(verify_artifact(root,context['recipe_ref'],max_bytes=1<<20))
    descriptors=json.loads(verify_artifact(root,context['descriptors'],max_bytes=1<<20))
    indices=list(map(str,range(100)))
    def normalize(row):
        return dict(id=row['id'],role='training',family=row['family'],source=row['source'],
            parameter_source=row['parameter_source'],assignment=row['assignment'],indices=row.get('indices',indices))
    raw=json.loads(verify_artifact(root,next(x['ref'] for x in originals if x['name']=='static'),max_bytes=2<<30))
    rows=[normalize(r) for r in raw]
    seed=settings['seed'];mode='fresh-control' if 'A' in settings['arms'] else 'reference-only'
    scientific=pac_recipe(control_mode=mode,optional_d='D' in settings['arms'])
    if settings['arms']!=scientific['arms']:raise ValueError('frozen arm order differs from canonical policy')
    draws,_=RefreshPlan(descriptors,seed=int(digest(['pac-probes/v1',seed])[7:15],16),capacity=8192).take(0,8192)
    probes=[normalize(r) for r in draws]
    output.mkdir(parents=True,exist_ok=False)
    write(output/'recipe.json',recipe);write(output/'source.json',source_snapshot())
    write(output/'rows.json',rows);write(output/'probes.json',probes)
    limits=envelope(purpose='research',arms=settings['arms'])
    deadline=time.monotonic()+seconds
    with SynthesisClient(digest(['pac-native-admission',settings['source']]),cache_root=output/'private/native') as native:
        def execute(source,indices):return native.execute(source,indices,deadline=min(deadline,time.monotonic()+4))
        # These deterministic commitments are computed before any model outcome.
        final=[json.loads(verify_artifact(root,x['ref'],max_bytes=2<<30)) for x in originals if x['role']!='training']
        pool=prepare_pac_training(rows,probes,output=output/'private/pool',execute=execute,final_exclusions=final,seed=seed,batch_size=32,purpose='research')
        pool_root=output/'private/pool'
        inventory=json.loads(verify_artifact(pool_root,pool['inventory']))
        literal=[r for ref in pool['literal'] for r in json.loads(verify_artifact(pool_root,ref))]
        targets=[r for ref in pool['parameterized'] for r in json.loads(verify_artifact(pool_root,ref))]
        configured=json.loads(json.dumps(recipe));configured['model']['seed']=seed
        encoder,decoder,_=models(configured,'cpu');initializer=initial_weights_identity(encoder,decoder)
        del encoder,decoder
        expected=dict(initializer=initializer,order=digest([r['id'] for r in literal]),model=digest({k:v for k,v in recipe['model'].items() if k!='seed'}),
            optimizer=digest(recipe['learning']),exposure=digest(dict(target_tokens=25000000,batch_size=32)),diagnostics=digest('pac-common-monitoring/v1'),backend=POLICY_ID)
        control=pac_control({},expected,original_study=context['study']['identity'],source=digest(context['study']),image=context['runtime']['images'][1],
            checkpoint=context['checkpoints']['static']['checkpoint']['identity'],historical_measurements=None)
        profiles={arm:dict(initializer=initializer,initializer_kind='random',pool=inventory['sidecar'],targets=digest(literal if arm=='A' else targets),
            model=expected['model'],optimizer=expected['optimizer'],backend=POLICY_ID,order=expected['order'],token_cap=25000000,
            sampling=scientific['sampling'][arm],exposure=expected['exposure'],diagnostics=expected['diagnostics']) for arm in settings['arms']}
        experiment=dict(version='PAC-screen/v1',arms=settings['arms'],comparison_factors=scientific['comparison_factors'],seed=seed,initializer=initializer,
            initializer_kind='random',pool=inventory['sidecar'],probes=pool['probes']['sha256'],targets=digest(targets),literal_targets=digest(literal),inventory=inventory['identity'],
            policy='pac-exp3s-leitner/v1',budget_graph=pac_budget_projection(limits,arms=settings['arms']),metrics=['certified_family_macro'],
            control=control['identity'],control_mode=mode,frozen=True,outcomes_seen=False,target_tokens=25000000,
            model=expected['model'],optimizer=expected['optimizer'],backend=POLICY_ID,order=expected['order'],source=settings['source'],exposure=expected['exposure'],diagnostics=expected['diagnostics'])
        plan=dict(version='pac-preparation/v1',originals=originals,rows=reference(output/'rows.json',output),probes=reference(output/'probes.json',output),
            protocol=structural.design(structural.prior_protocol()),seed=seed,batch_size=32)
        # Supply original references from their projected immutable roots.
        import shutil
        for item in originals:
            dest=output/item['ref']['path'];dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(Path(root)/item['ref']['path'],dest)
        runtime=dict(version='pac-runtime/v1',study='',purpose='research',recipe=reference(output/'recipe.json',output),source=reference(output/'source.json',output),
            images=settings['images'],control=control,profiles=profiles,preparation=plan,prepared=None,envelope=limits,feature=None,engine=settings['engine'])
        experiment,runtime=bind_pac(experiment,runtime)
        result=prepare_pac(output,experiment,runtime,output=output/'resolved',execute=execute,seconds=max(.001,deadline-time.monotonic()))
    write(output/'context.json',context)
    return dict(result,manifest='resolved/study.json',runtime='resolved/runtime.json',context='context.json')


def acceptance(inputs,output,settings,*,seconds):
    from .generalization_acceptance import pac_acceptance_fixture,pac_acceptance
    from oeis_learn.synthesis.client import SynthesisClient
    if set(settings)!={'recipe','source','environment'}:raise ValueError('closed campaign acceptance request')
    deadline=time.monotonic()+seconds
    with SynthesisClient(settings['source'],cache_root=Path(output)/'private/native') as native:
        config=pac_acceptance_fixture(Path(output)/'private/inputs',execute=lambda s,i:native.execute(s,i,deadline=min(deadline,time.monotonic()+4)),
            device='cuda',recipe=settings['recipe'],source=settings['source'],environment=settings['environment'])
    return pac_acceptance({'training_targets':Path(output)/'private/inputs'},Path(output)/'acceptance',dict(config=config),seconds=max(.001,deadline-time.monotonic()))
