"""Concrete tiny acceptance inputs. No historical or research producers invented."""
from pathlib import Path

from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, record, reference, seal


def profile():
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.models import resolve_profile
    recipe=resolve(dict(version='bigint-bootstrap/v1'))
    return resolve_profile(recipe,shape='smaller_width',seed=13,dropout=.1)


def assets(output):
    """Installed CPU image only: measure real initial weights for diagnostic freeze."""
    import torch
    from oeis_learn.generalization.progressive.models import construct
    output=Path(output)
    p=profile(); encoder,decoder,counts=construct(p,device='cpu')
    torch.save(dict(encoder=encoder.state_dict(),decoder=decoder.state_dict()),output/'initial.pt')
    atomic_write(output/'profile.json',p)
    atomic_write(output/'recipe.json',p['base_recipe'])
    atomic_write(output/'measurements.json',counts)
    from .progressive_acceptance import fixture
    fixture(output/'mechanics',interrupt=False)


def qualification_inputs(root, folder, *, base, matrix):
    """Build the closed plan consumed by the canonical discovery qualify command."""
    folder=Path(folder)
    plan=seal(dict(version='progressive-qualification-plan/v1',matrix=reference(root,matrix),
                   base_qualification=reference(root,Path(base)/'qualification.json'),seconds=1800))
    atomic_write(folder/'qualification-plan.json',plan)
    from oeis_learn.generalization.progressive.budgets import KEYS
    caps=dict.fromkeys(KEYS,0); caps.update(elapsed_us=43200*1000000,disk_bytes=64<<30,archive_bytes=512<<20)
    b=record('blueprint',dict(executable=False,selected_stages=['capability'],
        references=[dict(name='qualification_plan',role='custodian',kind='policy',artifact=reference(root,folder/'qualification-plan.json'))],
        unresolved=[],proposed_caps=caps,reference_campaign=str(folder.relative_to(root))),
        created_at='2026-11-10T00:00:00Z',purpose='acceptance',track='synthetic_strict')
    atomic_write(folder/'qualification-blueprint.json',b)
    return folder/'qualification-blueprint.json'


def preparation_inputs(root, folder, *, qualification, assets_root):
    """Real qualification + measured tiny tensors; freeze owns final authority.

    The outer campaign reserves the integrated feature rehearsal. Its named
    packaging task does not substitute for the actual installed lifecycle cases.
    """
    from oeis_learn.generalization.progressive.targets import defaults
    from oeis_learn.generalization.progressive.freeze import learning_compatibility
    from oeis_learn.pipeline_validation.progressive_roles import ROLE_POLICY
    from oeis_learn.generalization.progressive.budgets import KEYS
    folder=Path(folder); assets_root=Path(assets_root); qualification=Path(qualification)
    p=load(assets_root/'profile.json'); recipe=p['base_recipe']; q=load(qualification/'qualification.json')
    def envelope(kind,payload):
        return record(kind,payload,created_at='2026-11-10T00:00:00Z',purpose='acceptance',track='synthetic_strict')
    def put(name,value):
        atomic_write(folder/name,value); return reference(root,folder/name)
    scope=dict(source=q['source_closure']['sha256'],images=q['images'],
        model=digest({k:v for k,v in p['model'].items() if k!='seed'}),language='bigint-body/v1',
        precision='fp32',device='hip',execution_policy=q['runtime']['execution_policy'])
    runtime=envelope('runtime',dict(scope=scope,qualification=[reference(root,qualification/'qualification.json')],
        operations=['publish'],role_policy=ROLE_POLICY,fixture=False))
    runtime_ref=put('runtime-input.json',runtime)
    targets=defaults(); targets['payload']['selected']=True; targets=seal(targets)
    refs=[]
    def add(name,kind,ref,role='custodian'):
        refs.append(dict(name=name,role=role,kind=kind,artifact=ref)); return ref
    target=add('targets','policy',put('targets.json',targets))
    add('recipe','recipe',reference(root,assets_root/'recipe.json'))
    checkpoint=add('measured-initial-weights','checkpoint',reference(root,assets_root/'initial.pt'))
    add('source','source',reference(root,qualification/q['source_closure']['path']))
    add('images','image',put('images.json',dict(version='discovery-images/v1',images=q['images'])))
    add('qualification','qualification',reference(root,qualification/'qualification.json'))
    add('groups','groups',put('exposure.json',envelope('exposure',dict(previous=None,
        detector_limits=['sealed diagnostic fixtures; not a research cohort'],events=[]))))
    add('fixture-inventory','prepared',put('fixture.json',dict(version='progressive-workstation-fixture/v1',
        profile=p['identity'],scope='DIAGNOSTIC_ONLY',autonomous_discoveries=0)))
    channel=load(assets_root/'mechanics/training-channel.json')
    channel=seal(dict(channel,data=dict(channel['data'],artifact=reference(root,assets_root/'mechanics/training-input.json'))))
    add('training-fixture','prepared',put('training-channel.json',channel),role='learner')
    add('acceptance_plan','policy',put('acceptance-plan.json',seal(dict(version='progressive-acceptance-plan/v1',
        qualification=reference(root,qualification/'qualification.json'),seconds=3600))))
    caps=dict.fromkeys(KEYS,0); caps.update(tokens=1000000,draws=100000,rejections=100000,attempts=10000,
        native_work=100000000,elapsed_us=43200*1000000,disk_bytes=64<<30,archive_bytes=512<<20)
    child=dict.fromkeys(KEYS,0); child.update(elapsed_us=10000000,disk_bytes=1<<30,archive_bytes=1<<20)
    graph=dict(selected_stages=['capability'],nodes=[dict(id='capability',stage='capability',kind='routine',
        dependencies=[],maximum_visits=1,caps=child)],edges=[],diagnostic_pool=dict.fromkeys(KEYS,0))
    task=dict(id='diagnostic-publication',stage='capability',kind='packaging',arm='diagnostic',seed=13,
        initializer=dict(mode='random',artifact=None,compatibility=channel['data']['compatibility']),
        token_target=0,caps=child,startup_us=1000,finalization_us=1000,policy=targets['identity'],slot='capability',
        output=str((folder/'publication').relative_to(root)))
    payload=dict(runtime=runtime_ref,graph=graph,task_plans=[task],references=refs,caps=caps,
        checkpoint=checkpoint,milestone_pins=0,shard_bytes=0,package_bytes=512<<20,required_inodes=65536,
        target_policy=target,deficits=[])
    prep=put('preparation-plan.json',dict(version='progressive-preparation-plan/v1',payload=payload))
    blueprint=envelope('blueprint',dict(executable=False,selected_stages=['capability'],
        references=[dict(name='preparation_plan',role='custodian',kind='policy',artifact=prep)],
        unresolved=[],proposed_caps=caps,reference_campaign=str(folder.relative_to(root))))
    atomic_write(folder/'blueprint.json',blueprint)
    return folder/'blueprint.json'
