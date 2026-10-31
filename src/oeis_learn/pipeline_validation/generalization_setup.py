"""Host-only construction of a frozen, reduced acceptance session."""
import copy
from pathlib import Path
from .identity import seal,write,confined
from .generalization_preparation import reference
from .generalization_inputs import _fresh,_json,assemble_study,assemble_session
from oeis_learn.generalization.legacy import verify_legacy
from oeis_learn.generalization.bootstrap import pending
from oeis_learn.generalization.acceptance_inputs import mechanical_inputs,small_recipe,DIAGNOSTIC


def assemble_acceptance(workspace,*,imported,images,output):
    workspace=Path(workspace);target=_fresh(workspace,output)
    root=confined(workspace,imported)
    roster=_json(root/'import.json')
    if seal(roster)!=roster or roster.get('version')!='generalization-legacy-roster/v1' or len(roster['checkpoints'])!=2:
        raise ValueError('verified original uniform import required')
    descriptors=[_json(root/(label+'-checkpoint.json')) for label in ('best','latest')]
    if descriptors!=roster['checkpoints']:raise ValueError('import roster differs from original checkpoints')
    recipes=[verify_legacy(root,c)['recipe'] for c in descriptors]
    from oeis_learn.generalization.contracts import verify_artifact
    for c in descriptors:verify_artifact(root,c['weights'])
    if recipes[0]!=recipes[1]:raise ValueError('retained recipes differ')
    fixture=mechanical_inputs()
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    snapshot=collect_snapshot(workspace)
    target.mkdir(parents=True)
    def put(name,value):
        path=target/'common'/name;write(path,value);return str(path.relative_to(workspace))
    recipe=put('recipe.json',recipes[0]);source=put('source-closure.json',snapshot)
    ds=put('descriptors.json',fixture['descriptors']);exclusions=put('exclusions.json',[])
    put('mechanical-policy.json',fixture)
    shared=[]
    for cell in fixture['cohorts']:
        shared.append(dict(name=cell['name'],role=cell['role'],holdout=cell['holdout'],count=1,quotas={},structural=False,candidates=put(cell['name']+'.json',cell['rows'])))
    static=dict(name='static',role='training',holdout='parameter',count=4,quotas={},structural=False,refresh=dict(seed=fixture['seed'],capacity=256,cursor=0,draws=32))
    fresh=dict(name='fresh',role='training',holdout='parameter',count=16,quotas={},structural=False,refresh=dict(seed=fixture['seed'],capacity=256,cursor=32,draws=32))
    cfg=dict(version='generalization-input-blueprint/v1',stage='E0',purpose='acceptance',hypothesis='Bounded mechanical software acceptance; no research claim',
        recipe=recipe,source_closure=source,images=images,initializer=None,descriptors=ds,exclusions=exclusions,cells=[static,*copy.deepcopy(shared)],training={},
        evaluation=dict(role='selection',cells=['selection-parameter','selection-structural'],checkpoints=[str((root/(label+'-checkpoint.json')).relative_to(workspace)) for label in ('best','latest')],seconds=360,bank_seeds=None),
        replay=None,seeds=[fixture['seed']],budgets=dict(target_tokens=0,seconds=30,draws=1000,memory_bytes=8<<30,storage_bytes=1<<30,log_bytes=1<<20,attempts=8),milestones=[],
        storage=dict(study_bytes=4<<30,checkpoint_bytes=2<<30,shard_bytes=256<<20,raw_bytes=256<<20,sharing_bytes=32<<20),sharing_policy='explicit')
    results={}
    def assemble(name,config):
        put(name+'-blueprint.json',config)
        result=assemble_study(workspace,config,str((target/name).relative_to(workspace)));results[name]=result['manifest'];return result
    e0=assemble('retained-eval',cfg);e0study=_json(workspace/e0['manifest'])
    cfg.update(stage='P1-screen',recipe=put('fresh-fixture-recipe.json',small_recipe(recipes[0])),cells=[static,fresh,*copy.deepcopy(shared)],evaluation=None)
    cfg['budgets']['target_tokens']=256;cfg['milestones']=[128,256]
    cfg['storage'].update(study_bytes=3<<30,checkpoint_bytes=2<<30)
    cfg['training']={'static':dict(cells=['static'],data_mode='static',batch_size=4),'refreshed':dict(cells=['static','fresh'],data_mode='refresh',batch_size=4)}
    assemble('refresh',cfg)
    cfg.update(stage='P3',cells=[static,*copy.deepcopy(shared)],training={a:dict(cells=['static'],data_mode='static',batch_size=4) for a in ('literal','parameterized')})
    assemble('parameter-targets',cfg)
    cfg.update(stage='P4',recipe=recipe,initializer=pending(e0study['identity']),training={a:dict(cells=['static'],data_mode='static',batch_size=4) for a in ('continued_sft','verified_replay')},
        evaluation=dict(role='selection',cells=['selection-parameter','selection-structural'],checkpoints=[],seconds=30,bank_seeds=None),
        replay=dict(cell='static',seconds=30,proposal_policy=DIAGNOSTIC))
    cfg['storage'].update(study_bytes=30<<30,checkpoint_bytes=26<<30)
    cfg['milestones']=[256]
    assemble('replay-resume',cfg)
    session=assemble_session(workspace,results,str((target/'generalization-acceptance.json').relative_to(workspace)))
    result=seal(dict(version='generalization-acceptance-setup/v1',status='ASSEMBLED',session=session,images=images,source_closure=reference(workspace/source,workspace),
        imported=roster['identity'],initializer='DEFERRED_TO_NEW_RETAINED_SELECTION',native_admission='NOT_RUN',qualification='NOT_RUN',scientific_status='UNMEASURED'))
    write(target/'setup.json',result)
    return result
