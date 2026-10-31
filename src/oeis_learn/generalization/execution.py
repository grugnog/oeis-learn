"""Frozen execution policy, independent of containers and tensor dependencies.

Only preparation reference resolution may change the execution binding, producing
new study/runtime identities. Resealing a sidecar alone cannot change an experiment.
"""
import copy
from collections import Counter
from oeis_learn.pipeline_validation.identity import digest,seal

RUNTIME_FIELDS={'version','identity','study','purpose','recipe','images','preparation',
                'training','evaluation','replay','assessment','feature','storage'}
PAIRS={'P1-screen':('static','refreshed'),'P1-confirm':('static','refreshed'),
       'P2':('baseline_support','expanded_support'),'P3':('literal','parameterized'),
       'P4':('continued_sft','verified_replay')}
RECIPES={'E0':'E0-evaluation-extension/v1','P0':'P0-uniform-replication/v1',
         'P1-screen':'P1-refresh-screen/v1','P1-confirm':'P1-refresh-confirm/v1',
         'P2':'P2-support/v1','P3':'P3-targets/v1','P4':'P4-replay/v1'}
FACTORS={'E0':'evaluation','P0':'seed','P1-screen':'refresh','P1-confirm':'refresh',
         'P2':'support','P3':'target_format','P4':'verified_replay'}
CASES={'retained-eval':'E0','refresh':'P1-screen','parameter-targets':'P3','replay-resume':'P4'}


def binding(runtime):
    # Preparation binds the proposed study, which binds these execution choices.
    # Excluding only that link and study avoids a circular content hash.
    return digest({k:v for k,v in runtime.items() if k not in {'identity','study','preparation','version'}})


def bind(study,runtime):
    study=copy.deepcopy(study);runtime=copy.deepcopy(runtime)
    runtime['version']='generalization-runtime/v2'
    study.update(version='generalization-study/v3',execution=binding(runtime))
    study=seal(study);runtime=seal(dict(runtime,study=study['identity']))
    return study,runtime


def verify_binding(study,runtime):
    if (study.get('version')!='generalization-study/v3' or runtime.get('version')!='generalization-runtime/v2'
        or set(runtime)!=RUNTIME_FIELDS or seal(study)!=study or seal(runtime)!=runtime
        or runtime['study']!=study['identity'] or study['execution']!=binding(runtime)):
        raise ValueError('execution binding requires intact study v3/runtime v2; reassemble older inputs')


def validate_policy(study,runtime,*,case=None):
    stage=study['progression']['stage']
    if study['recipe']!=RECIPES[stage] or study['changing_factors']!=[FACTORS[stage]]:
        raise ValueError('registered recipe and single changing factor required')
    if case is not None and (case not in CASES or stage!=CASES[case]):
        raise ValueError('acceptance case differs from registered experiment')
    training=runtime['training'];tokens=study['budgets']['target_tokens'];milestones=study['checkpoints']['milestones']
    if type(tokens) is not int or tokens<0 or milestones!=sorted(set(milestones)) or any(type(x) is not int or not 0<x<=tokens for x in milestones):
        raise ValueError('bounded increasing checkpoint exposures required')
    if stage=='E0':
        if training or tokens!=0 or milestones or runtime['replay'] is not None or runtime['evaluation'] is None:
            raise ValueError('E0 is evaluation-only with zero learner tokens and milestones')
    else:
        if not tokens or not milestones:raise ValueError('learning requires positive tokens and milestones')
        if stage=='P0':raise ValueError('P0 uses canonical historical pilot-followup --arms uniform')
        if set(study['arms'])!=set(PAIRS[stage]) or set(training)!=set(PAIRS[stage]):
            raise ValueError('registered complete paired arms required')
        views=list(training.values())
        if len({v['batch_size'] for v in views})!=1:raise ValueError('paired batch sizes must match')
        if stage.startswith('P1'):
            if training['static']['data_mode']!='static' or training['refreshed']['data_mode']!='refresh':
                raise ValueError('registered static/refreshed modes required')
        elif any(v['data_mode']!='static' for v in views):raise ValueError('unregistered refresh interaction')
    if stage!='P4' and runtime['replay'] is not None:raise ValueError('unregistered replay interaction')
    if stage=='P4' and (study['model']['initialization']=='random' or runtime['replay'] is None or runtime['evaluation'] is None or runtime['evaluation']['role']!='selection'):
        raise ValueError('P4 requires retained initializer, replay tasks and selection evaluation')
    if study['data']['target_format']!='literal':raise ValueError('paired targets use literal authority and separately verified P3 views')
    if study['data']['sharing_policy'] not in {'explicit','distinct-occurrences'}:raise ValueError('registered parameter sharing required')
    if runtime['purpose']=='research' and stage in {'P2','P3','P4'} and study['model']['initialization']=='random':
        raise ValueError('research comparison requires selection-designated incumbent')
    if runtime['purpose']=='acceptance':
        if len(study['seeds'])!=1 or tokens>4096 or study['budgets']['seconds']>900:
            raise ValueError('reduced single-seed acceptance required')
        view=runtime['evaluation']
        if view is not None and (view['role']!='selection' or runtime['assessment'] is not None):
            raise ValueError('acceptance uses selection fixtures, never assessment exposure')
    if case=='retained-eval' and not runtime['evaluation']['checkpoints']:
        raise ValueError('retained evaluation needs actual checkpoint descriptors')


def validate_refresh(static,refreshed):
    """Fixed family proportions and genuinely distinct numeric programs."""
    for rows in (static,refreshed):
        if not rows or len({r['id'] for r in rows})!=len(rows) or len({r['source'] for r in rows})!=len(rows):
            raise ValueError('empty or duplicate refresh task/source')
    if not {r['id'] for r in static}<{r['id'] for r in refreshed}:
        raise ValueError('refresh must extend static variants')
    a=Counter(r['family'] for r in static);b=Counter(r['family'] for r in refreshed)
    if set(a)!=set(b) or any(a[k]*len(refreshed)!=b[k]*len(static) for k in a):
        raise ValueError('refresh must preserve fixed family mixture')


def verify_resolution(original,resolved):
    """Preparation may resolve cell paths, never alter the registered intervention."""
    def choices(runtime):
        value=copy.deepcopy(runtime)
        for key in ('identity','study','preparation','version'):value.pop(key,None)
        for view in value['training'].values():
            view.pop('cells',None);view.pop('shards',None)
        if value['evaluation'] is not None:
            for key in ('cells','prompts','targets'):value['evaluation'].pop(key,None)
        if value['replay'] is not None:
            value['replay'].pop('cell',None);value['replay'].pop('tasks',None)
        return value
    if choices(original)!=choices(resolved):raise ValueError('preparation resolution changed execution choices')
