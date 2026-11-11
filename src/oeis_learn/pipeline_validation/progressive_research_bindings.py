"""Bind registered private research materials to the finite role construction.

No model, optimizer, native service or scoring outcome is loaded by this stdlib
custodian. Original artifacts are hash checked; canonical freeze remains the
execution authority.
"""
from pathlib import Path
import json

from .identity import confined
from .progressive_research_inputs import checked_file, _json
from .progressive_research_design import scientific_round
from oeis_learn.generalization.progressive.contracts import (
    atomic_write, digest, load, reference, record, seal, validate,
)
from oeis_learn.generalization.progressive.research_roster import construct


def inputs(root, data, design, materials):
    if design!=scientific_round(data):
        raise ValueError('original registered scientific design required')
    if (seal(materials)!=materials or materials.get('version')!='progressive-research-materials/v1'
            or materials['status']!='COMPLETE' or materials['diagnostic'] or materials['deficits']):
        raise ValueError('complete original native research materials required')
    census=_json(checked_file(root,materials['census']),maximum=32<<20)
    request=_json(checked_file(root,materials['request']),maximum=32<<20)
    from .progressive_research_bridge import request as expected_request
    if (request!=expected_request(data) or census['request']!=request['identity']
            or seal(census)!=census or census['diagnostic'] or census['native']!=materials['native']):
        raise ValueError('research materials differ from original registered census')
    base=confined(root,materials['census']['path']).parent
    candidates=[r for r in census['artifacts'] if Path(r['path']).name=='panel-reservations.json']
    if len(candidates)!=1:
        raise ValueError('one original private panel reservation required')
    reservations=_json(checked_file(base,candidates[0]),maximum=128<<20)
    panels={};rosters={}
    for item in materials['panels']:
        name=item['panel']
        if name in panels or item['status']!='COMPLETE':
            raise ValueError('distinct complete original cohort publications required')
        panel=validate('cohort',_json(checked_file(root,item['cohort']),maximum=32<<20))
        if panel['purpose']!='research' or panel['payload']['panel']!=name or panel['payload']['role']!=item['role']:
            raise ValueError('original panel publication scope differs')
        original={r['id']:r for r in reservations[name]}
        if set(original)!={r['id'] for r in panel['payload']['rows']}:
            raise ValueError('original reserved panel membership differs')
        selected=[]
        for r in panel['payload']['rows']:
            source=original[r['id']]
            if source['group'] not in r['groups']:
                raise ValueError('original component assignment missing from published cohort')
            selected.append(dict(id=r['id'],group=source['group'],stratum=r['stratum']))
        panels[name]=panel;rosters[name]=selected
    screening=_json(checked_file(root,materials['screening']))
    roster=construct(design,rosters,screening)
    support=_json(checked_file(root,materials['support']),maximum=256<<20)
    summary=_json(checked_file(root,materials['support_summary']))
    exposure=validate('exposure',_json(checked_file(root,materials['exposure'])))
    if any(p['payload']['exposure']!=exposure['identity'] for p in panels.values()):
        raise ValueError('published cohort exposure registry differs')
    return dict(roster=roster,panels=panels,reservations=reservations,screening=screening,
                support=support,support_summary=summary,exposure=exposure)


def publish_roster(root, output, data, design, materials):
    bound=inputs(root,data,design,materials)
    output=Path(output)
    atomic_write(output/'private/role-roster.json',bound['roster'])
    from collections import Counter
    rows=bound['roster']['tasks']
    value=seal(dict(version='progressive-research-roster-publication/v1',
        status='COMPLETE',executable=False,design=design['identity'],materials=materials['identity'],
        roster=reference(root,output/'private/role-roster.json'),
        steps=len(rows),by_stage=dict(Counter(r['stage'] for r in rows)),
        training_tasks=sum(r['operation']=='train' for r in rows),
        screening_trials=16,screening_looks=3,confirmation_positions=4,
        assessment_exposed=False,missing_execution_authorities=[
            'current selected-model qualification','measured evaluation feasibility',
            'exact-input calibration','canonical campaign freeze','scoped installed acceptance']))
    atomic_write(output/'roster-summary.json',value)
    return value


class Compiler:
    """Write closed role requests and channels; never execute their contents."""
    def __init__(self,root,output,*,qualification,recipe,checkpoint,created_at):
        self.root=Path(root);self.output=Path(output);self.created_at=created_at
        self.qroot=Path(qualification);self.q=load(self.qroot/'qualification.json')
        self.qref=reference(root,self.qroot/'qualification.json')
        self.recipe=recipe;self.checkpoint=checkpoint
        self.refs=[];self.plans=[];self.runtimes={};self.channels={}

    def put(self,name,value,*,kind=None,role='custodian'):
        path=self.output/name
        atomic_write(path,value)
        ref=reference(self.root,path)
        if kind:self.refs.append(dict(name=name.removesuffix('.json'),kind=kind,role=role,artifact=ref))
        return ref

    def channel(self,name,kind,role,data):
        from oeis_learn.generalization.progressive.projections import _validate_channel
        _validate_channel(kind,data)
        value=seal(dict(version='discovery-channel/v1',kind=kind,role=role,data=data))
        key=value['identity']
        if key not in self.channels:
            self.channels[key]=self.put('private/channels/'+name+'.json',value,kind='prepared',role=role)['path']
        return self.channels[key]

    def runtime(self,profile):
        from .progressive_roles import ROLE_POLICY
        key=digest({k:v for k,v in profile['model'].items() if k!='seed'})
        if key not in self.runtimes:
            value=record('runtime',dict(scope=dict(source=self.q['source_closure']['sha256'],
                images=self.q['images'],model=key,language='bigint-body/v1',precision='fp32',
                device='hip',execution_policy=self.q['runtime']['execution_policy']),
                qualification=[self.qref],operations=['census','audit-language','train','generate','score','propose-trial','publish'],
                role_policy=ROLE_POLICY,fixture=False),created_at=self.created_at,purpose='research',track='synthetic_strict')
            self.runtimes[key]=self.put('runtimes/'+key[7:]+'.json',value,kind='qualification')
        return self.runtimes[key]

    def task(self,row,channels,*,seed,compatibility,study,parameters,scopes,maximum_update,
             checkpoint_bytes):
        from oeis_learn.generalization.progressive.budgets import KEYS
        operation='publish' if row['operation'].startswith('publish-') else row['operation']
        trial=dict(row['trial'],study=study['identity']) if row['trial'] else None
        shared=trial and trial['phase'] not in {'selection','confirmation'}
        if shared:trial['shared_finalization_us']=1800*1000000
        finalization=1 if shared else 1800
        elapsed=row['seconds']*1000000
        caps=dict.fromkeys(KEYS,0)
        caps.update(elapsed_us=elapsed,disk_bytes=(2*checkpoint_bytes+(128<<20)) if operation=='train' else checkpoint_bytes+(128<<20),
                    archive_bytes=16<<20)
        if operation=='train':
            target=row['token_target'];before=0 if row['parent'] is None else (1000000 if target==2000000 else 2000000)
            caps['tokens']=target-before+maximum_update
        # Include per-member native validation of every retained selection.
        members=len(row.get('members',[]))
        attempts=members*(48 if row.get('all_methods') else 8)
        caps.update(draws=attempts,rejections=attempts,attempts=attempts,
                    native_work=max(1000000,members*8*100*100000))
        reservation=dict.fromkeys(KEYS,0)
        reservation.update(elapsed_us=elapsed,draws=attempts,rejections=attempts,attempts=attempts,native_work=caps['native_work'])
        image=self.q['images'][1 if operation in {'train','generate'} else 0]
        request=dict(operation=operation,artifacts=channels,image=image,seconds=row['seconds'],reservation=reservation)
        policy=self.put('requests/'+row['id']+'.json',request,kind='policy')['sha256']
        init=dict(mode='full_state' if row.get('parent') else 'random',artifact=None,compatibility=compatibility)
        if row.get('parent'):init['parent_task']=row['parent']
        task=dict(id=row['id'],stage=row['stage'],kind='training' if operation=='train' else 'evaluation',
            arm=row['arm'],seed=seed,initializer=init,token_target=row.get('token_target',0),caps=caps,
            startup_us=1000000,finalization_us=finalization*1000000,policy=policy,slot=row['stage'],
            output=str((self.output/'execution'/row['id']).relative_to(self.root)))
        if trial:task['trial']=trial
        if parameters:
            task['trial_choices']=[dict(parameters=p,compatibility=scopes[digest(p)]['compatibility'],
                policy=policy,runtime=scopes[digest(p)]['runtime']) for p in parameters]
        self.plans.append(task)
        return task


def choice_domain(name):
    from itertools import product
    from oeis_learn.generalization.progressive.tuning import SPACE,BASELINE,SHAPES
    if name=='baseline':return [dict(BASELINE)]
    shapes=SHAPES if name=='all' else [name if name in SHAPES else 'reference']
    if name not in {*SHAPES,'all','learning'}:raise ValueError('registered finite choice domain required')
    return [dict(zip(SPACE,values),**({'shape':shape} if name!='learning' else {}))
            for shape in shapes for values in product(*SPACE.values())]


def learner_scopes(compiler, support, summary):
    """Two exact trajectories and the full original finite learning/shape grid."""
    from oeis_learn.generalization.progressive.prepared_support import training_input
    from oeis_learn.generalization.progressive.models import resolve_profile
    from .progressive_training import resolve_update_policy
    from .progressive_learner import compatibility
    bases={};scopes={}
    for trajectory,horizon in [('screening',4000000),('confirmation',400000000)]:
        profile=resolve_profile(compiler.recipe,seed=0,dropout=0)
        base=training_input(support,summary,profile,resolve_update_policy(profile,schedule_horizon_tokens=horizon))
        ref=compiler.put('private/packages/'+trajectory+'.json',base)
        channel=compiler.channel(trajectory+'-training','training_request','learner',
            dict(membership='training',artifact=ref,compatibility=compatibility(base),profile='bigint-body/v1'))
        bases[trajectory]=dict(package=base,channel=channel,compatibility=compatibility(base))
        scopes[trajectory]={}
        for params in choice_domain('all'):
            profile=resolve_profile(compiler.recipe,seed=0,shape=params['shape'],dropout=params['dropout'])
            policy=resolve_update_policy(profile,schedule_horizon_tokens=horizon,
                **{k:v for k,v in params.items() if k not in {'shape','dropout'}},
                microbatch_size=16,warmup_tokens=1250000 if trajectory=='confirmation' and params['schedule']=='warmup_cosine' else None)
            package=seal(dict(base,profile=profile,update_policy=policy))
            value=dict(compatibility=compatibility(package),runtime=compiler.runtime(profile))
            scopes[trajectory][digest(params)]=value
            if params['shape']=='reference':
                scopes[trajectory][digest({k:v for k,v in params.items() if k!='shape'})]=value
    return bases,scopes


def compile_plan(root,output,data,design,materials,*,qualification,checkpoint_bytes):
    """Construct all literal role requests over original custodian publications.

    Returned data are a preparation plan, not a campaign. Calibration must add
    original iteration producers before freeze can authorize any learning.
    """
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.generalization.progressive.tuning import SHAPES
    from oeis_learn.generalization.progressive.targets import defaults
    from .progressive_training import resolve_update_policy
    from .progressive_learner import compatibility
    root=Path(root);output=Path(output)
    bound=inputs(root,data,design,materials)
    manifest=_json(checked_file(root,data['manifest']))
    retained=manifest['retained']['checkpoints']['D']
    checkpoint=reference(root,checked_file(root/retained['root'],retained['descriptor']['weights']))
    state=reference(root,checked_file(root/retained['root'],retained['descriptor']['state']))
    recipe=_json(checked_file(root,data['recipe']))
    c=Compiler(root,output,qualification=qualification,recipe=recipe,checkpoint=checkpoint,
               created_at='2026-11-11T00:00:00Z')
    base,scopes=learner_scopes(c,bound['support'],bound['support_summary'])
    basecomp=base['confirmation']['compatibility']
    maximum_update=64*bound['support_summary']['maximum_target_tokens']
    roster=bound['roster'];allrows=roster['tasks']
    atomic_write(output/'private/role-roster.json',roster)
    c.refs.extend([
        dict(name='recipe',role='custodian',kind='recipe',artifact=data['recipe']),
        dict(name='measured-original-state',role='custodian',kind='checkpoint',artifact=state),
        dict(name='retained-D-evaluation-weights',role='generator',kind='checkpoint',artifact=checkpoint),
        dict(name='current-qualification',role='custodian',kind='qualification',artifact=c.qref),
        dict(name='source',role='custodian',kind='source',artifact=reference(root,checked_file(c.qroot,c.q['source_closure']))),
        dict(name='exposure',role='custodian',kind='groups',artifact=materials['exposure'])])
    c.put('images.json',dict(version='discovery-images/v1',images=c.q['images']),kind='image')
    panelrefs={x['panel']:x['cohort'] for x in materials['panels']}
    visible={};truth={}
    for name,panel in bound['panels'].items():
        if name=='assessment':continue
        c.refs.append(dict(name='panel:'+name,role='custodian',kind='groups',artifact=panelrefs[name]))
        for r in panel['payload']['rows']:
            short=digest(r['id'])[7:]
            visible[r['id']]=c.channel('visible-'+short,'visible_prompt','generator',_json(checked_file(root,r['visible'])))
            kind='development_truth' if panel['payload']['role']=='development' else 'confirmation_truth'
            truth[r['id']]=c.channel('truth-'+short,kind,'scorer',_json(checked_file(root,r['truth'])))
    primary_policy=c.channel('primary-portfolio','oeis_generation','generator',dict(methods=['correct'],forced=None))
    co=c.channel('development-cohort','cohort','custodian',bound['panels']['development'])
    ex=c.channel('census-exposure','exposure','custodian',bound['exposure'])
    public_co=c.channel('public-cohort','public_evidence','verifier',bound['panels']['development'])
    public_ex=c.channel('public-exposure','public_evidence','verifier',bound['exposure'])
    witnesses=[]
    for r in bound['reservations']['development']:
        witnesses.append(c.channel('witness-'+digest(r['id'])[7:],'coverage_witness','custodian',
            dict(task=r['id'],source=r.get('source'),indices=r['indices'],values=r['values'],
                 operator_class=r.get('operations',[]) if r.get('source') is not None else [],
                 missing_operations=[],created_at=c.created_at)))
    evaluations={name:dict(value,qualification=None) for name,value in roster['study_evaluations'].items()}
    # Three cumulative token milestones share one original 4M+whole-update cap.
    study=seal(dict(version='progressive-study-policy/v1',token_cap=4000000+3*maximum_update,
        target_tokens=4000000,elapsed_cap_us=7200*1000000,
        screening=dict(startup_us=20*1000000,evaluation_us=[810*1000000,540*1000000,540*1000000],
                       training_us=2700*1000000,finalization_us=1800*1000000),
        panel=bound['screening'],evaluations=evaluations,
        confirmation=dict(panels=[panelrefs[f'confirmation-tuning-{i}'] for i in range(4)],seeds=design['seeds'])))
    c.put('study-policy.json',study,kind='policy',role='proposal')
    targets=defaults();targets.update(purpose='research',created_at=c.created_at)
    targets['payload']['selected']=True
    comparison_designs={}
    for item in roster['comparisons']:
        rows=[r for name in (item['panel'],item['retention']) for r in bound['panels'][name]['payload']['rows']]
        groups={s:{r['groups'][0] for r in rows if r['stratum']==s} for s in ('composition','unseen_structure')}
        links=[[f'composition/{g}',f'unseen_structure/{g}'] for g in sorted(groups['composition']&groups['unseen_structure'])]
        comparison_designs[item['id']]=dict(claim='capability',bootstrap_seed=20261112,
                                             comparisons=4,links=links,costs=None)
    for target in targets['payload']['targets']:
        if target['id']=='gain':target['evidence'].extend(digest(d) for d in comparison_designs.values())
    targets=seal(targets);targetref=c.put('targets.json',targets,kind='policy')
    def slot(task,operation,output,kind):return dict(task=task,operation=operation,output=output,kind=kind)
    slot_of={}
    for row in allrows:
        meta=row['trial']
        if row['stage']!='tuning':slot_of[row['id']]=row['stage']
        elif not meta or meta['phase']!='confirmation':slot_of[row['id']]='tuning-screening'
        else:
            position=row.get('finalist_position',meta['ordinal'])
            if row['id'].startswith('confirmation-baseline-'):position=int(row['id'].split('-')[2])
            slot_of[row['id']]=f'tuning-confirmation-{position}'
    comparison_by_id={r['id']:r for r in roster['comparisons']}
    profile_channels={}
    def profile_channel(seed,trajectory):
        key=(seed,trajectory)
        if key not in profile_channels:
            profile=resolve_profile(recipe,seed=seed,dropout=0)
            profile_ref=c.put(f'private/profiles/{seed}-{trajectory}.json',profile)
            comp=base[trajectory]['compatibility']
            profile_channels[key]=c.channel(f'profile-{seed}-{trajectory}','recipe','generator',
                dict(artifact=profile_ref,compatibility=comp,profile='bigint-body/v1'))
        return profile_channels[key]
    for row in allrows:
        meta=row['trial'];ordinal=meta['ordinal'] if meta else None
        seed=row['seed'] if row['seed'] is not None else int(digest([study['identity'],ordinal])[7:15],16)
        trajectory='screening' if meta and meta['phase'] not in {'confirmation','selection'} else 'confirmation'
        comp=base[trajectory]['compatibility'];params=None
        if meta and meta['phase'] not in {'proposal','selection'}:
            domain=('baseline' if ordinal==0 else 'all') if meta['phase']=='confirmation' else ('learning' if ordinal<12 else SHAPES[ordinal-12])
            # Fixed reference random retention control has no selected weights.
            if row.get('control')!='random':params=choice_domain(domain)
        operation=row['operation']
        if operation=='train':
            if meta:training=base[trajectory]['channel']
            else:
                package=seal(dict(base['confirmation']['package'],profile=resolve_profile(recipe,seed=seed,dropout=0)))
                package=seal(dict(package,update_policy=resolve_update_policy(package['profile'],schedule_horizon_tokens=400000000)))
                ref=c.put(f'private/packages/diversity-{seed}.json',package) if not (output/f'private/packages/diversity-{seed}.json').exists() else reference(root,output/f'private/packages/diversity-{seed}.json')
                training=c.channel(f'diversity-training-{seed}','training_request','learner',dict(
                    membership='training',artifact=ref,compatibility=compatibility(package),profile='bigint-body/v1'))
            channels={'training':[training]}
        elif operation=='generate':
            model=[profile_channel(seed,trajectory)]
            if row['checkpoint']:
                model.append(slot(row['checkpoint'],'train','weights','checkpoint'))
            else:
                model.append(c.channel('checkpoint-'+str(seed)+'-'+str(row['control'])+'-'+trajectory,
                    'checkpoint','generator',dict(artifact=checkpoint,compatibility=comp,
                        profile='bigint-body/v1',control=row['control'])))
            if not row['all_methods']:model.append(primary_policy)
            channels=dict(visible=[visible[i] for i in row['members']],model=model)
        elif operation=='score':
            channels=dict(candidates=[slot(row['generation'],'generate','candidates','candidate')],
                          score_truth=[truth[i] for i in row['members']])
        elif operation=='census':channels=dict(sources=[co,ex])
        elif operation=='audit-language':channels=dict(sources=witnesses)
        elif operation=='propose-trial':
            channels=dict(search=[c.channel('proposal-'+str(ordinal),'search_space','proposal',
                dict(study=study['identity'],ordinal=ordinal,history=[],selected_learning=None))])
        elif operation=='publish-selection':
            channels=dict(public=[c.channel('selection','public_evidence','verifier',
                dict(version='progressive-selection-request/v1',study=study['identity']))])
        elif operation=='publish-readiness':
            public=[public_co,public_ex,slot('capability-census','census','census','public_evidence'),
                    slot('capability-language','audit-language','coverage','public_evidence')]
            for pair in row['evaluations']:
                public.extend([slot(pair['generation'],'generate','evaluation','public_evidence'),
                               slot(pair['scoring'],'score','scores','public_evidence')])
            channels=dict(public=public)
        elif operation=='publish-comparison':
            item=comparison_by_id[row['id']];registered=[];producers=[];public=[]
            selected_panels=[bound['panels'][name] for name in (item['panel'],item['retention'])]
            comparison_roster=[dict(task=r['id'],group=sorted(r['groups'])[0],stratum=r['stratum'])
                               for panel in selected_panels for r in panel['payload']['rows']]
            target_slot=row['id']+'-disposition'
            for p in item['producers']:
                gen=p['primary']['generation'];scored=p['primary']['scoring']
                registered.append(dict(arm=p['arm'],seed=p['seed'],checkpoint=slot(p['checkpoint'],'train','weights','checkpoint'),
                    model=None,candidates=slot(gen,'generate','candidates','candidate'),scores=slot(scored,'score','scores','public_evidence')))
                producers.append(dict(arm=p['arm'],seed=p['seed'],generation_task=gen,scoring_task=scored))
                public.extend([slot(gen,'generate','evaluation','public_evidence'),slot(scored,'score','scores','public_evidence')])
            gate=seal(dict(version='progressive-gate-registration/v1',source_slot=slot_of[row['id']],target_slot=target_slot,
                panels=[p['identity'] for p in selected_panels],seeds=design['seeds'],
                design=digest(comparison_designs[row['id']]),producers=registered))
            c.put('comparisons/'+row['id']+'-gate.json',gate,kind='policy',role='verifier')
            c.put('comparisons/'+row['id']+'-design.json',comparison_designs[row['id']],kind='policy',role='verifier')
            template=dict(version='progressive-decision-template/v1',source_slot=slot_of[row['id']],target_slot=target_slot,
                roster=comparison_roster,seeds=design['seeds'],design=comparison_designs[row['id']],policy=targets,
                remaining={},kwargs=dict(target_id='gain',action='RETAIN_BASELINE'),producers=producers)
            channels=dict(public=[c.channel(row['id']+'-template','public_evidence','verifier',template),*public])
        else:raise ValueError('unregistered scientific role '+operation)
        plan=c.task(row,channels,seed=seed,compatibility=comp,study=study,parameters=params,
                    scopes=scopes[trajectory],maximum_update=maximum_update,checkpoint_bytes=checkpoint_bytes)
        plan['slot']=slot_of[row['id']]
    return c,bound,study,targetref,state,retained['root']


def finish_plan(c,bound,study,targetref,state,checkpoint_root,*,data,design,materials):
    """Publish the non-executable canonical prepare input and all setup ceilings."""
    from oeis_learn.generalization.progressive.budgets import KEYS
    from oeis_learn.generalization.progressive.graph import validate_graph
    from .progressive_calibration import selected_inputs
    from .progressive_calibration_storage import required_bytes,POLICY
    def zeros():
        return dict.fromkeys(KEYS,0)
    nodes=[]
    slots=['capability','diversity','tuning-screening',*[f'tuning-confirmation-{i}' for i in range(4)]]
    for slot in slots:
        stage=slot if slot in {'capability','diversity'} else 'tuning'
        dependencies=[] if stage=='capability' else ['capability'] if stage=='diversity' else ['diversity'] if slot=='tuning-screening' else ['tuning-screening']+(['tuning-confirmation-0'] if slot!='tuning-confirmation-0' else [])
        children=[p for p in c.plans if p['slot']==slot]
        caps={k:sum(p['caps'][k] for p in children) for k in KEYS}
        if slot=='tuning-screening':caps['elapsed_us']+=16*1800*1000000
        nodes.append(dict(id=slot,stage=stage,kind='routine',dependencies=dependencies,maximum_visits=1,caps=caps))
    for item in bound['roster']['comparisons']:
        source=next(p['slot'] for p in c.plans if p['id']==item['id'])
        nodes.append(dict(id=item['id']+'-disposition',stage=item['stage'],kind='routine',
                          dependencies=[source],maximum_visits=1,caps=zeros()))
    caps={k:sum(n['caps'][k] for n in nodes) for k in KEYS}
    # Shared projection, metadata, source support and final public packaging.
    caps['disk_bytes']+=16<<30
    caps['archive_bytes']=max(caps['archive_bytes'],512<<20)
    graph=dict(selected_stages=['capability','diversity','tuning'],nodes=nodes,edges=[],diagnostic_pool=zeros())
    validate_graph(graph,caps=caps)
    if sum(p['caps']['elapsed_us'] for p in c.plans if p['kind']!='training')>136*3600*1000000:
        raise ValueError('scientific auxiliary roles exceed the original campaign ceiling')
    profile=__import__('oeis_learn.generalization.progressive.models',fromlist=['resolve_profile']).resolve_profile(c.recipe,seed=0,dropout=0)
    runtime=c.runtime(profile)
    c.put('acceptance-plan.json',seal(dict(version='progressive-acceptance-plan/v1',qualification=c.qref,seconds=3600)),kind='policy')
    # The named entry is required by the canonical acceptance selector.
    c.refs[-1]['name']='acceptance_plan'
    payload=dict(runtime=runtime,graph=graph,task_plans=c.plans,references=c.refs,caps=caps,
        checkpoint=state,checkpoint_root=checkpoint_root,milestone_pins=0,
        shard_bytes=materials['support']['bytes'],package_bytes=512<<20,required_inodes=1000000,
        target_policy=targetref,deficits=[])
    preview=dict(purpose='research',track='synthetic_strict',payload=payload)
    cases,_=selected_inputs(c.root,preview)
    tokens=sum(2*x['package']['update_policy']['effective_batch']*
        max(len(r['tokens']) for shard in x['package']['shards'] for r in shard) for x in cases)
    inputs_bytes=sum(2*(len(json.dumps(x['package'],sort_keys=True,indent=2,allow_nan=False).encode())+1) for x in cases)
    checkpoint_ceiling=max((p['caps']['disk_bytes']-(128<<20))//2 for p in c.plans if p['kind']=='training')
    setup=zeros();setup.update(tokens=tokens,elapsed_us=12*3600*1000000,
        disk_bytes=required_bytes(len(cases),checkpoint_ceiling,inputs_bytes,retention=POLICY),archive_bytes=512<<20)
    calibration=seal(dict(version='progressive-calibration-plan/v1',seconds=12*3600,
        caps=setup,checkpoint_bytes=checkpoint_ceiling,maximum_inodes=64*len(cases)+256,checkpoint_retention=POLICY))
    calibration_ref=c.put('calibration-plan.json',calibration)
    preparation_ref=c.put('preparation-plan.json',dict(version='progressive-preparation-plan/v1',payload=payload))
    proposed={k:caps[k]+setup[k] for k in KEYS}
    blueprint=record('blueprint',dict(executable=False,selected_stages=graph['selected_stages'],
        references=[dict(name='preparation_plan',role='custodian',kind='policy',artifact=preparation_ref),
                    dict(name='iteration_calibration_plan',role='custodian',kind='policy',artifact=calibration_ref)],
        unresolved=[],proposed_caps=proposed,reference_campaign=str(c.output.relative_to(c.root))),
        created_at=c.created_at,purpose='research',track='synthetic_strict')
    c.put('blueprint.json',blueprint)
    summary=seal(dict(version='progressive-research-construction/v1',status='COMPLETE',executable=False,
        blueprint=reference(c.root,c.output/'blueprint.json'),design=design['identity'],materials=materials['identity'],
        registration=data['identity'],tasks=len(c.plans),exact_calibration_cases=len(cases),
        panel_truth_isolated=True,assessment_exposed=False,
        role_hours=sum(p['caps']['elapsed_us'] for p in c.plans)/3600000000,
        shared_screening_finalization_hours=8,calibration_hours=12,
        storage_ceiling_bytes=proposed['disk_bytes'],checkpoint_ceiling_bytes=checkpoint_ceiling,
        frozen_campaign_required=True,measured_feasibility_required=True))
    c.put('construction-summary.json',summary)
    return summary


def build(root,output,data,design,materials,*,qualification,checkpoint_bytes=8<<30):
    if type(checkpoint_bytes) is not int or not 0<checkpoint_bytes<=64<<30:
        raise ValueError('finite prospective checkpoint storage ceiling required')
    args=compile_plan(root,output,data,design,materials,qualification=qualification,checkpoint_bytes=checkpoint_bytes)
    return finish_plan(*args,data=data,design=design,materials=materials)
