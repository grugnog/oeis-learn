"""Finite outcome-blind scientific task roster, before artifact/engine binding.

This is a construction rule, not execution authority. The canonical custodian,
calibration and freeze paths must bind each role and its original inputs. No
assessment role is constructed here, including for an unused finalist position.
"""
from .contracts import digest, seal
from .tuning import SPACE, BASELINE, SHAPES


def construct(design, panels, screening):
    """Compile the selected capability/diversity/tuning round into exact roles.

    Names, partitions, children, continuations and all possible confirmation
    positions are fixed before model outcomes. Later conditional branches retain
    separate reservation requirements; a negative result never creates a task.
    """
    if (seal(design)!=design or design.get('version') not in {'progressive-research-round-design/v1','progressive-research-round-design/v2'}
            or design['selected_stages']!=['capability','diversity','tuning']):
        raise ValueError('original prospective capability/diversity/tuning round required')
    from .tuning import validate_screening_panel
    grouping=design['tuning'].get('screening_group_policy','paired-16/v1')
    expected='balanced-at-most-two/v2' if design['version'].endswith('/v2') else 'paired-16/v1'
    if grouping!=expected:raise ValueError('screening grouping differs from prospective design version')
    validate_screening_panel(screening,grouping=grouping)
    names={'development','confirmation-diversity',*[f'confirmation-tuning-{i}' for i in range(4)],
           'retention-development','retention-diversity','retention-tuning','assessment'}
    if set(panels)!=names:
        raise ValueError('complete independently reserved panel inventory required')
    for name,panel in panels.items():
        if (not isinstance(panel,list) or not panel or len({r['id'] for r in panel})!=len(panel)
                or any(set(r)!={'id','group','stratum'} for r in panel)):
            raise ValueError('closed original panel roster required: '+name)
        strata=['retention'] if name.startswith('retention-') else ['parameter','composition','unseen_structure','oeis']
        if set(r['stratum'] for r in panel)!=set(strata):
            raise ValueError('original panel strata differ: '+name)
        for s in strata:
            selected=[r for r in panel if r['stratum']==s]
            if len(selected)<128 or len({r['group'] for r in selected})<16:
                raise ValueError('original full-panel quota deficit: '+name)
    original={r['id']:r for r in panels['development']}
    if any(original.get(r['id'])!={k:r[k] for k in ('id','group','stratum')} for r in screening):
        raise ValueError('screening members differ from original development reservations')
    if len(design['seeds'])!=2 or len(set(design['seeds']))!=2:
        raise ValueError('two original paired seeds required')
    t=design['tuning']
    if (t['learning_trials']!=12 or t['sizing_trials']!=4 or t['space']!=SPACE
            or t['shapes']!=list(SHAPES) or t['unchanged_baseline']!=BASELINE
            or t['screening_looks']!=[1000000,2000000,4000000]
            or t['screen_seconds']!=7200 or t['protected_finalization_seconds']!=1800
            or t['confirmation_tokens']!=25000000 or t['confirmation_seconds']!=43200):
        raise ValueError('complete registered study domain and horizons required')
    from .research_budget import ROLE_SECONDS
    tasks=[];comparisons=[]
    # Duration allocations are ceilings, not measurements or promised throughput.
    # Exact-input/evaluation feasibility must reject insufficient allocations.
    def add(name,stage,operation,*,seed=None,arm='refreshed',trial=None,**kw):
        row=dict(id=name,stage=stage,operation=operation,seed=seed,arm=arm,trial=trial,**kw)
        tasks.append(row)
        return name
    def evaluation(name,stage,panel,checkpoint,*,seed,arm='refreshed',trial=None,members=None,
                   control=None,all_methods=False,seconds=None,retention=None):
        selected=panels[panel] if members is None else members
        if retention:selected=[*selected,*panels[retention]]
        gen=add(name+'-generate',stage,'generate',seed=seed,arm=arm,trial=trial,
                panel=panel,members=[r['id'] for r in selected],checkpoint=checkpoint,
                retention=retention,control=control,all_methods=all_methods,
                seconds=seconds or ROLE_SECONDS['generate'])
        score_trial=dict(trial,phase='scoring') if trial and trial['phase']=='generation' else trial
        score=add(name+'-score',stage,'score',seed=seed,arm=arm,trial=score_trial,
                  panel=panel,retention=retention,members=[r['id'] for r in selected],generation=gen,
                  seconds=20 if trial and trial['phase'] not in {'confirmation','selection'} else ROLE_SECONDS['score'])
        return dict(generation=gen,scoring=score)
    development=sorted(panels['development'],key=lambda r:r['id'])
    add('capability-census','capability','census',seed=design['seeds'][0],panel='development',seconds=2100)
    add('capability-language','capability','audit-language',seed=design['seeds'][0],panel='development',
        members=[r['id'] for r in development],seconds=ROLE_SECONDS['audit_language'])
    capability=[]
    for control in ('retained_D','random'):
        for i in range(0,len(development),64):
            capability.append(evaluation(f'capability-{control}-{i//64:02d}','capability','development',
                None,seed=design['seeds'][0],members=development[i:i+64],control=control,all_methods=True))
    add('capability-readiness','capability','publish-readiness',seed=design['seeds'][0],
        evaluations=capability,seconds=2100)
    diversity=[]
    for seed in design['seeds']:
        for arm in ('static','refreshed'):
            name=f'diversity-{arm}-{seed}'
            train=add(name,'diversity','train',seed=seed,arm=arm,token_target=25000000,
                      horizon=400000000,parent=None,domain='baseline',seconds=43200)
            primary=evaluation(name+'-primary','diversity','confirmation-diversity',train,
                               seed=seed,arm=arm,retention='retention-diversity')
            diversity.append(dict(seed=seed,arm='baseline' if arm=='static' else 'candidate',
                                  checkpoint=train,primary=primary,retention=primary))
    comparisons.append(dict(id='diversity-comparison',stage='diversity',panel='confirmation-diversity',
                            retention='retention-diversity',producers=diversity))
    add('diversity-comparison','diversity','publish-comparison',seed=design['seeds'][0],seconds=2100)
    evaluations={}
    for ordinal in range(16):
        prefix=f'trial-{ordinal:02d}'
        # Seed is derived from the *bound study policy* after the roster is
        # published. No outcome or path spelling participates in that choice.
        meta=dict(ordinal=ordinal,phase='proposal',coordinate=0)
        add(prefix+'-propose','tuning','propose-trial',trial=meta,seconds=20)
        baseline=evaluation(prefix+'-retention-baseline','tuning','retention-development',None,
            seed=None,trial=dict(meta,phase='retention'),control='random',seconds=250)
        parent=None
        for coordinate in t['screening_looks']:
            name=prefix+'-'+str(coordinate)
            train=add(name+'-train','tuning','train',trial=dict(meta,phase='training'),
                token_target=coordinate,horizon=4000000,parent=parent,
                domain='learning' if ordinal<12 else SHAPES[ordinal-12],seconds=900)
            retention=evaluation(name+'-retention','tuning','retention-development',train,
                seed=None,trial=dict(meta,phase='retention'),seconds=250)
            primary=evaluation(name+'-development','tuning','development',train,
                seed=None,trial=dict(meta,phase='generation',coordinate=coordinate),members=screening,seconds=250)
            evaluations[primary['scoring']]=dict(generation_task=primary['generation'],checkpoint_task=train,
                roster=[dict(task=r['id'],group=r['group'],stratum=r['stratum']) for r in screening],
                retention_baseline_generation=baseline['generation'],retention_baseline_scoring=baseline['scoring'],
                retention_candidate_generation=retention['generation'],retention_candidate_scoring=retention['scoring'])
            parent=train
    add('tuning-selection','tuning','publish-selection',seed=design['seeds'][0],
        trial=dict(ordinal=0,phase='selection',coordinate=0),seconds=2100)
    confirmations={}
    for position in range(4):
        for seed in design['seeds']:
            name=f'confirmation-{position}-{seed}'
            meta=dict(ordinal=position,phase='confirmation',coordinate=0)
            train=add(name,'tuning','train',seed=seed,trial=meta,token_target=25000000,
                      horizon=400000000,parent=None,domain='baseline' if position==0 else 'all',seconds=43200)
            primary=evaluation(name+'-primary','tuning',f'confirmation-tuning-{position}',train,
                seed=seed,trial=meta,retention='retention-tuning')
            confirmations[position,seed]=dict(checkpoint=train,primary=primary,retention=primary)
    for position in range(1,4):
        paired=[]
        for seed in design['seeds']:
            base=confirmations[0,seed]
            # Baseline must be evaluated on the candidate's *same* independent
            # panel. Its model choice remains baseline, not that candidate's.
            primary=evaluation(f'confirmation-baseline-{position}-{seed}','tuning',
                f'confirmation-tuning-{position}',base['checkpoint'],seed=seed,
                trial=dict(ordinal=0,phase='confirmation',coordinate=0),retention='retention-tuning')
            paired.extend([dict(seed=seed,arm='baseline',checkpoint=base['checkpoint'],primary=primary,retention=base['retention']),
                           dict(seed=seed,arm='candidate',**confirmations[position,seed])])
        name=f'tuning-comparison-{position}'
        comparisons.append(dict(id=name,stage='tuning',panel=f'confirmation-tuning-{position}',
                                retention='retention-tuning',producers=paired,finalist_position=position))
        add(name,'tuning','publish-comparison',seed=design['seeds'][0],seconds=2100,finalist_position=position,
            trial=dict(ordinal=position,phase='confirmation',coordinate=0))
    if len(tasks)>512 or len({r['id'] for r in tasks})!=len(tasks):
        raise ValueError('bounded unique scientific role roster required')
    return seal(dict(version='progressive-research-role-roster/v1',design=design['identity'],
        panels={k:digest(v) for k,v in panels.items()},screening=digest(screening),tasks=tasks,
        comparisons=comparisons,study_evaluations=evaluations,
        partition_rule='lexicographic original member identity; groups/panels remain original; 64 members per capability task',
        executable=False,assessment_exposed=False,
        feasibility='all per-task allocations require original measured feasibility before freeze'))
