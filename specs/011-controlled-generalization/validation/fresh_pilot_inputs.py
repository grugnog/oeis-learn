"""Frozen fresh P1 research inputs; no model, hardware or qualification decisions."""
from pathlib import Path
from oeis_learn.data.bigint_programs import BigintSampler
from oeis_learn.generalization.refresh import RefreshPlan
from oeis_learn.pipeline_validation.identity import seal,write,digest

SEED=20261016
NUMERIC_LAW='independent-uniform-integers[-1024,1024]/v1'
BANK_SEEDS=list(range(20261101,20261109))


def diverse(descriptor):
    """Source-only screen, independent of holdouts or model outcomes.

    Reject fixed/near-fixed20-term difference windows across16 numeric variants.
    This removes additive-only families and parameter-independent long branches;
    it is a distribution choice, not proof that all later quotas are feasible.
    """
    from collections import Counter
    from oeis_learn.data.bigint_programs import materialize
    rows,_=RefreshPlan([descriptor],seed=SEED+301,capacity=4096).take(0,16)
    windows=Counter()
    for row in rows:
        try:values=list(map(int,materialize(row)['values']))
        except ValueError:return False
        differences=[b-a for a,b in zip(values,values[1:])]
        windows.update({tuple(differences[i:i+19]) for i in range(81)})
    return max(windows.values(),default=16)<=2


def research_inputs():
    """Bounded generic descriptor selection, before any learning outcomes.

    Thirty-two equally weighted training families: A4/B6/C2/D20. Each shard
    contains1024 admitted instances/family. Twelve shards provide a one-pass
    refreshed arm; static repeats the first. Protected roles are prepared first
    so every training draw is checked against the full frozen holdout roster.
    """
    from collections import Counter
    from oeis_learn.generalization.targets import ground
    from oeis_learn.decoder.bigint_codec import encode_body
    sampler=BigintSampler(SEED);catalog={}
    for ordinal in range(4096):
        d=seal(dict(sampler.descriptor(ordinal),numeric_law=NUMERIC_LAW))
        catalog.setdefault(d['family'],d)
    wanted=Counter(A=4,B=6,C=2,D=20);training=[];used=set()
    for d in catalog.values():
        if len(d['parameter_names'])>=3 and wanted[d['stage']] and diverse(d):
            training.append(d);used.add(d['family']);wanted[d['stage']]-=1
    if any(wanted.values()):raise ValueError('bounded training descriptor shortfall')
    cohorts=[]
    for offset,role in enumerate(('selection','assessment')):
        # Parameter holdout shares only training families, never admitted rows.
        rows,_=RefreshPlan(training,seed=SEED+101+offset,capacity=4096).take(0,2048)
        cohorts.append(dict(name=role+'-parameter',role=role,holdout='parameter',count=128,
            quotas={'A':8,'B':8,'C':8,'D':8},structural=False,length_boundary=16,rows=_unique(rows)))
        structural=[];wanted=Counter(A=8,B=8,C=5,D=8)
        # Reserve one generic loop family with large-magnitude support for each
        # structural role. This is source-only feasibility, never model selection.
        from oeis_learn.generalization.coverage import observe
        for d in catalog.values():
            if d['family'] in used or d['stage']!='C':continue
            source=ground(d['parameter_source'],{p:'2' for p in d['parameter_names']})
            if observe(source,tuple(map(str,range(100))),stage='C').above_64_bits:
                structural.append(d);used.add(d['family']);wanted['C']-=1;break
        else:raise ValueError('bounded large-magnitude loop support shortfall')
        for d in catalog.values():
            if d['family'] not in used and d['parameter_names'] and wanted[d['stage']]:
                structural.append(d);used.add(d['family']);wanted[d['stage']]-=1
        if any(wanted.values()):raise ValueError('bounded structural descriptor shortfall')
        rows,_=RefreshPlan(structural,seed=SEED+201+offset,capacity=4096).take(0,2048)
        cohorts.append(dict(name=role+'-structural',role=role,holdout='structural',count=128,
            quotas={'A':16,'B':16,'C':16,'D':16,'short':32,'long':32,'within64':32,
                    'above64':16,'loop_active':32,'branch_active':32},structural=True,length_boundary=16,rows=_unique(rows)))
    quotas={'family:'+d['family']:1024 for d in training}
    cells=[dict(name='static' if i==0 else f'refresh-{i:02}',role='training',holdout='parameter',
        count=32768,quotas=quotas.copy(),structural=False,length_boundary=16,
        refresh=dict(seed=SEED,capacity=1_000_000,cursor=65536*i,draws=65536)) for i in range(12)]
    return seal(dict(version='fresh-pilot-design/v2',seed=SEED,descriptors=training,
        training=cells,cohorts=cohorts,numeric_law=NUMERIC_LAW,descriptor_screen='16-variant-difference-window/v1',bank_seeds=BANK_SEEDS,
        target_tokens=25_000_000,learner_seconds=21600,preparation_seconds=21600,feasibility_seconds=21600,reference_workers=4,
        generation_seconds=21600,scoring_seconds=21600,campaign_seconds=60*3600,
        primary='latest available per arm; matched-cap comparison only if both learners complete; selection supervision only',
        controls=['raw','rescue','equal_budget','shuffled','absent','template','bank'],
        progress_rule='no automatic advancement; require matched exposure and95% primary completion before interpreting the screen',
        assessment='SEALED; no target or score access',
        checkpoint_policy='every-update-full-state; verified byte-identical blob reuse',
        historical_comparison='DEFERRED',qualification='NOT_RUN'))


def _unique(rows):
    seen=set();result=[]
    for row in rows:
        if row['source'] not in seen:seen.add(row['source']);result.append(row)
    return result


def blueprint(workspace,design,*,recipe,source,images,feature,output):
    """Write only the exact design inputs used by canonical study assembly."""
    from oeis_learn.pipeline_validation.identity import confined
    root=Path(workspace);target=confined(root,output,exists=False)
    if target.exists():raise ValueError('fresh research design root required')
    if seal(design)!=design or design['version']!='fresh-pilot-design/v2':raise ValueError('intact frozen design required')
    target.mkdir(parents=True)
    def put(name,value):
        p=target/name;write(p,value);return str(p.relative_to(root))
    ds=put('descriptors.json',design['descriptors']);ex=put('exclusions.json',[])
    put('design.json',design)
    cells=[]
    for cell in design['cohorts']:
        cells.append(dict({k:v for k,v in cell.items() if k!='rows'},candidates=put(cell['name']+'.json',cell['rows'])))
    cells.extend(design['training'])
    cfg=dict(version='generalization-input-blueprint/v1',stage='P1-screen',purpose='research',
        hypothesis='Exploratory fresh static versus within-support refresh; frozen pilot protocol '+design['identity']+'; historical E0/P0 deferred',
        recipe=recipe,source_closure=source,images=images,initializer=None,feature=feature,
        descriptors=ds,exclusions=ex,cells=cells,
        training={'static':dict(cells=['static'],data_mode='static',batch_size=16),
                  'refreshed':dict(cells=[c['name'] for c in design['training']],data_mode='refresh',batch_size=16)},
        evaluation=None,replay=None,seeds=[SEED],
        budgets=dict(target_tokens=25_000_000,seconds=21600,draws=1_000_000,
            memory_bytes=64<<30,storage_bytes=8<<30,log_bytes=4<<30,attempts=8),
        milestones=[4_000_000,8_000_000,16_000_000,25_000_000],
        storage=dict(study_bytes=64<<30,checkpoint_bytes=32<<30,shard_bytes=8<<30,raw_bytes=4<<30,sharing_bytes=512<<20),
        sharing_policy='explicit')
    put('blueprint.json',cfg)
    return cfg
