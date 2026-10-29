"""Closed bounded post-run diagnostic authority; no learning or resume."""
from .identity import seal,validate,digest

BUDGETS=dict(global_seconds=28800,build_per_engine_seconds=14400,phase_seconds={'1':3600,'2':1800,'4':12000,'5':300},
    archive_seconds=900,cleanup_seconds=300,stream_bytes=8<<20,transient_bytes=16<<30,trace_bytes=128<<20,
    zip_bytes=512<<20,mandatory_reserve_bytes=16<<20,free_disk_reserve_bytes=5<<30,bundle_members=4097)


def plan(*,operation,device='hip',snapshots=None,training_rows=32768,spectra=False,include_selected=False,evaluation_policy=None):
    snapshots=list(range(6)) if snapshots is None else list(snapshots)
    if type(include_selected) is not bool:raise ValueError('boolean selected checkpoint option required')
    if include_selected:snapshots=sorted(set(snapshots)|{6})
    if operation not in ('collect','models','evaluate') or device not in ('hip','cpu'):raise ValueError('registered diagnostic operation/device required')
    if (not snapshots or snapshots!=sorted(set(snapshots)) or any(type(i) is not int or not 0<=i<=(6 if include_selected else 5) for i in snapshots)
        or type(training_rows) is not int or not 32<=training_rows<=32768 or type(spectra) is not bool):
        raise ValueError('finite frozen snapshot/population roster required')
    b={**BUDGETS,'phase_seconds':dict(BUDGETS['phase_seconds'])}
    if operation=='collect':b.update(global_seconds=600,archive_seconds=60,cleanup_seconds=60)
    result=seal(dict(version='pipeline-pilot-diagnostics/v1',scope='postrun-diagnostics',operation=operation,device=device,
        snapshots=snapshots,training_rows=training_rows,spectra=spectra,batch=16,matched_rows=288,activation_rows=64,
        training=False,resume=False,final_access=False,resource_policy='workstation90',budgets=b))
    if include_selected:
        result.update(version='pipeline-pilot-diagnostics/v2',include_selected=True)
        result['budgets']['phase_seconds']['4']=14000
        result=seal(result)
    if operation=='evaluate':
        if device!='hip':raise ValueError('saved-model evaluation requires qualified HIP')
        from .pilot_clocks import MATCHED,DRAIN
        evaluation_policy=evaluation_policy or DRAIN
        if evaluation_policy not in (MATCHED,DRAIN):raise ValueError('matched evaluation policy required')
        result.update(version='pipeline-pilot-diagnostics/v3',include_selected=include_selected,evaluation_policy=evaluation_policy)
        result['budgets'].update(global_seconds=86400,phase_seconds={'1':3600,'2':1800,'4':78000,'5':300})
        result=seal(result)
    elif evaluation_policy is not None:raise ValueError('evaluation policy requires evaluate operation')
    return result


def parse(record):
    version=record.get('version');selected=version in ('pipeline-pilot-diagnostics/v2','pipeline-pilot-diagnostics/v3')
    fields={'version','scope','operation','device','snapshots','training_rows','spectra','batch','matched_rows',
        'activation_rows','training','resume','final_access','resource_policy','budgets'}
    if selected:fields.add('include_selected')
    if version=='pipeline-pilot-diagnostics/v3':fields.add('evaluation_policy')
    validate(record,fields,version)
    expected=plan(**{k:record[k] for k in ('operation','device','snapshots','training_rows','spectra')},
        include_selected=record.get('include_selected',False),evaluation_policy=record.get('evaluation_policy'))
    if record!=expected:raise ValueError('registered closed diagnostic plan required')
    return record


def populations(examples,roster,*,training_rows):
    from oeis_learn.fresh.conditioning_controls import derangement
    splits=('training','parameter_development','structural_development');ids={k:sorted(r['id'] for r in examples[k]) for k in splits}
    if len(set(p for values in ids.values() for p in values))!=sum(map(len,ids.values())):raise ValueError('unique diagnostic population required')
    panel=sorted(roster['training_panel'])
    if len(panel)!=32 or len(set(panel))!=32 or not set(panel)<=set(ids['training']):raise ValueError('fixed training panel missing')
    sample=sorted(ids['training'],key=lambda p:(digest(['diagnostic-training-sample/v1',p]),p))[:training_rows]
    correct=sorted(sample+ids[splits[1]]+ids[splits[2]])
    matched=sorted(panel+ids[splits[1]]+ids[splits[2]]);shuffled={}
    for group in (panel,ids[splits[1]],ids[splits[2]]):shuffled.update(derangement(group,20261016))
    activation=sorted(panel+[p for key in splits[1:] for p in sorted(ids[key],key=lambda p:digest(['diagnostic-activation/v1',p]))[:16]])
    return seal(dict(version='pipeline-diagnostic-populations/v1',correct=correct,matched=matched,shuffled=shuffled,
        activation=activation,training_available=len(ids['training']),training_selected=len(sample),
        training_sampling='full' if len(sample)==len(ids['training']) else 'fixed ID hash subset; not a full-population estimate'))
