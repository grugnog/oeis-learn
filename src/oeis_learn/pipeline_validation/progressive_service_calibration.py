"""Training-only native service measurements; never research or HIP authority."""
import json
import os
from pathlib import Path
import sys
import time
import traceback

from .identity import confined, seal, digest
from .progressive_research_inputs import checked_file, _json
from .progressive_workstation_state import read
from .preparation_recovery import retain, revise
from oeis_learn.generalization.progressive.contracts import reference

VERSION = 'progressive-service-calibration/v1'
SECONDS = 1200
HOST_SECONDS = SECONDS + 120
CASE_SECONDS = 120


def gather(root, donor, output):
    import fcntl
    root=Path(root);folder=confined(root,donor)
    if Path(output).resolve().is_relative_to(folder):
        raise ValueError('service diagnostic needs a separate session')
    lock=confined(root,folder/'session.lock',exists=False)
    with lock.open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _gather(root,donor,output)


def _gather(root, donor, output):
    """Read the original publications, not the raw corpus or a new census."""
    from .progressive_handoff import original_authority
    from .progressive_timing_coverage import coverage
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.progressive.training_rows import validate_row
    root, output = Path(root), Path(output)
    folder = confined(root, donor)
    if not folder.is_relative_to(root/'run-state'):
        raise ValueError('original workstation preparation required')
    binding = read(folder/'session.json')['binding']
    journal = read(folder/'journal.json')
    base = folder/'prepare-research'
    if any(k.startswith(('run-research/', 'assess/')) for k in journal['steps']):
        raise ValueError('service planning must precede scientific execution')
    originals = {}
    for name in ('native-census', 'materials', 'role-roster', 'qualify', 'evaluation-plan'):
        step = journal['steps'].get('prepare-research/'+name, {})
        if step.get('status') != 'COMPLETE':
            raise ValueError('complete original preparation step required: '+name)
        for ref in step['outputs']:
            checked_file(root, ref)
            originals[ref['path']] = ref

    def owned(path, maximum=32<<20):
        ref = reference(root, path)
        if originals.get(ref['path']) != ref:
            raise ValueError('original publication not journal-bound: '+str(path))
        return _json(path, maximum=maximum)

    materials = owned(base/'materials/materials.json')
    if materials.get('status') != 'COMPLETE' or materials.get('diagnostic') is not False:
        raise ValueError('complete original research materials required')
    support_path = checked_file(root, materials['support'])
    support = owned(support_path, maximum=256<<20)
    if (support.get('diagnostic') is not False or seal(support)!=support
            or support.get('version')!='progressive-prepared-support/v1'):
        raise ValueError('native gradient support required')
    allrows = {r['draw_identity']:r for shard in support['shards'] for r in shard}
    for row in allrows.values():
        validate_row(row, native_verified=True)
    probes={p['id'] for p in support['layout']['probes'].values()}
    rows={k:r for k,r in allrows.items() if k not in probes}
    if len(rows) < 4:
        raise ValueError('four distinct original training members required')
    # Deliberately no probes, panels, model scores or replacement after outcomes.
    order = sorted(rows.values(), key=lambda r:r['identity'])
    chosen = {}
    for key in (lambda r:len(r['tokens']), lambda r:max(len(v) for v in r['values'][:20]),
                lambda r:sum(len(v) for v in r['values']), lambda r:len(r['source'])):
        row = max(order, key=lambda r:(key(r), r['identity']))
        chosen[row['draw_identity']] = row
    for row in order:
        if len(chosen) == 4:break
        chosen.setdefault(row['draw_identity'], row)
    samples = [dict(id=k, source=r['source'], indices=r['indices'], values=r['values'],
                    training_row=r['identity']) for k,r in sorted(chosen.items())]
    retain(output/'private/training-services.json', samples)

    timing = base/'evaluation-timing'
    req = owned(timing/'request.json')
    from .progressive_research_feasibility import validate_request
    validate_request(req)
    if req.get('scope') != 'training-only prospective throughput; no panel exposure' or len(req['cases']) != 14:
        raise ValueError('original training-only timing request required')
    if req['support']!=support['identity']:
        raise ValueError('historical timing used different training support')
    originals_samples=_json(checked_file(timing,req['samples']))
    for sample in originals_samples:
        original=allrows.get(sample['id'])
        if original is None or any(sample[k]!=original[k] for k in ('indices','values')):
            raise ValueError('historical timing sample is not original training support')
    prompts=sorted([dict(id=r['id'],indices=r['indices'][:20],values=r['values'][:20])
                    for r in originals_samples],key=lambda r:r['id'])
    q, qroot = original_authority(root, req['qualification'])
    previous = _json(checked_file(qroot, q['source_closure']), maximum=64<<20)
    current = collect_snapshot(root)
    if digest(previous) != req['source'] or req['source'] != binding['source']:
        raise ValueError('original timing source differs from qualification')
    prefixes = ('src/oeis_learn/decoder/', 'src/oeis_learn/synthesis/', 'src/oeis_learn/fresh/',
                'src/oeis_learn/sandbox/', 'crates/')
    names = {'Dockerfile', 'pyproject.toml', 'uv.lock',
             'src/oeis_learn/pipeline_validation/progressive_evaluation.py',
             'src/oeis_learn/pipeline_validation/pilot_clocks.py',
             'src/oeis_learn/generalization/progressive/baselines.py',
             'src/oeis_learn/generalization/progressive/models.py'}
    protected = {k for k in set(previous)|set(current) if k in names or k.startswith(prefixes)}
    if any(previous.get(k) != current.get(k) for k in protected):
        raise ValueError('historical generation/service implementation changed; cannot extrapolate it')
    measurements = []
    for i, case in enumerate(req['cases']):
        gpath = confined(root,timing/'cases'/str(i)/'generation/candidates.json')
        spath = confined(root,timing/'cases'/str(i)/'scoring/scores.json')
        g, s = _json(gpath, maximum=64<<20), _json(spath, maximum=64<<20)
        if (seal(g) != g or seal(s) != s or g['recipe'] != digest(case['profile'])
                or g['methods'] != case['methods'] or g['model_control'] != case['control']
                or g['prompts']!=digest(prompts) or set(g['records'])!={r['id'] for r in originals_samples}
                or type(g['elapsed_us']) is not int or g['elapsed_us']<=0
                or s['candidates'] != reference(root,gpath)['sha256']):
            raise ValueError('original timing generation/scoring binding differs')
        measured = coverage(g,s)
        measurements.append(dict(index=i,case=case,generation_us=g['elapsed_us'],
            samples=len(g['records']),coverage=measured,
            generation=reference(root,gpath),scoring=reference(root,spath)))
    step = journal['steps'].get('prepare-research/evaluation-timing', {})
    if step.get('status') != 'FAILED':
        raise ValueError('this recovery requires the original failed timing stage')
    elapsed = float(step['elapsed_seconds'])
    if not 0 < elapsed <= req['seconds']-HOST_SECONDS:
        raise ValueError('original evaluation reservation has insufficient remaining time')
    census = owned(base/'census/census.json')
    roster = owned(base/'roster/private/role-roster.json', maximum=64<<20)
    # Public task costs need counts, not sequence IDs or scoring truth.
    tasks = [dict(id=r['id'],stage=r['stage'],operation=r['operation'],seconds=r['seconds'],
        members=len(r.get('members',[])),all_methods=r.get('all_methods',False),trial=r['trial'],
        control=r.get('control'),generation=r.get('generation'),
        token_target=r.get('token_target'),parent=r.get('parent'),domain=r.get('domain'))
        for r in roster['tasks']]
    result = seal(dict(version='progressive-service-inputs/v1',
        donor_session=reference(root,folder/'session.json'),donor_journal=reference(root,folder/'journal.json'),
        support=materials['support'],materials=reference(root,base/'materials/materials.json'),
        native=census['native'],source=digest(current),original_source=req['source'],
        original_request=reference(root,timing/'request.json'),original_request_identity=req['identity'],
        prior_timing_seconds=elapsed,additional_seconds=HOST_SECONDS,
        original_total_seconds=req['seconds'],historical_measurements=measurements,tasks=tasks,
        samples=reference(output,output/'private/training-services.json'),
        scope='training-only planning; no panel exposure, scientific execution or HIP authority'))
    retain(output/'service-inputs.json',result)
    return result


def plan(root, output, inputs, image, recipient):
    """A donor has one finite diagnostic continuation, not one per output path."""
    root, output = Path(root), Path(output)
    req = seal(dict(version=VERSION,source=inputs['source'],image=image,native=inputs['native'],
        inputs=inputs['identity'],samples=inputs['samples'],seconds=SECONDS,
        cases=[dict(index=i,sample=i//2,repeat=i%2) for i in range(8)],
        limits=dict(raw_seconds=4,fit_seconds=4,score_seconds=65,case_seconds=CASE_SECONDS),
        scope='training-only canonical-program service diagnostic; never learned capability'))
    claim = confined(root,root/'run-state/discovery-service-continuations'/(inputs['original_request_identity'][7:]+'.json'),exists=False)
    claim.parent.mkdir(parents=True,exist_ok=True)
    owner = seal(dict(version=VERSION,recipient=str(Path(recipient).relative_to(root)),request=req['identity']))
    if claim.exists():
        if claim.is_symlink() or read(claim) != owner:
            raise ValueError('original service continuation already owned; no sibling allowance reset')
    else:
        with claim.open('x') as stream:
            json.dump(owner,stream,sort_keys=True);stream.flush();os.fsync(stream.fileno())
        fd=os.open(claim.parent,os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    retain(output/'request.json',req)
    return req


def validate_request(req):
    fields={'version','identity','source','image','native','inputs','samples','seconds','cases','limits','scope'}
    if (set(req)!=fields or seal(req)!=req or req['version']!=VERSION or req['seconds']!=SECONDS
            or req['cases']!=[dict(index=i,sample=i//2,repeat=i%2) for i in range(8)]
            or req['limits']!=dict(raw_seconds=4,fit_seconds=4,score_seconds=65,case_seconds=CASE_SECONDS)
            or req['scope']!='training-only canonical-program service diagnostic; never learned capability'):
        raise ValueError('closed bounded training-only service request required')
    return req


def run_one(root,index):
    """Fresh native owners for fitting, raw admission and full-horizon scoring."""
    from oeis_learn.synthesis.client import SynthesisClient
    from oeis_learn.synthesis.admission import VisiblePrompt
    from oeis_learn.decoder.bigint_codec import fitting_policy
    from oeis_learn.fresh.bigint_diagnostics import _raw, _seal, score
    root=Path(root);req=validate_request(read(root/'request.json'));began=time.monotonic()
    samples=_json(checked_file(root,req['samples']))
    case=req['cases'][index];row=samples[case['sample']]
    prompt=VisiblePrompt(tuple(row['indices'][:20]),tuple(row['values'][:20]))
    timings={};fit={};outcome=None;errors=[]
    try:
        for phase in ('fit','raw','score'):
            with SynthesisClient(digest([req['identity'],index,phase])) as client:
                if client.runtime_identity('bigint-body/v1') != req['native']:
                    raise ValueError('original native runtime identity differs')
                start=time.monotonic()
                if phase=='fit':
                    deadline=start+req['limits']['fit_seconds']
                    try:
                        original=client.fit(row['source'],prompt,policy=fitting_policy(row['source']),deadline=deadline)
                        fit={k:original.get(k) for k in ('complete','outcome','reason')}
                        if fit['complete'] is not True:
                            errors.append(dict(phase='fit',error='INCOMPLETE',message=str(fit)))
                    except Exception as error:
                        # Raw admission/scoring use different owners and do not
                        # depend on fitting. Keep their diagnostic opportunity.
                        errors.append(dict(phase='fit',error=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
                        fit=dict(complete=False,outcome='service_error',reason=type(error).__name__)
                    fit['within_deadline']=time.monotonic()<deadline
                elif phase=='raw':
                    chosen,status=_raw(client,row['source'],prompt,start+req['limits']['raw_seconds'])
                    if chosen is None or status['complete'] is not True:
                        raise ValueError('original training witness did not pass independent visible admission: '+str(status))
                    selection=_seal([chosen],prompt,cost=0)
                else:
                    truth={row['id']:{k:row[k] for k in ('id','indices','values')}}
                    result=score({row['id']:dict(selections={'equal_budget':selection},
                        arm_evidence={'equal_budget':dict(complete=True,reason='original_training_witness')})},
                        truth,client,deadline=start+req['limits']['score_seconds'])
                    outcome=result['outcomes'][row['id']]['equal_budget']
                    if outcome['complete'] is not True or outcome['success'] is not True:
                        raise ValueError('original training witness failed independent full-horizon scoring')
                timings[phase+'_us']=max(1,int((time.monotonic()-start)*1e6))
        value=seal(dict(version=VERSION,status='FAIL' if errors else 'PASS',request=req['identity'],case=case,
            training_row=row['training_row'],native=req['native'],measurements=timings,fit=fit,
            errors=errors,
            scoring=outcome['scoring'],actual_elapsed_us=max(1,int((time.monotonic()-began)*1e6)),
            scope='known training witness; no learned capability or HIP evidence'))
    except Exception as error:
        value=seal(dict(version=VERSION,status='FAIL',request=req['identity'],case=case,
            error=type(error).__name__,message=str(error),traceback=traceback.format_exc(),
            measurements=timings,fit=fit,actual_elapsed_us=max(1,int((time.monotonic()-began)*1e6))))
        retain(root/'cases'/str(index)/'producer.json',value)
        raise
    retain(root/'cases'/str(index)/'producer.json',value)
    return value


def run_inside(root, seconds):
    from .progressive_acceptance import _bounded_process
    from oeis_learn.experiments.source_snapshot import source_snapshot
    root=Path(root);req=validate_request(read(root/'request.json'));began=time.monotonic()
    if req['version']!=VERSION or req['source']!=digest(source_snapshot()) or not 0<seconds<=SECONDS:
        raise ValueError('current bounded service diagnostic required')
    if req['image']!=os.environ.get('OEIS_QUALIFICATION_IMAGE'):
        raise ValueError('original service image required')
    previous=read(root/'service-cases.json') if (root/'service-cases.json').exists() else None
    if previous and previous['request']!=req['identity']:
        raise ValueError('original service case journal differs')
    originals={r['index']:r['producer'] for r in previous['cases'] if 'producer' in r} if previous else {}
    rows=[]
    for case in req['cases']:
        i=case['index'];path=root/'cases'/str(i)/'producer.json'
        # A failed or interrupted original is not retried merely to get a pass.
        if path.exists():
            if i in originals:checked_file(root,originals[i])
            value=read(path)
            if value.get('request')!=req['identity'] or value.get('case')!=case:
                raise ValueError('original service producer changed scope')
        elif (root/'cases'/str(i)/'started.json').exists():
            value=seal(dict(version=VERSION,status='FAIL',request=req['identity'],case=case,
                error='INTERRUPTED',message='original case termination has no producer; no automatic retry'))
            retain(path,value)
        else:
            left=seconds-(time.monotonic()-began)-30
            if left<CASE_SECONDS:
                rows.append(dict(index=i,status='BLOCKED',reason='remaining diagnostic allowance'));continue
            retain(root/'cases'/str(i)/'started.json',seal(dict(request=req['identity'],case=case)))
            code="from oeis_learn.pipeline_validation.progressive_service_calibration import run_one;import sys;run_one(sys.argv[1],int(sys.argv[2]))"
            try:
                result=_bounded_process([sys.executable,'-c',code,str(root),str(i)],cwd=root,env=dict(os.environ),
                    log=root/f'case-{i}.log',seconds=CASE_SECONDS,maximum=4<<20)
                if result or not path.exists():raise RuntimeError('service subprocess exited '+str(result))
            except Exception as error:
                if not path.exists():retain(path,seal(dict(version=VERSION,status='FAIL',request=req['identity'],case=case,
                    error=type(error).__name__,message=str(error),traceback=traceback.format_exc())))
            value=read(path)
        rows.append(dict(index=i,status=value['status'],producer=reference(root,path)))
        revise(root/'service-cases.json',seal(dict(version=VERSION,request=req['identity'],cases=rows,status='RUNNING')))
    result=seal(dict(version=VERSION,request=req['identity'],cases=rows,
        status='PASS' if len(rows)==8 and all(r['status']=='PASS' for r in rows) else 'FAIL'))
    revise(root/'service-cases.json',result)
    return result


def report(root, inputs):
    """Counterfactual reservations only; cannot freeze or launch a campaign."""
    import math
    from .research_limits import TIMING_MULTIPLIER
    root=Path(root);req=validate_request(read(root/'request.json'))
    if req['inputs']!=inputs['identity'] or seal(inputs)!=inputs:
        raise ValueError('original service inputs required')
    result=read(root/'service-cases.json') if (root/'service-cases.json').exists() else None
    producers=[];deficits=[]
    if not result or result['request']!=req['identity']:
        deficits.append('service_case_batch_missing')
    else:
        if [r['index'] for r in result['cases']]!=list(range(len(result['cases']))) or len(result['cases'])>8:
            raise ValueError('ordered registered service result roster required')
        deficits.extend('service_case:'+str(i) for i in range(len(result['cases']),8))
        for row in result['cases']:
            if 'producer' not in row:
                deficits.append('service_case:'+str(row['index']));continue
            p=read(checked_file(root,row['producer']))
            if p.get('request')!=req['identity'] or p.get('case')!=req['cases'][row['index']]:
                raise ValueError('original service measurement scope differs')
            if p['status']!='PASS':
                deficits.append('service_case:'+str(row['index']));continue
            if (set(p['measurements'])!={'fit_us','raw_us','score_us'}
                    or any(type(v) is not int or v<=0 for v in p['measurements'].values())
                    or p['scoring']['complete'] is not True or p['native']!=req['native']):
                raise ValueError('original positive service measurement required')
            producers.append(p)
    maxima={phase:max((p['measurements'][phase] for p in producers),default=None)
            for phase in ('fit_us','raw_us','score_us')}
    # Owner startup/cleanup is outside the timed method bodies. Retain and pad it.
    startup=max((max(0,p['actual_elapsed_us']-sum(p['measurements'].values())) for p in producers),default=0)
    historical=inputs['historical_measurements']
    by_id={r['id']:r for r in inputs['tasks']};tasks=[]
    for row in inputs['tasks']:
        if row['operation'] not in {'generate','score'}:continue
        shared=row['trial'] and row['trial']['phase'] not in {'confirmation','selection'}
        reserve=1 if shared else 1800
        n=row['members'];measured=None;unexercised=0
        if row['operation']=='generate':
            candidates=[r for r in historical if (len(r['case']['methods'])>1)==row['all_methods']]
            if row['control'] in {'random','retained_D'} and row['all_methods']:
                candidates=[r for r in candidates if r['case']['control']==row['control']]
            if not candidates:raise ValueError('historical model/control measurement missing')
            pairs=math.ceil(n/2)
            measured=max(r['generation_us'] for r in candidates)*pairs
            unexercised=max(r['coverage']['unmeasured_native_reserve_us'] for r in candidates)*pairs
        elif maxima['score_us'] is not None:
            generation=by_id[row['generation']]
            # Every selected arm might return a different candidate; never assume
            # repeated-source/native cache hits from this diagnostic witness.
            arms=8 if generation['all_methods'] else 3
            measured=startup+n*arms*maxima['score_us']
        padded=None if measured is None else TIMING_MULTIPLIER*(measured+unexercised)
        proposed=None if padded is None else math.ceil(padded/1e6)+reserve+1
        tasks.append(dict(task=row['id'],stage=row['stage'],operation=row['operation'],members=n,
            trial=row['trial'],allocated_seconds=row['seconds'],measured_extrapolation_us=measured,
            unexercised_service_reserve_us=unexercised,padded_estimate_us=padded,
            proposed_seconds=proposed,needs_partition=proposed is not None and proposed>86400,
            fits_original=proposed is not None and proposed<=row['seconds']))
    original_deficits=[r['task'] for r in tasks if not r['fits_original']]
    stages={stage:dict(original_evaluation_seconds=sum(r['allocated_seconds'] for r in tasks if r['stage']==stage),
        proposed_evaluation_seconds=None if any(r['proposed_seconds'] is None for r in tasks if r['stage']==stage)
            else sum(r['proposed_seconds'] for r in tasks if r['stage']==stage))
        for stage in ('capability','diversity','tuning')}
    trials=[]
    for ordinal in range(16):
        children=[r for r in inputs['tasks'] if r['trial'] and r['trial']['ordinal']==ordinal
                  and r['trial']['phase'] not in {'confirmation','selection'}]
        estimates={r['task']:r['proposed_seconds'] for r in tasks}
        selected=[estimates.get(r['id'],r['seconds']) for r in children]
        proposed=None if None in selected else 1800+sum(selected)
        trials.append(dict(ordinal=ordinal,original_seconds=7200,proposed_seconds=proposed,
            fits_24h=proposed is not None and proposed<=86400))
    # Token targets on continuations are cumulative. Count only new work; neither
    # a reservation nor a worst-input calibration is expected training throughput.
    training=[]
    for row in inputs['tasks']:
        if row['operation']!='train':continue
        parent=by_id[row['parent']] if row.get('parent') else None
        previous=parent['token_target'] if parent else 0
        delta=row['token_target']-previous
        if delta<=0:raise ValueError('increasing original training targets required')
        training.append(dict(task=row['id'],stage=row['stage'],domain=row['domain'],
            incremental_tokens=delta,expected_seconds=None,
            reason='representative charged-token/update throughput not measured'))
    runtime=dict(status='INCOMPLETE',expected_total_seconds=None,
        expected_training_seconds=None,training_tasks=training,
        all_eligible_training_tokens=sum(r['incremental_tokens'] for r in training),
        stages={stage:dict(
            all_eligible_training_tokens=sum(r['incremental_tokens'] for r in training if r['stage']==stage),
            observed_generation_projection_seconds=sum(r['measured_extrapolation_us']/1e6
                for r in tasks if r['stage']==stage and r['operation']=='generate'),
            positive_witness_scoring_projection_seconds=None if maxima['score_us'] is None else
                sum(r['measured_extrapolation_us']/1e6 for r in tasks if r['stage']==stage and r['operation']=='score'))
            for stage in stages},
        excludes=['image/setup and remaining qualification', 'exact-input calibration',
            'training updates, refresh probes, checkpointing and restore',
            'unexercised generated-program services', 'reporting and final assessment'],
        interpretation='Unpadded planning projections if all eligible roles execute; not a statistical expected runtime. '
            'Generation uses slowest compatible original measurement; scoring uses slowest known positive training witness. '
            'Do not add safety multipliers, finalization reserves or hard caps to an expected runtime. '
            'Early stopping reduces work, but its probability is unmeasured.',
        next_measurement='representative HIP charged tokens per update and elapsed time, with setup, '
            'steady updates, probes, checkpoint publication/restore and cleanup timed separately')
    value=seal(dict(version='progressive-service-budget-review/v1',
        status='MEASURED' if not deficits else 'INCOMPLETE',executable=False,research_ready=False,
        original_request=inputs['original_request'],service_request=req['identity'],
        prior_timing_seconds=inputs['prior_timing_seconds'],additional_reserved_seconds=HOST_SECONDS,
        service_maxima_us=maxima,owner_overhead_us=startup,measurement_deficits=deficits,
        original_budget_deficits=original_deficits,tasks=tasks,stages=stages,trials=trials,
        runtime_estimate=runtime,
        safety_multiplier=TIMING_MULTIPLIER,assessment_exposed=False,
        next_action='prospective schedule/domain revision; no automatic training, panel reduction or gate change',
        limitations=['training witnesses measure service paths, not all future generated programs',
            'historical generation timing remains original-source evidence, not current HIP qualification',
            'proposals exceeding 24h require partition or a smaller prospective design',
            'aggregate ancillary/training caps require a new validated freeze before execution']))
    revise(root/'service-budget-review.json',value)
    return value
