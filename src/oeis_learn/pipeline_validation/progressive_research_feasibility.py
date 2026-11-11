"""Training-only evaluation timing before any scientific panel/model scoring.

The installed canonical generator/scorer run unchanged. These measurements are
resource estimates, never research observations or guarantees about trained models.
"""
from .research_limits import case_allowance, HEALTH_SECONDS, TIMING_MULTIPLIER, padded_us

import json
import os
from pathlib import Path
import sys
import time
import traceback

from oeis_learn.generalization.progressive.contracts import atomic_write,digest,load,reference,seal,verify_refs
from .identity import confined

VERSION='progressive-research-evaluation-request/v1'
SECONDS=4*3600
METHODS=('correct','shuffled','absent','template','polynomial','linear_recurrence')


def request(root, output, support, profiles, qualification, *, checkpoint):
    """Select the largest original training inputs, without reading held-out truth."""
    from .progressive_handoff import original_authority,validate_device_proofs
    from oeis_learn.generalization.progressive.training_rows import validate_row
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    root=Path(root);output=Path(output)
    q,qroot=original_authority(root,qualification)
    source=digest(collect_snapshot(root))
    # The original exact engine/device scope comes from the qualified matrix.
    matrix_refs=[]
    for ref,raw in zip(q['producer']['artifacts'],verify_refs(qroot,q['producer']['artifacts'],max_bytes=512<<20)):
        try:value=json.loads(raw)
        except (ValueError,UnicodeError):continue
        if isinstance(value,dict) and value.get('version')=='progressive-hip-matrix/v1':matrix_refs.append(value)
    if len(matrix_refs)!=1:raise ValueError('one original selected matrix required')
    matrix,_=validate_device_proofs(qroot,q,source=source,image=q['images'][1],engine=matrix_refs[0]['engine'])
    qualified={c['profile']['identity'] for c in matrix['cases']}
    if not profiles or {p['identity'] for p in profiles}!=qualified:
        raise ValueError('complete original selected evaluation model scope required')
    if seal(support)!=support or support.get('diagnostic') is not False:
        raise ValueError('original native training support required')
    rows={r['draw_identity']:r for shard in support['shards'] for r in shard}
    for row in rows.values():validate_row(row,native_verified=True)
    if len(rows)<2:raise ValueError('two distinct original training timing inputs required')
    # Exact input size and target length are both fixed before any model score.
    ordered=sorted(rows.values(),key=lambda r:(max(len(v) for v in r['values'][:20]),len(r['tokens']),r['identity']),reverse=True)
    chosen=[ordered[0],max((r for r in ordered if r is not ordered[0]),key=lambda r:(len(r['tokens']),r['identity']))]
    samples=[dict(id=r['draw_identity'],indices=r['indices'],values=r['values']) for r in chosen]
    if (output/'private/training-samples.json').exists():raise ValueError('fresh prospective timing request required')
    atomic_write(output/'private/training-samples.json',samples)
    sample_ref=reference(output,output/'private/training-samples.json')
    cases=[dict(profile=p,control='random',methods=['correct']) for p in profiles]
    base=[p for p in profiles if p['shape']=='reference' and p['model']['dropout']==0]
    if len(base)!=1:raise ValueError('one fixed reference scope required')
    cases.extend(dict(profile=base[0],control=control,methods=list(METHODS)) for control in ('random','retained_D'))
    # Copy no checkpoint: the pinned installed role receives the original read-only mount.
    value=seal(dict(version=VERSION,source=source,image=q['images'][1],engine=matrix['engine'],hardware=matrix['hardware'],
        execution_policy=q['runtime']['execution_policy'],qualification=qualification,support=support['identity'],
        samples=sample_ref,checkpoint=checkpoint,cases=[seal(c) for c in cases],seconds=SECONDS,
        scope='training-only prospective throughput; no panel exposure',safety_multiplier=TIMING_MULTIPLIER))
    atomic_write(output/'request.json',value)
    return value


def validate_request(value):
    fields={'version','identity','source','image','engine','hardware','execution_policy','qualification',
            'support','samples','checkpoint','cases','seconds','scope','safety_multiplier'}
    if (set(value)!=fields or seal(value)!=value or value.get('version')!=VERSION or value.get('seconds')!=SECONDS
            or value.get('safety_multiplier')!=TIMING_MULTIPLIER or len(value.get('cases',[]))!=14
            or len({c['identity'] for c in value['cases']})!=14):
        raise ValueError('complete bounded prospective evaluation request required')
    from oeis_learn.generalization.progressive.models import validate_profile
    for case in value['cases']:
        validate_profile(case['profile'])
        if (seal(case)!=case or set(case)!={'identity','profile','control','methods'}
                or case['control'] not in {'random','retained_D'}
                or case['methods'] not in [['correct'],list(METHODS)]):
            raise ValueError('fixed canonical evaluation case required')
    primary=[c for c in value['cases'] if c['methods']==['correct']]
    controls=[c for c in value['cases'] if c['methods']==list(METHODS)]
    if (len(primary)!=12 or len({c['profile']['identity'] for c in primary})!=12
            or any(c['control']!='random' for c in primary)
            or len(controls)!=2 or {c['control'] for c in controls}!={'random','retained_D'}
            or any(c['profile']['shape']!='reference' or c['profile']['model']['dropout']!=0 for c in controls)
            or controls[0]['profile']!=controls[1]['profile']
            or controls[0]['profile']['identity'] not in {c['profile']['identity'] for c in primary}):
        raise ValueError('twelve primary profiles and two fixed reference controls required')
    return value


def run_one(root,index,seconds):
    """Fresh installed process; real HIP and native execution, no forced candidates."""
    from .progressive_qualification import require_hip
    from .progressive_evaluation import generate,score
    from oeis_learn.synthesis.client import SynthesisClient
    root=Path(root);req=validate_request(load(root/'request.json'));case=req['cases'][index]
    torch=require_hip();torch.cuda.reset_peak_memory_stats()
    rows=json.loads(verify_refs(root,[req['samples']])[0])
    prompts=[dict(id=r['id'],indices=r['indices'][:20],values=r['values'][:20]) for r in rows]
    output=root/'cases'/str(index);output.mkdir(parents=True)
    began=time.monotonic()
    with SynthesisClient(digest(['research-timing',req['identity'],index])) as client:
        kwargs=dict(model_control=case['control'],device='cuda',methods=tuple(case['methods']),
            seed=case['profile']['model']['seed'] if case['control']=='random' else None,
            checkpoint=req['checkpoint'] if case['control']=='retained_D' else None)
        generated=generate(Path('/work/inputs'),'generation',prompts,case['profile'],client=client,
            output_root=output,seconds=min(21600,seconds),**kwargs)
        generated_us=max(1,int((time.monotonic()-began)*1000000))
        remaining=seconds-(time.monotonic()-began)
        if remaining<=0:raise TimeoutError('evaluation timing exhausted before scorer')
        score_began=time.monotonic()
        scored=score(output,generated,rows,client=client,output='scoring',seconds=min(21600,remaining))
        scoring_us=max(1,int((time.monotonic()-score_began)*1000000))
    original=load(confined(output,generated['path']),maximum=512<<20)
    selected=['equal_budget' if m=='correct' else m for m in case['methods']]
    if (any(row['arm_evidence'][m].get('complete') is not True for row in original['records'].values() for m in selected)
            or set(scored['outcomes'])!=set(original['records'])
            or any(row[m].get('complete') is not True for row in scored['outcomes'].values() for m in selected)):
        raise ValueError('incomplete portfolio cannot establish evaluation throughput')
    # Preserve all producer attempts and phase clocks, but no outcomes influence
    # the design. Exceeding a portfolio cap is recorded, never counted as a win.
    from .progressive_evidence import generation_timing
    atomic_write(output/'timing.json',generation_timing(original))
    value=seal(dict(version='progressive-research-evaluation-case/v1',request=req['identity'],case=case['identity'],
        device='actual_HIP',status='PASS',samples=len(rows),generation_us=generated_us,scoring_us=scoring_us,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        producer=generated,scoring=reference(output,output/'scoring/scores.json'),timing=reference(output,output/'timing.json'),
        actual_elapsed_us=max(1,int((time.monotonic()-began)*1000000))))
    atomic_write(output/'producer.json',value)
    return value


def run_inside(root):
    from .progressive_qualification import require_hip,hardware_facts
    from .generalization_execution import verify_execution
    from oeis_learn.experiments.source_snapshot import source_snapshot
    from .progressive_acceptance import _bounded_process,_record_failure
    root=Path(root);req=validate_request(load(root/'request.json'));began=time.monotonic()
    require_hip()
    if (req['source']!=digest(source_snapshot()) or req['image']!=os.environ.get('OEIS_QUALIFICATION_IMAGE')
            or req['engine']!=os.environ.get('OEIS_QUALIFICATION_ENGINE')
            or req['hardware']!=hardware_facts() or req['execution_policy']!=verify_execution()):
        raise ValueError('current original HIP timing scope required')
    outcomes=[];failures=[];blocked=None
    for i,case in enumerate(req['cases']):
        row=dict(index=i,case=case['identity'],status='BLOCKED' if blocked else 'RUNNING')
        if blocked:row['reason']=blocked
        outcomes.append(row)
        atomic_write(root/'case-events'/str(i)/'running.json',seal(row))
        if not blocked:
            try:
                remaining=req['seconds']-(time.monotonic()-began)
                if remaining<=1:raise TimeoutError('aggregate evaluation timing deadline')
                seconds=case_allowance(remaining)
                code='import sys;from oeis_learn.pipeline_validation.progressive_research_feasibility import run_one;run_one(sys.argv[1],int(sys.argv[2]),float(sys.argv[3]))'
                code=_bounded_process([sys.executable,'-c',code,str(root),str(i),str(seconds)],cwd=root,
                    env=dict(os.environ),log=root/f'case-{i}.log',seconds=seconds,maximum=4<<20)
                if code:raise RuntimeError('evaluation timing subprocess failed: '+str(code))
                ref=reference(root,root/'cases'/str(i)/'producer.json')
                producer=verify_case(root,req,case,ref)
                if (seal(producer)!=producer or producer['request']!=req['identity']
                        or producer['case']!=case['identity'] or producer['device']!='actual_HIP' or producer['status']!='PASS'):
                    raise ValueError('original complete evaluation timing producer required')
                row.update(status='PASS',producer=ref)
            except (OSError,ValueError,TimeoutError,RuntimeError) as error:
                row.update(status='FAIL',error=type(error).__name__,message=str(error))
                _record_failure(root,failures,dict(case=case['identity'],error=type(error).__name__,traceback=traceback.format_exc()))
                remaining=req['seconds']-(time.monotonic()-began)
                try:
                    if remaining<=1 or _bounded_process([sys.executable,'-c',
                        'from oeis_learn.pipeline_validation.progressive_qualification import require_hip;t=require_hip();t.cuda.synchronize()'],
                        cwd=root,env=dict(os.environ),log=root/f'health-{i}.log',seconds=min(HEALTH_SECONDS,remaining),maximum=65536):
                        raise RuntimeError('device cleanup/health unavailable')
                except (OSError,ValueError,TimeoutError,RuntimeError):blocked='DEVICE_BLOCKED'
        atomic_write(root/'case-events'/str(i)/'outcome.json',seal(row))
        atomic_write(root/'case-events'/str(i)/'evaluation-progress.json',seal(dict(version='progressive-research-evaluation-cases/v1',
            request=req['identity'],status='FAIL' if failures else 'RUNNING',cases=outcomes)))
    result=seal(dict(version='progressive-research-evaluation-run/v1',request=req['identity'],
        status='PASS' if all(r['status']=='PASS' for r in outcomes) else 'FAIL',cases=outcomes,
        actual_elapsed_us=max(1,int((time.monotonic()-began)*1000000))))
    atomic_write(root/'evaluation-cases.json',result)
    if failures:atomic_write(root/'failures.json',failures)
    if result['status']!='PASS':raise ValueError('evaluation calibration incomplete; original diagnostics retained')
    return result


def verify_case(root,req,case,ref):
    """Replay the complete original generation/scoring/timing artifact closure."""
    from .progressive_evidence import generation_timing
    root=Path(root)
    p=json.loads(verify_refs(root,[ref])[0]);base=confined(root,ref['path']).parent
    fields={'version','identity','request','case','device','status','samples',
        'generation_us','scoring_us','peak_allocated_bytes','producer','scoring','timing','actual_elapsed_us'}
    if (set(p)!=fields or seal(p)!=p or p['version']!='progressive-research-evaluation-case/v1'
            or p['request']!=req['identity'] or p['case']!=case['identity']
            or p['device']!='actual_HIP' or p['status']!='PASS' or p['samples']!=2
            or type(p['peak_allocated_bytes']) is not int or p['peak_allocated_bytes']<0):
        raise ValueError('closed original measured evaluation case required')
    generated,scored,timing=[json.loads(raw) for raw in verify_refs(base,[p[k] for k in ('producer','scoring','timing')],max_bytes=512<<20)]
    rows=json.loads(verify_refs(root,[req['samples']])[0])
    if len(rows)!=2 or len({r['id'] for r in rows})!=2 or any(
            set(r)!={'id','indices','values'} or len(r['indices'])!=100 or len(r['values'])!=100 for r in rows):
        raise ValueError('two original training-only timing members required')
    prompts=sorted([dict(id=r['id'],indices=r['indices'][:20],values=r['values'][:20]) for r in rows],key=lambda r:r['id'])
    fields={'version','identity','model','checkpoint','model_control','recipe','methods','prompts','records','clock','counts','elapsed_us'}
    if (set(generated)!=fields or seal(generated)!=generated or generated['version']!='discovery-candidates/v1'
            or generated['recipe']!=digest(case['profile']) or generated['methods']!=case['methods']
            or generated['model_control']!=case['control']
            or generated['checkpoint']!=(req['checkpoint']['sha256'] if case['control']=='retained_D' else None)
            or generated['prompts']!=digest(prompts) or set(generated['records'])!={r['id'] for r in rows}
            or type(generated['elapsed_us']) is not int or not 0<=generated['elapsed_us']<=p['generation_us']):
        raise ValueError('original timing generation member/model/profile/method scope differs')
    fields={'version','identity','candidates','outcomes','denominator','scientific_status','claim'}
    if (set(scored)!=fields or seal(scored)!=scored or scored['version']!='discovery-scoring/v1'
            or scored['candidates']!=p['producer']['sha256'] or scored['denominator']!=2
            or scored['scientific_status']!='UNMEASURED' or scored['claim']!='finite_agreement_only'
            or set(scored['outcomes'])!=set(generated['records'])):
        raise ValueError('original independent timing scorer binding differs')
    methods=['equal_budget' if m=='correct' else m for m in case['methods']]
    for member,row in generated['records'].items():
        for method in methods:
            if (row.get('arm_evidence',{}).get(method,{}).get('complete') is not True
                    or scored['outcomes'][member].get(method,{}).get('complete') is not True):
                raise ValueError('incomplete original timing portfolio')
    if timing!=generation_timing(generated):
        raise ValueError('original timing phase clocks differ')
    return p


def estimates(root,request_value,roster,*,publish=True):
    """Conservative sampled estimates, always subject to original hard task caps."""
    root=Path(root);req=validate_request(request_value);result=load(root/'evaluation-cases.json')
    if (seal(result)!=result or result['request']!=req['identity'] or result['status']!='PASS'
            or len(result['cases'])!=len(req['cases'])):
        raise ValueError('complete original evaluation measurement batch required')
    times={};receipts=[]
    for index,(expected,row) in enumerate(zip(req['cases'],result['cases'])):
        if row.get('status')!='PASS' or row.get('index')!=index or row.get('case')!=expected['identity']:
            raise ValueError('complete ordered original evaluation cases required')
        p=verify_case(root,req,expected,row['producer'])
        if (p.get('version')!='progressive-research-evaluation-case/v1' or seal(p)!=p or p['case']!=expected['identity'] or p['request']!=req['identity']
                or p['device']!='actual_HIP' or p['status']!='PASS' or p['samples']!=2
                or any(type(p[k])is not int or p[k]<=0 for k in ('generation_us','scoring_us','actual_elapsed_us'))
                or p['generation_us']+p['scoring_us']>p['actual_elapsed_us']+2
                or p['actual_elapsed_us']>req['seconds']*1000000):
            raise ValueError('original measured evaluation case differs')
        key='all' if len(expected['methods'])>1 else 'primary'
        for operation,field in [('generate','generation_us'),('score','scoring_us')]:
            times[key,operation]=max(times.get((key,operation),0),p[field])
        receipts.append(row['producer'])
    deficits=[];tasks=[]
    for row in roster['tasks']:
        if row['operation'] not in {'generate','score'}:continue
        shared=row['trial'] and row['trial']['phase'] not in {'confirmation','selection'}
        available=(row['seconds']-(1 if shared else 1800)-1)*1000000
        # Each measurement contains two members and model startup. Charging the
        # whole batch per pair, then applying the frozen 4x margin, deliberately repeats startup cost.
        estimate=padded_us(times['all' if row.get('all_methods') or row['operation']=='score' else 'primary',row['operation']]*((len(row['members'])+1)//2))
        tasks.append(dict(task=row['id'],estimated_us=estimate,available_us=available))
        if estimate>available:deficits.append('evaluation_time:'+row['id'])
    value=seal(dict(version='progressive-research-evaluation-feasibility/v1',request=req['identity'],
        roster=roster['identity'],status='READY' if not deficits else 'BLOCKED',deficits=deficits,tasks=tasks,
        producers=receipts,assessment_exposed=False,
        limits='training-only sampled estimate with 4x margin; later task clocks remain hard bounds'))
    if publish:atomic_write(root/'evaluation-feasibility.json',value)
    return value


def execution_contract(root, output, *, request_path, roster_path, report_path):
    root=Path(root)
    req=load(request_path);roster=load(roster_path,maximum=4<<20)
    expected=estimates(Path(request_path).parent,req,roster,publish=False)
    if expected['status']!='READY' or load(report_path)!=expected:
        raise ValueError('original feasible evaluation design required')
    value=seal(dict(version='progressive-research-execution-contract/v1',
        request=reference(root,request_path),roster=reference(root,roster_path),
        evaluation=reference(root,report_path)))
    atomic_write(output,value)
    return value


def verify_contract(root, ref, runtime):
    value=json.loads(verify_refs(root,[ref],max_bytes=4<<20)[0])
    if (set(value)!={'version','identity','request','roster','evaluation'} or seal(value)!=value
            or value['version']!='progressive-research-execution-contract/v1'):
        raise ValueError('closed research execution contract required')
    req,roster,report=[json.loads(raw) for raw in verify_refs(root,[value[k] for k in ('request','roster','evaluation')],max_bytes=8<<20)]
    q=json.loads(verify_refs(root,[req['qualification']])[0])
    source=json.loads(verify_refs(confined(root,req['qualification']['path']).parent,[q['source_closure']],max_bytes=64<<20)[0])
    if (req['qualification'] not in runtime['payload']['qualification'] or req['source']!=digest(source)
            or req['image']!=runtime['payload']['scope']['images'][1]
            or req['execution_policy']!=runtime['payload']['scope']['execution_policy']):
        raise ValueError('research evaluation qualification/source scope differs')
    expected=estimates(confined(root,value['request']['path']).parent,req,roster,publish=False)
    if expected!=report or report['status']!='READY':
        raise ValueError('original measured evaluation design is not feasible')
    return value
