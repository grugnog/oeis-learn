"""Temporary paired wiring smoke using prepared data and canonical operations.

No weights or measurements from this diagnostic initialize/select the full pilot.
"""
import json
import sys
import time
from pathlib import Path
from oeis_learn.pipeline_validation.identity import seal,write,confined,digest
from oeis_learn.pipeline_validation.generalization_preparation import reference
from oeis_learn.generalization.contracts import verify_artifact
from oeis_learn.generalization.budgets import charge_tokens
from oeis_learn.decoder.bigint_codec import encode_body

LIMITS={'preflight':1200,'prepare':1800,'train-static':2100,'train-refreshed':2100,
        'generate-static':2400,'score-static':2400,'generate-refreshed':2400,'score-refreshed':2400}


def batch_budget(static,refreshed):
    """Both arms must afford their first three complete batches."""
    if len(static)!=2 or len(refreshed)!=2 or any(type(n) is not int or n<=0 for n in static+refreshed):
        raise ValueError('two positive batch charges per shard required')
    cap=sum(static)+max(static[0],refreshed[0])
    if not 0<cap<=16384 or cap>sum(static+refreshed):raise ValueError('bounded smoke capacity shortfall')
    return cap


def batch_charges(rows,expected_rows=32):
    if len(rows)!=expected_rows or expected_rows%16:raise ValueError('complete fixed-size shard required')
    return [charge_tokens([encode_body(r['source']) for r in rows[i:i+16]],bos=1,pad=0) for i in range(0,expected_rows,16)]


def reduced_blueprint(root,job,output,*,probe=False):
    """Freeze two32-row shards from the already native-admitted data."""
    from oeis_learn.generalization.preparation import scan,exclusion_record
    from oeis_learn.generalization.exclusions import ExclusionIndex
    from oeis_learn.pipeline_validation.generalization_preparation import draw_candidates
    cfg=json.loads(verify_artifact(root,job['blueprint']))
    design=json.loads(verify_artifact(root,job['design'],max_bytes=16<<20))
    if seal(design)!=design or design['identity'] not in cfg['hypothesis']:raise ValueError('bound original protocol required')
    if design['smoke']['version']!='paired-source-smoke/v1':raise ValueError('frozen smoke policy required')
    prepared=confined(root,job['manifest']).parent
    target=Path(output);target.mkdir(parents=True)
    rows_per_family=8 if probe else 1
    cells=[];training=[];exclusion=ExclusionIndex();seen=set()
    for original in cfg['cells']:
        if original['role']=='training' and len(training)==2:continue
        cell={k:v for k,v in original.items() if k not in {'refresh','candidates'}}
        if original['role']=='training':
            cell.update(count=32*rows_per_family,quotas={'family:'+d['family']:rows_per_family for d in design['descriptors']})
            cell['refresh']=dict(original['refresh'],draws=2048)
            candidates,_=draw_candidates(root,reference(confined(root,cfg['descriptors']),root),cell)
            choice,ledger=scan(candidates,**{k:cell[k] for k in ('role','holdout','count','quotas','structural','length_boundary')},
                execute=None,exclusions=exclusion,unique_sources=True,seen_sources=seen,workers=3,seconds=900)
            if not choice.complete:raise ValueError('smoke source selection incomplete')
            rows=list(choice.rows);training.append(rows);seen.update(digest(r['source']) for r in rows)
        else:
            rows=json.loads((prepared/'cells'/original['name']/'private/selected.json').read_bytes())
            exclusion=exclusion.add_many(exclusion_record(r,cell['role'],cell['holdout']) for r in rows)
        path=target/(cell['name']+'.json');write(path,rows)
        if original['role']=='training':
            # Preserve the canonical descriptor/cursor authority. A fixed candidate
            # list cannot satisfy the declared refresh arm's runtime verifier.
            cell['refresh']=dict(original['refresh'],draws=2048)
        else:cell['candidates']=str(path.relative_to(root))
        cells.append(cell)
    if len(training)!=2:raise ValueError('two prepared smoke shards required')
    charges=[batch_charges(rows,32*rows_per_family) for rows in training]
    from fresh_pilot_training_policy import budget
    cap=budget(charges) if probe else batch_budget(*charges)
    protocol=seal(dict(version='fresh-pilot-training-probe/v1' if probe else 'fresh-pilot-smoke/v1',parent_protocol=design['identity'],
        bank_seeds=design['bank_seeds'],role_seconds=1800,target_tokens=cap,
        first_shard_tokens=sum(charges[0]),minimum_updates=24 if probe else 3,selection_rows=0 if probe else 2,
        batch_charges=charges,training_sources=[digest([r['source'] for r in rows]) for rows in training],
        gate='finite changed weights; fresh rows consumed; all scoring outcomes recorded; zero accuracy threshold',
        purpose='software diagnostic only; never scientific comparison or initializer'))
    if probe:protocol=seal(dict(protocol,checkpoint_policy='hourly/v1',research_milestones=len(cfg['milestones']),
        gate='finite changed weights; refreshed exposure; complete timing windows; measured snapshot overhead'))
    write(target/'protocol.json',protocol)
    cfg.update(hypothesis='Paired source smoke '+protocol['identity'],cells=cells,
        milestones=[cap],budgets=dict(cfg['budgets'],target_tokens=cap,seconds=1800 if probe else 900),preparation_seconds=1800 if probe else 900,
        training={'static':dict(cells=[cells[-2]['name']],data_mode='static',batch_size=16),
                  'refreshed':dict(cells=[c['name'] for c in cells[-2:]],data_mode='refresh',batch_size=16)})
    write(target/'blueprint.json',cfg)
    return cfg,protocol


def verify_smoke_plan(base,study,runtime,protocol):
    """Check actual native-prepared training bytes before model initialization."""
    if seal(protocol)!=protocol or protocol['minimum_updates']!=3:raise ValueError('frozen three-update smoke plan required')
    checked=[]
    for arm,count in (('static',1),('refreshed',2)):
        view=runtime['training'][arm]
        if view['batch_size']!=16 or len(view['shards'])!=count:raise ValueError('smoke batch/shard layout changed')
        for i,ref in enumerate(view['shards']):
            rows=json.loads(verify_artifact(base,ref,max_bytes=8<<20))
            if digest([r['source'] for r in rows])!=protocol['training_sources'][i]:raise ValueError('native smoke selection differs from frozen source plan')
            if any(r['tokens']!=encode_body(r['source']) for r in rows):raise ValueError('smoke target token encoding differs')
            costs=batch_charges(rows)
            if costs!=protocol['batch_charges'][i]:raise ValueError('native smoke batch costs differ')
            if arm=='refreshed':checked.append(costs)
    cap=batch_budget(*checked)
    if cap!=protocol['target_tokens'] or cap!=study['budgets']['target_tokens'] or sum(checked[0])!=protocol['first_shard_tokens']:
        raise ValueError('smoke exposure differs from frozen budget')
    return dict(target_tokens=cap,first_shard_tokens=sum(checked[0]),minimum_updates=3,
        minimum_tokens_by_arm=dict(static=sum(checked[0])+checked[0][0],refreshed=sum(checked[0])+checked[1][0]))


def scoring_recorded(scoring,targets):
    """Accounted negatives/censoring are valid; missing or malformed output is not."""
    from fresh_pilot_report import CONTROLS
    outcomes=scoring['result']['outcomes']
    if set(outcomes)!={r['id'] for r in targets}:return False
    for row in outcomes.values():
        if set(row)!=set(CONTROLS):return False
        for value in row.values():
            if type(value.get('complete')) is not bool or type(value.get('success')) is not bool:return False
            if value['success'] and not value['complete']:return False
            if not isinstance(value.get('scoring'),dict) or not isinstance(value.get('upstream'),dict):return False
            if value['scoring'].get('outcome') in {'not_attempted','scoring_incomplete'}:return False
    return True


def run(api,job_path,output):
    from oeis_learn.pipeline_validation.generalization_inputs import assemble_study
    from oeis_learn.pipeline_validation.generalization_runtime import load
    from oeis_learn.pipeline_validation.generalization_lifecycle import scope,guard
    from oeis_learn.pipeline_validation.generalization_acceptance import training_proof
    from foundation_local import parser,Host
    root=api.ROOT;target=confined(root,output,exists=False)
    if target.exists():raise ValueError('fresh smoke output required')
    target.mkdir(parents=True)
    stages=[];record=dict(version='fresh-pilot-smoke-result/v1',status='FAIL',stages=stages,
        scientific_status='UNMEASURED',scope='software smoke only; weights never initialize research')
    began=time.monotonic()
    def rel(path):return str(Path(path).relative_to(root))
    def command(name,argv,destination):
        left=min(LIMITS[name],21600-(time.monotonic()-began))
        if left<=0:raise TimeoutError('smoke reservation exhausted')
        code=api.invoke(argv,target/'private/logs'/name,seconds=left)
        result=api.read(destination/'result.json') if (destination/'result.json').is_file() else None
        status=api.terminal_status(code,result) if result is None or seal(result)==result else 'FAIL'
        stages.append(dict(stage=name,status=status,exit_code=code))
        write(target/'progress.json',seal(record),exclusive=False)
        if status!='PASS':raise RuntimeError('smoke '+name+' did not pass')
        return result,status
    try:
        job=api.read(confined(root,job_path))
        if (seal(job)!=job or set(job)!={'version','engine','qualified','manifest','design','blueprint','identity'} or
            job['version']!='fresh-pilot-smoke-job/v1'):raise ValueError('intact smoke job required')
        cfg,protocol=reduced_blueprint(root,job,target/'private/design')
        record['protocol']=protocol
        assembly=assemble_study(root,cfg,rel(target/'private/inputs'));manifest=assembly['manifest']
        images=cfg['images'];qualified=confined(root,job['qualified']);engine=job['engine']
        output=target/'preflight'
        command('preflight',api.canonical(engine,images[0],'preflight',rel(output),['--manifest',manifest]),output)
        output=target/'prepare'
        result,_=command('prepare',api.canonical(engine,images[0],'prepare',rel(output),
            ['--manifest',manifest,'--qualification',job['qualified']]),output)
        manifest=result['result']['manifest'];base,study,runtime=load(root,manifest)
        record['batch_plan']=verify_smoke_plan(base,study,runtime,protocol)
        print('smoke-batch-plan: '+json.dumps(record['batch_plan'],sort_keys=True),flush=True)
        # One deterministic selection prompt/stratum, chosen before model work.
        targets=[];prompts=[]
        for ref in study['data']['manifests']:
            cohort=json.loads(verify_artifact(base,ref,max_bytes=16<<20))
            if cohort['role']!='selection':continue
            rows=json.loads(verify_artifact(base,cohort['private_payloads'][0]))
            chosen=min(rows,key=lambda row:row['id']);targets.append(chosen)
            visible=json.loads(verify_artifact(base,cohort['public_payloads'][0]))
            prompts.append(next(row for row in visible if row['id']==chosen['id']))
        if len(targets)!=2 or {r['stratum'] for r in targets}!={'parameter','structural'}:raise ValueError('both smoke selection strata required')
        visible=target/'private/evaluation/visible';truth=target/'private/evaluation/scorer'
        write(visible/'prompts.json',prompts);write(truth/'targets.json',targets)
        prompt_ref=reference(visible/'prompts.json',visible);truth_ref=reference(truth/'targets.json',truth)
        protocol_ref=reference(target/'private/design/protocol.json',root)
        recipe=json.loads(verify_artifact(base,runtime['recipe']))
        host=Host(parser().parse_args(api.canonical(engine,images[0],'preflight','unused',['--manifest',manifest])[2:]))
        selected={};proofs={}
        with scope(target,runtime['storage']),guard(target/'private/disk'):
            for arm in ('static','refreshed'):
                output=target/('train-'+arm)
                result,_=command('train-'+arm,api.canonical(engine,images[0],'train',rel(output),
                    ['--manifest',manifest,'--qualification',job['qualified'],'--arms',arm,'--seeds','20261016']),output)
                runs=result['result']['runs']
                if len(runs)!=1:raise ValueError('single smoke learner required')
                training=runs[0]['result'];learner=confined(output/'training',runs[0]['root'])
                if training['updates']<3 or training['tokens']<=protocol['first_shard_tokens']:
                    raise ValueError(f"smoke exposure shortfall arm={arm} updates={training['updates']} tokens={training['tokens']} minimum_updates=3 first_shard_tokens={protocol['first_shard_tokens']} target_tokens={protocol['target_tokens']}")
                checkpoint,weights=api.latest_checkpoint(learner,training,study=study['identity'],arm=arm)
                proof=training_proof(host,root,target/('proof-'+arm),runtime,study,[(learner,training)])
                if proof['status']!='PASS':raise ValueError('finite smoke update proof failed')
                proofs[arm]=proof;selected[arm]=(checkpoint,weights)
            if selected['static'][0]['initialization']!=selected['refreshed'][0]['initialization']:
                raise ValueError('smoke initial weights differ')
            record['training_proofs']=proofs
            for arm,(checkpoint,weights) in selected.items():
                scored,_=api.evaluate_arm(engine,qualified,target,manifest,study,protocol_ref,arm,checkpoint,
                    weights,visible,truth,prompt_ref,truth_ref,recipe,command)
                if not scoring_recorded(scored,targets):raise ValueError('smoke scoring missing or incomplete')
        record['status']='PASS'
    except (Exception,KeyboardInterrupt) as exc:
        record['status']='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAIL'
        record['error']=type(exc).__name__+': '+str(exc)[:1024]
    record['elapsed_seconds']=time.monotonic()-began
    write(target/'result.json',seal(record))
    return api.EXIT_CODES[record['status']]
