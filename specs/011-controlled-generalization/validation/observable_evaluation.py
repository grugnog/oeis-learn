#!/usr/bin/env python3
"""Conservative within-family continuation certificates; no training or hardware."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import zipfile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(HERE.parents[2]/'src'))
from post_pilot_analysis import verify_input, branch_visibility, branch_witness, require, check_envelope, bank_envelope
from oeis_learn.synthesis.reference_parser import parse
from oeis_learn.synthesis.admission import source_identity
from oeis_learn.pipeline_validation.identity import digest, seal, file_hash, write
from oeis_learn.pipeline_validation.bundle import pack
from oeis_learn.data.bigint_programs import BigintSampler, canonical_family

POLICY=seal(dict(version='within-family-continuation-policy/v1',visible=list(range(20)),
    continuation=list(range(20,100)),categories=['certified','ambiguous','unresolved'],
    certificate='path polynomial degree d requires d+1 visible indices, or parameter-independent output',
    scope='fixed registered family; mathematical integer semantics; no cross-family identification or runtime guarantee',
    unsupported='loops, parameter-dependent guards, nonpolynomial expressions',
    witness='constructed coefficients2 versus3; independent reference execution; absence is not proof'))


class Unsupported(ValueError):pass


def certificate(source):
    p=parse(source,allow_parameters=True)
    if any(kind!='bigint' for _,kind in p.declarations):raise Unsupported('non-bigint locals')
    paths={}
    for index in range(100):
        env={name:(0,frozenset()) for name,_ in p.declarations};path=[]
        def expr(n):
            if n=='$index':return 1,frozenset()
            if isinstance(n,str):raise Unsupported('unsupported atom')
            op=n[0]
            if op=='int':return 0,frozenset()
            if op=='param.get':return 0,frozenset([n[1]])
            if op=='get':return env[n[1]]
            if op in {'add','sub','mul'}:
                a,pa=expr(n[1]);b,pb=expr(n[2])
                return (a+b if op=='mul' else max(a,b)),pa|pb
            raise Unsupported('nonpolynomial expression:'+op)
        def block(nodes,location=()):
            for position,n in enumerate(nodes):
                loc=(*location,position)
                if n[0]=='set':env[n[1]]=expr(n[2])
                elif n[0]=='if':
                    guard=n[1]
                    if guard[0] not in {'lt','le','eq'} or guard[1]!='$index' or guard[2][0]!='int':
                        raise Unsupported('guard is not a fixed index comparison')
                    threshold=int(guard[2][1]);value=(index<threshold if guard[0]=='lt' else index<=threshold if guard[0]=='le' else index==threshold)
                    path.append((loc,value));block(n[2 if value else 3][1:],(*loc,int(value)))
                else:raise Unsupported('unsupported statement:'+n[0])
        block(p.statements)
        degree,parameters=expr(p.result)
        key=tuple(path)
        r=paths.setdefault(key,dict(path=path,degree_bound=degree,parameters=sorted(parameters),visible_indices=[],continuation_indices=[]))
        require(r['degree_bound']==degree and r['parameters']==sorted(parameters),'inconsistent path abstraction')
        r['visible_indices' if index<20 else 'continuation_indices'].append(index)
    for r in paths.values():
        r['sufficient']=not r['continuation_indices'] or not r['parameters'] or len(r['visible_indices'])>=r['degree_bound']+1
    return dict(proved=all(r['sufficient'] for r in paths.values()),paths=list(paths.values()),
                theorem='A degree<=d polynomial difference with d+1 distinct roots is identically zero.',
                scope=POLICY['scope'])


def classify(source,*,deadline):
    parse(source,allow_parameters=True) # Invalid input is an error, never an unresolved scientific case.
    result=dict(source_sha256=source_identity(source),category='unresolved',certificate=None,witness=None,reason=None)
    if time.monotonic()>=deadline:return dict(result,reason='analysis_deadline')
    try:result['certificate']=certificate(source)
    except Unsupported as exc:result['reason']=str(exc)
    visibility=branch_visibility(source)
    if visibility['unseen_parameters']:
        try:result['witness']=branch_witness(source,visibility['unseen_parameters'],min(deadline,time.monotonic()+2))
        except ValueError:result['reason']='witness_reference_unavailable'
    proved=result['certificate'] is not None and result['certificate']['proved']
    require(not (proved and result['witness']),'contradictory certificate and ambiguity witness')
    if proved:result.update(category='certified',reason=None)
    elif result['witness']:result.update(category='ambiguous',reason=None)
    elif result['reason'] is None:result['reason']='insufficient_path_observations; no witness found'
    return result


def project(rows,scores):
    result={}
    for arm,outcomes in scores.items():
        require(set(outcomes)==set(rows),'projection task roster mismatch')
        groups={category:dict(classification_tasks=0,successes=0,evaluation_complete=0,evaluation_unresolved=0)
                for category in POLICY['categories']}
        for task,meta in rows.items():
            r=outcomes[task]['equal_budget'];g=groups[meta['category']]
            require(type(r['success']) is bool and type(r['complete']) is bool and (not r['success'] or r['complete']),
                    'invalid original outcome')
            g['classification_tasks']+=1;g['successes']+=r['success'];g['evaluation_complete']+=r['complete'];g['evaluation_unresolved']+=not r['complete']
        result[arm]=groups
    return result


def validate_scores(records,report,ref,bank):
    require(report['candidates']['sha256']==ref['sha256'] and report['candidates']['bytes']==ref['bytes'],
            'scoring/candidate file mismatch')
    outcomes=report['result']['outcomes'];recoveries=0
    require(set(records)==set(outcomes),'scoring task roster mismatch')
    for task,row in records.items():
        require(set(row['selections'])==set(outcomes[task]),'scoring control roster mismatch')
        for control,e in row['selections'].items():
            require((e['selected']['source_sha256'] if e else None)==outcomes[task][control]['candidate_sha256'],
                    'scoring/selected source mismatch')
            if e is not None:
                if control=='bank':
                    e,finding=bank_envelope(e,bank);recoveries+=finding is not None
                check_envelope(e)
    return outcomes,recoveries


def freeze_protocol(manifest,plan,task_ids,study,selection,*,deadline):
    require(len(task_ids)==256 and len(set(task_ids))==256,'frozen256-task selection roster required')
    roster={}
    for task in sorted(task_ids):
        meta=plan['roster'][digest(['static','equal_budget',task])]
        other=plan['roster'][digest(['refreshed','equal_budget',task])]
        require(meta['family']==other['family'] and meta['cell'].split('/')[-1]==other['cell'].split('/')[-1], 'paired task/family mismatch')
        roster[task]=dict(family=meta['family'],stratum=meta['cell'].split('/')[-1])
    needed={r['family'] for r in roster.values()};catalog={};sampler=BigintSampler(20261016)
    for ordinal in range(4096):
        if time.monotonic()>=deadline:raise TimeoutError('classification deadline')
        d=sampler.descriptor(ordinal)
        if d['family'] in needed:catalog.setdefault(d['family'],d)
    require(set(catalog)==needed,'unavailable source family')
    families={}
    for key,d in sorted(catalog.items()):
        require(canonical_family(d['parameter_source'])==key,'family source identity mismatch')
        families[key]=dict(stage=d['stage'],parameter_source=d['parameter_source'],
                           **classify(d['parameter_source'],deadline=deadline))
    for meta in roster.values():meta['category']=families[meta['family']]['category']
    require(selection['assessment']=='SEALED','assessment must remain sealed')
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    return seal(dict(version='observable-selection-diagnostic/v1',policy=POLICY,
        original_manifest=manifest['identity'],original_study=study['identity'],implementation=dict(helper=file_hash(__file__),audit_helper=file_hash(HERE/'post_pilot_analysis.py'),
                            production_sources=digest(collect_snapshot(HERE.parents[2]))),
        scope='post-hoc projection of previously inspected selection; not prospective scientific evidence',
        model_input_changes=False,assessment='SEALED',families=families,roster=roster,
        checkpoints={arm:row['checkpoint']['weights'] for arm,row in selection['checkpoints'].items()}))


def run(bundle,output,*,seconds=600):
    output=Path(output)
    if output.exists() or output.with_suffix('.zip').exists():raise FileExistsError('fresh output required')
    output.mkdir(parents=True)
    start=time.monotonic();deadline=start+seconds
    summary=dict(version='observable-diagnostic-summary/v1',selected_scope='FAIL',qualified=False,scientific_status='UNCHANGED',original_mutated=False)
    try:
        manifest=verify_input(bundle)
        with zipfile.ZipFile(bundle) as z:
            read=lambda n:json.loads(z.read('payloads/fresh/'+n))
            # No scoring rows enter the classification function.
            protocol=freeze_protocol(manifest,read('measurement-plan.json'),list(read('generation-static/selections.json')),
                                     read('study.json'),read('selection.json'),deadline=deadline)
            write(output/'protocol.json',protocol)
            scores={};recoveries={}
            for arm in ('static','refreshed'):
                name=f'payloads/fresh/generation-{arm}/selections.json'
                ref=next(r for r in manifest['members'] if r['path']==name)
                scores[arm],recoveries[arm]=validate_scores(json.loads(z.read(name)),read('scoring-'+arm+'.json'),ref,read('generation-'+arm+'/proposal-bank.json'))
            projection=seal(dict(version='observable-score-projection/v1',protocol=protocol['identity'],
                results=project(protocol['roster'],scores),legacy_bank_recoveries=recoveries,scope=protocol['scope']))
            write(output/'projection.json',projection)
        summary.update(selected_scope='PASS',protocol=protocol['identity'],source_bundle=file_hash(bundle),
                       categories=dict(Counter(r['category'] for r in protocol['roster'].values())))
    except Exception as exc:summary['error']=type(exc).__name__+': '+str(exc)
    finally:
        summary['elapsed_seconds']=time.monotonic()-start
        write(output/'summary.json',summary)
        pack(output,mandatory=[p.name for p in sorted(output.iterdir()) if p.is_file()],cap=32<<20)
    if summary['selected_scope']!='PASS':raise RuntimeError(summary['error'])
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=int,choices=range(1,1801),default=600,metavar='1..1800')
    args=p.parse_args();print(json.dumps(run(args.bundle,args.output,seconds=args.seconds),indent=2))
