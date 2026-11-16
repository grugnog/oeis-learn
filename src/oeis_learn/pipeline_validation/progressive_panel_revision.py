"""Prospective panel regrouping from retained native outcomes, without new draws.

Copy into run-state to use the unchanged round016 qualified package. This is a
custodian metadata operation, not qualification, model scoring or a campaign.
Original artifacts stay immutable; deficient panels never gain replacements.
"""
import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import signal
import sqlite3
import sys
import time

SOURCE='sha256:e3592c86ab55c8aad83763cbaae686eaca895bfe58b560e1b64b422231b17ce5'


def regroup(rows, request):
    """Apply the repaired composition relation using the original allocator.

    Temporary aliases encode only the union of original declared relationships
    and corrected composition links. They never reach published rows. The result
    is identical to the corrected allocator, including original family keys.
    """
    from oeis_learn.pipeline_validation.progressive_research_census import Union,reserve_panels
    from oeis_learn.generalization.progressive.prepared_panels import composition
    from oeis_learn.generalization.progressive.contracts import digest
    if len({r['id'] for r in rows})!=len(rows):
        raise ValueError('unique original candidate identities required')
    sharing='reuse' in request
    union=Union(len(rows));seen={}
    relations={r['id']:composition(r) for r in rows if r.get('source')}
    keys={relations[r['id']] for r in rows if r['id'] in relations
          and not (sharing and r['stratum']=='parameter')}
    for i,row in enumerate(rows):
        links=[('declared',row[k]) for k in ('group','family') if row.get(k)]
        relation=relations.get(row['id'])
        if relation in keys:links.append(('composition',relation))
        for key in links:
            if key in seen:union.join(i,seen[key])
            else:seen[key]=i
    components=defaultdict(list)
    for i,row in enumerate(rows):components[union.root(i)].append(row['id'])
    aliases={i:'panel-revision:'+digest(sorted(ids)) for i,ids in components.items()}
    original={r['id']:r for r in rows}
    augmented=[dict(row,group=aliases[union.root(i)]) for i,row in enumerate(rows)]
    roster,counts,deficits=reserve_panels(augmented,seed=request['seed'],required=request['required_per_stratum'],
        minimum_groups=request['minimum_groups'],panels=request['panels'],retention_panels=request['retention_panels'],
        parameter_sharing=sharing)
    for members in roster.values():
        for row in members:
            prior=original[row['id']]
            row['declared_groups']=sorted({prior[k] for k in ('group','family') if prior.get(k)})
    return roster,counts,deficits


def reconstruct(root, materials, progress=lambda *a,**k:None):
    """Replay original selection inputs from read-only receipts and SQLite rows."""
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file,_json,ExposureIndex
    from oeis_learn.pipeline_validation.progressive_research_census import validate_request,reserve_panels
    from oeis_learn.generalization.progressive.research_sampling import source_eligibility
    from oeis_learn.generalization.progressive.contracts import seal
    census_path=checked_file(root,materials['census'])
    census=_json(census_path,maximum=32<<20)
    req=validate_request(_json(checked_file(root,materials['request']),maximum=32<<20))
    if (seal(census)!=census or census['request']!=req['identity']
            or census['diagnostic']!=req['diagnostic'] or census['diagnostic']!=materials['diagnostic']
            or census['native']!=materials['native']):
        raise ValueError('original research census binding differs')
    files={Path(r['path']).name:checked_file(census_path.parent,r) for r in census['artifacts']}
    if set(files)!={'training-reservoir.json','panel-reservations.json','census.sqlite','training-exposure.sqlite'}:
        raise ValueError('complete original census artifacts required')
    expected_training=_json(files['training-reservoir.json'],maximum=128<<20)
    rows=[];training=[];counts=Counter()
    progress('reading-original-native-outcomes')
    with sqlite3.connect(files['census.sqlite'].as_uri()+'?mode=ro',uri=True) as db:
        if (json.loads(db.execute('SELECT value FROM binding').fetchone()[0])!=req
                or db.execute('SELECT value FROM native').fetchone()[0]!=census['native']):
            raise ValueError('original native database scope differs')
        for ordinal,status,raw in db.execute('SELECT ordinal,status,row FROM draws ORDER BY ordinal'):
            counts[status]+=1
            if status not in {'TRAINING','EVALUATION','REJECTED'}:
                raise ValueError('incomplete original native census; no new draws permitted')
            if status=='REJECTED':continue
            row=json.loads(raw)
            if seal(row)!=row:raise ValueError('original native outcome identity differs')
            if (ordinal<req['training_draws'])!=(status=='TRAINING'):
                raise ValueError('original native interval differs')
            (training if status=='TRAINING' else rows).append(row)
    if (sum(counts.values())!=req['training_draws']+req['evaluation_draws'] or training!=expected_training
            or any(counts[k]!=census['counts'].get(k,0) for k in counts)):
        raise ValueError('original native supply/counts differ')
    candidates=_json(checked_file(root,req['candidates']),maximum=128<<20)
    if not isinstance(candidates,list) or len(candidates)>4096 or len({r['id'] for r in candidates})!=len(candidates):
        raise ValueError('bounded original source projection required')
    source_counts=Counter()
    progress('replaying-original-source-exclusions',candidates=len(candidates))
    with ExposureIndex(checked_file(root,req['exposure']),readonly=True) as history, \
            ExposureIndex(files['training-exposure.sqlite'],readonly=True) as train:
        for i,row in enumerate(candidates):
            if 'reuse' in req and source_eligibility(row):source_counts['oeis_ineligible']+=1
            elif set(history.matches(row))|set(train.matches(row)):source_counts['oeis_exposed']+=1
            else:rows.append(dict(row,stratum='oeis'));source_counts['oeis_available']+=1
            if i%256==0:progress('replaying-original-source-exclusions',completed=i,total=len(candidates))
    if any(source_counts[k]!=census['counts'].get(k,0) for k in ('oeis_ineligible','oeis_exposed','oeis_available')):
        raise ValueError('original source supply/counts differ')
    original=_json(files['panel-reservations.json'],maximum=128<<20)
    progress('replaying-original-reservations',candidates=len(rows))
    replay,_,_=reserve_panels(rows,seed=req['seed'],required=req['required_per_stratum'],
        minimum_groups=req['minimum_groups'],panels=req['panels'],retention_panels=req['retention_panels'],
        parameter_sharing='reuse' in req)
    if replay!=original:
        raise ValueError('original reservation replay differs; no inferred candidate replacement')
    return rows,original,census,req,files


def authenticated_materials(session):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.progressive.contracts import digest
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    if session.binding['source']!=SOURCE or digest(collect_snapshot(session.root))!=SOURCE:
        raise ValueError('keep the exact original round016 qualified checkout')
    if any(k.startswith(('run-research/','assess/')) for k in session.journal['steps']):
        raise ValueError('panel revision requires no scientific execution or assessment')
    owner=read(session.folder/'preparation-owner.json');r=owner['receipt']
    active=session.root/'run-state/discovery-preparation-revisions'/('runtime-'+r['census'][7:]+'.json')
    if read(active)!=owner or r['recipient']!=str(session.folder.relative_to(session.root)):
        raise ValueError('original active preparation ownership required')
    base=session.folder/'prepare-research'
    if (base/'frozen/campaign.json').exists() or session.journal['steps'].get('prepare-research/freeze',{}).get('status')!='FAILED':
        raise ValueError('original pre-freeze panel failure required')
    audit_row=session.journal['steps'].get('panel-audit/inspect-v2',{})
    if audit_row.get('status')!='COMPLETE' or len(audit_row.get('outputs',[]))!=1:
        raise ValueError('completed original panel audit required')
    audit=read(checked_file(session.root,audit_row['outputs'][0]))
    if (audit['source']!=SOURCE or audit['status']!='OVERLAPS_FOUND' or audit['errors']
            or audit['exposure_events'] or any(set(p['reasons'])!={'composition'} for p in audit['panel_pairs'])
            or any(set(p['reasons']['composition']['strata'])!={'unseen_structure/unseen_structure'} for p in audit['panel_pairs'])):
        raise ValueError('only the reviewed unseen-composition allocation defect is eligible')
    prepared=session.journal['steps'].get('prepare-research/prepare',{})
    if prepared.get('status')!='COMPLETE' or audit['preparation_result'] not in prepared.get('outputs',[]):
        raise ValueError('audited preparation is no longer the completed original')
    checked_file(session.root,audit['prepared']);checked_file(session.root,audit['preparation_result'])
    failure=read(base/'frozen/result.json')
    if (failure.get('operation')!='freeze' or failure.get('producer_error',{}).get('message')
            !='cross-panel detectable related group leakage'):
        raise ValueError('original detected-panel freeze failure required')
    wanted=str((base/'materials/materials.json').relative_to(session.root))
    refs=[ref for row in session.journal['steps'].values() if row['status']=='COMPLETE'
          for ref in row.get('outputs',[]) if ref['path']==wanted]
    if not refs or any(ref!=refs[0] for ref in refs):raise ValueError('original journal-bound materials required')
    materials=read(checked_file(session.root,refs[0]))
    if materials['status']!='COMPLETE' or materials['diagnostic'] or materials['deficits']:
        raise ValueError('complete original research materials required')
    expected=[materials['exposure'],*[p['cohort'] for p in materials['panels'] if p['role']!='assessment']]
    if sorted(expected,key=lambda r:r['path'])!=sorted(audit['inputs'],key=lambda r:r['path']):
        raise ValueError('audited panels differ from original materials')
    return materials,audit,refs[0]


def revise_panels(session, script):
    from oeis_learn.pipeline_validation.progressive_workstation_state import read,write,Blocked
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file,_json
    from oeis_learn.pipeline_validation.preparation_recovery import retain
    from oeis_learn.generalization.progressive.contracts import digest,reference,seal,validate
    materials,audit,materials_ref=authenticated_materials(session)
    output=session.folder/'panel-revision';began=time.monotonic()
    def progress(phase,**counts):
        value=seal(dict(phase=phase,elapsed_seconds=time.monotonic()-began,**counts))
        write(output/'panel-revision-progress.json',value)
        print('Panel revision: '+phase+' '+json.dumps(counts),flush=True)
    rows,original,census,req,files=reconstruct(session.root,materials,progress)
    progress('regrouping-original-candidates',candidates=len(rows))
    roster,counts,deficits=regroup(rows,req)
    changes={name:dict(before=len(original[name]),after=len(members),
        removed=len({r['id'] for r in original[name]}-{r['id'] for r in members}),
        added=len({r['id'] for r in members}-{r['id'] for r in original[name]})) for name,members in roster.items()}
    report=seal(dict(version='progressive-panel-revision/v1',status='INCOMPLETE' if deficits else 'ALLOCATED',
        source=SOURCE,script=reference(session.root,script),original_materials=materials_ref,audit=audit['identity'],
        changes=changes,counts=counts,deficits=deficits,original_candidates=len(rows),
        new_draws=0,model_calls=0,assessment_scored=False,original_inputs_modified=False,
        qualification_repeated=False,calibration_repeated=False,research_ready=False))
    retain(output/'panel-revision-allocation.json',report)
    if deficits:
        raise Blocked('corrected original allocation has panel deficits; see panel-revision-allocation.json')
    reservations=output/'private/panel-reservations.json';retain(reservations,roster)
    # A common ancestor permits references to original large artifacts without
    # copying them or using symlinks/traversal. Only a new census record is added.
    common=Path(os.path.commonpath([str(p.parent) for p in [*files.values(),reservations]]))
    if not common.is_relative_to(session.root/'run-state'):
        raise ValueError('original artifacts must share confined run-state ancestry')
    name='panel-revision-'+digest([str(session.folder),materials_ref])[7:]+'.json'
    revised_census=common/name
    revised=seal(dict(census,panels=counts,
        deficits=[d for d in census['deficits'] if not d.startswith('panel_supply:')],
        artifacts=[reference(common,reservations if key=='panel-reservations.json' else path) for key,path in files.items()],
        reserved_panel_rows=sum(map(len,roster.values())),
        unreserved_evaluation_rows=len(rows)-sum(map(len,roster.values())),
        panel_revision=dict(original=materials['census'],allocation=reference(session.root,output/'panel-revision-allocation.json'),
                            original_native_work_retained=True,new_native_work=0)))
    retain(revised_census,revised)
    retain(output/'panel-revision-census.json',revised)
    progress('publishing-revised-materials')
    from oeis_learn.pipeline_validation.progressive_research_materials import build
    policy_path=checked_file(session.root,materials['support']).parent.parent/'support-policy.json'
    # A crash may leave a partial publication. Preserve it and use a fresh
    # metadata-only attempt; a completed publication is revalidated and reused.
    destination=output/'materials'
    for attempt in range(8):
        destination=output/('materials' if attempt==0 else 'materials-retry-'+str(attempt))
        if not destination.exists() or (destination/'materials.json').exists():break
    else:raise ValueError('bounded metadata publication attempts exhausted')
    if (destination/'materials.json').exists():result=read(destination/'materials.json')
    else:
        result=build(session.root,str(revised_census.relative_to(session.root)),materials['request']['path'],
            str(destination.relative_to(session.root)),_json(policy_path))
    if result['census']!=reference(session.root,revised_census) or result['request']!=materials['request']:
        raise ValueError('revised publication input binding differs')
    for key in ('support','support_summary'):
        if (result[key]['sha256'],result[key]['bytes'])!=(materials[key]['sha256'],materials[key]['bytes']):
            raise ValueError('original training support changed; calibration must not be reused')
    if result['status']!='COMPLETE':raise Blocked('revised materials deficient; original model work remains retained')
    progress('canonical-panel-validation')
    from oeis_learn.generalization.progressive.cohorts import validate_frozen_panels,census as panel_census
    panels=[validate('cohort',_json(checked_file(session.root,p['cohort']),maximum=32<<20)) for p in result['panels']]
    exposure=validate('exposure',_json(checked_file(session.root,result['exposure'])))
    if {p['payload']['panel'] for p in panels}!=set(roster):raise ValueError('revised panel inventory differs')
    for panel in panels:
        p=panel['payload'];originals={r['id']:r for r in roster[p['panel']]}
        if set(originals)!={r['id'] for r in p['rows']}:raise ValueError('revised membership differs')
        for row in p['rows']:
            prior=originals[row['id']]
            truth=_json(checked_file(session.root,row['truth']))
            if truth!={k:prior[k] for k in ('id','indices','values')} or prior['group'] not in row['groups']:
                raise ValueError('revised panel differs from original native/source outcome')
        if panel_census(panel,exposure=exposure,other_cohorts=panels)['status']!='COMPLETE':
            raise Blocked('canonical revised panel counts deficient')
    # Custodian-only inspection includes reserved assessment separation. Neither
    # its truth nor its members enter a model/scorer or the public report.
    validate_frozen_panels(session.root,panels,{exposure['identity']:exposure})
    value=seal(dict(report,status='PANELS_READY',materials=reference(session.root,destination/'materials.json'),
        revised_census=reference(session.root,revised_census),training_support_unchanged=True,
        elapsed_seconds=time.monotonic()-began,
        remaining=['rebind role requests and measured estimates','canonical preparation replay and freeze','campaign-bound acceptance']))
    if (output/'panel-revision.json').exists():
        prior=read(output/'panel-revision.json')
        if seal(dict(value,elapsed_seconds=prior['elapsed_seconds']))!=prior:
            raise ValueError('completed panel revision differs from replay')
        value=prior
    retain(output/'panel-revision.json',value)
    return value,[output/'panel-revision.json',output/'panel-revision-allocation.json',output/'panel-revision-census.json',revised_census,
                  *sorted(destination.rglob('*.json')),reservations]


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.')
    p.add_argument('--session',default='run-state/discovery/013-research-round-016');args=p.parse_args(argv)
    root=Path(args.root).resolve();sys.path.insert(0,str(root/'src'))
    from oeis_learn.pipeline_validation.identity import confined
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session,read,NAMES
    from oeis_learn.generalization.progressive.contracts import reference
    folder=confined(root,args.session);binding=read(folder/'session.json')['binding']
    session=Session(root,args.session,binding,'panel-revision')
    NAMES.update({'panel-revision.json','panel-revision-allocation.json','panel-revision-progress.json','panel-revision-census.json'})
    def interrupt(*_):raise KeyboardInterrupt()
    previous={s:signal.signal(s,interrupt) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    try:
        with session.own():
            authenticated_materials(session)
            bound=reference(root,Path(__file__).resolve())
            old=session.journal['steps'].get('panel-revision/regroup',{})
            if old and old.get('binding')!=bound:raise ValueError('original panel repair script changed')
            value=session.step('regroup',lambda:revise_panels(session,Path(__file__).resolve()),
                binding=bound,recover=old.get('status') in {'FAILED','INTERRUPTED'})
            return 0 if value is not None else session.code()
    except KeyboardInterrupt:return 130
    finally:
        for s,handler in previous.items():signal.signal(s,handler)


if __name__=='__main__':raise SystemExit(main())
