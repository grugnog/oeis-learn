"""Freeze one adopted reservoir amendment; reuse probes only through a code proof."""
import json
from pathlib import Path

from . import selector_validation_commands as host
from .identity import seal,digest,confined,write
from .generalization_preparation import reference
from oeis_learn.generalization.selector_reservoir import registry,reservoir_design,source_compatibility

AUDIT_SOURCE='sha256:33c1423edc1b05c10a7c3b6ebc114ae6e7038ff3f5bb525b5bc79d0c7080764b'


def parent_evidence(workspace,previous,support):
    """Validate original blocked custody and original successful score producers."""
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.qualification import consume_public,FACTS
    previous=Path(previous);support=Path(support)
    parent=host.read(previous/'plan.json',sealed=True)
    if 'reservoir_amendment' in parent or parent['protocol']!=host.fresh_design():raise ValueError('original unamended structural parent required')
    host.verify_structural_plan(workspace,previous/'plan.json',parent)
    execution=confined(workspace,parent['execution'])
    if host.read(execution/'plan.json',sealed=True)!=parent:raise ValueError('parent execution plan differs')
    if any((execution/a).exists() for a in host.ARMS):raise ValueError('blocked parent must precede all model evaluation')
    role=execution/'publication/role';blocked=host.read(role/'result.json',sealed=True)
    if blocked.get('availability')!=reference(role/'availability.json',role):raise ValueError('original availability producer reference required')
    available=host.read(host.check_ref(role,blocked['availability']),sealed=True)
    policy=registry();required={c['family']:c['count'] for c in parent['protocol']['cells'] if c['family'] in policy['draws']}
    actual_deficits={c['family']:c['count']-available['available'].get(c['family'],0)
        for c in parent['protocol']['cells'] if available['available'].get(c['family'],0)<c['count']}
    if (blocked.get('status')!='BLOCKED' or blocked.get('reason')!='FRESH_COHORT_QUOTA_DEFICIT'
            or blocked['deficits']!=required or available['deficits']!=required or actual_deficits!=required
            or available['protocol']!=parent['protocol']['identity'] or available['draws']!=2288
            or any(available['available'].get(f,0) for f in required)):
        raise ValueError('exact original four-family quota block required')
    census=host.read(support/'result.json',sealed=True)
    plan_ref=reference(execution/'plan.json',workspace)
    if (census.get('version')!='pac-fresh-support-audit/v2' or census.get('status')!='PASS'
            or census.get('scientific_status')!='DIAGNOSTIC_ONLY' or census.get('freeze_eligible') is not False
            or census.get('training_tokens')!=0 or census.get('assessment')!='EXCLUSION_ONLY'
            or census.get('audit_source')!=AUDIT_SOURCE or census['protocol']!=parent['protocol']['identity']
            or census['plan']!=dict(plan_ref,identity=parent['identity']) or set(census['families'])!=set(required)):
        raise ValueError('bound original all-window support census required')
    expected_inputs={x['name']:(x['ref'],x['count']) for x in parent['inputs']['originals']}
    expected_inputs.update({'observed-PAC':(parent['inputs']['observed'],128),'training-probes':(parent['inputs']['probes'],256)})
    if len(census['cells'])!=18 or len({c['name'] for c in census['cells']})!=18:
        raise ValueError('all original support input cells required')
    if {c['name']:(c['input'],c['rows']) for c in census['cells']}!=expected_inputs:raise ValueError('support input references changed')
    for family,quota in required.items():
        value=census['families'][family];domain=2049 if family.startswith('sha256:3f7f') else 2049**2
        if (value['allowed_assignments']!=domain or value['window_free_assignments']<quota
                or value['window_free_assignments']+value['window_blocked_assignments']!=domain
                or value.get('all_assignments_blocked') is not False
                or value.get('category')!=parent['protocol']['families'][family]['category']):
            raise ValueError('positive complete numeric support required')
        remaining=domain
        for cell in census['cells']:
            row=cell['families'][family]
            if (row['newly_blocked_assignments']<0 or row['cumulative_window_free_assignments']<0
                    or remaining-row['newly_blocked_assignments']!=row['cumulative_window_free_assignments']):
                raise ValueError('support census increments disagree')
            remaining=row['cumulative_window_free_assignments']
        if remaining!=value['window_free_assignments']:raise ValueError('support census final count changed')
    probe=host.read(host.check_ref(workspace,parent['probe_plan']),sealed=True)
    result=host.read(host.check_ref(workspace,parent['probe_result']),sealed=True)
    if (result['selector']!=host.POLICY['identity'] or probe['selector']!=host.POLICY
            or result['phase']!='probes' or result['status']!='PASS' or not result['freeze_eligible']
            or probe['inputs']!=parent['inputs']):raise ValueError('unchanged successful original probe required')
    if host.discover(workspace,parent['inputs']['campaign'])!=parent['inputs']:
        raise ValueError('original campaign/checkpoint inputs changed')
    for arm in host.ARMS:
        scored=host.read(host.check_ref(workspace,result['scoring'][arm]),sealed=True)
        if (scored['status']!='PASS' or scored['study']!=probe['identity'] or scored['trace_incomplete']
                or result['generation'][arm]['status']!='PASS'
                or any(scored['metrics']['training']['equal_budget'][s]['complete']<244 for s in ('original','alternative'))):
            raise ValueError('original complete probe scoring/generation required')
    # Verify historical producer bytes against their historical facts; this does
    # not claim those facts describe the new source or current machine.
    qroot=confined(workspace,probe['qualification']);q=host.read(qroot/'qualification.json',sealed=True)
    historical=consume_public(q,{k:q[k] for k in FACTS},root=qroot,
        required_suites=['bigint-cpu','bigint-gpu'],required_capabilities=['weight-load','finite-update','native-exact'])
    old=host.read(host.check_ref(qroot,q['source_closure']))
    if digest(old)!=probe['source'] or q['images']!=probe['images']:raise ValueError('original probe qualification source/images differ')
    proof=source_compatibility(workspace,old,collect_snapshot(workspace))
    return parent,dict(policy=policy['identity'],parent_plan=reference(previous/'plan.json',workspace),
        parent_execution_plan=plan_ref,parent_result=reference(role/'result.json',workspace),
        parent_availability=reference(role/'availability.json',workspace),support=reference(support/'result.json',workspace),
        historical_qualification=reference(qroot/'qualification.json',workspace),historical_producer=historical,compatibility=proof)


def amended_plan(workspace,destination,parent,evidence,images):
    return seal(dict(parent,version='pac-selector-validation-plan/v2',protocol=reservoir_design(),
        source=evidence['compatibility']['current_source'],images=images,
        qualification=str((destination/'qualification').relative_to(workspace)),
        execution=str((destination/'execution').relative_to(workspace)),reservoir_amendment=evidence))


def verify(workspace,path,plan):
    evidence=plan['reservoir_amendment']
    parent_path=host.check_ref(workspace,evidence['parent_plan'])
    support_path=host.check_ref(workspace,evidence['support'])
    parent,current=parent_evidence(workspace,parent_path.parent,support_path.parent)
    expected=amended_plan(workspace,path.parent,parent,current,plan['images'])
    if plan!=expected:raise ValueError('amended structural plan or fixed output differs')
    qroot=confined(workspace,plan['qualification']);q=host.read(qroot/'qualification.json',sealed=True)
    if (q.get('version')!='generalization-qualification/v2' or q.get('status')!='PASS'
            or q['images']!=plan['images'] or digest(host.read(host.check_ref(qroot,q['source_closure'])))!=plan['source']):
        raise ValueError('new source/image qualification differs from frozen amendment')
    oldq=host.read(host.check_ref(workspace,evidence['historical_qualification']),sealed=True)
    from oeis_learn.generalization.qualification import FACTS
    for key in FACTS-{'source_closure','images'}:
        if q[key]!=oldq[key]:raise ValueError('draw-only amendment changed qualified '+key)


def amend(args,workspace,host_factory):
    from .engines import Images
    from .generalization_qualification_io import produce
    workspace=Path(workspace)
    parent,evidence=parent_evidence(workspace,confined(workspace,args.previous),confined(workspace,args.support))
    destination=confined(workspace,args.output,exists=False)
    if destination.exists():raise ValueError('fresh reservoir amendment output required')
    destination.mkdir(parents=True);args._selector_output=destination
    write(destination/'amendment-evidence.json',seal(evidence))
    # Stream original hashes before builds; never open tensor payloads on host.
    inputs=parent['inputs']
    for ref in [inputs['recipe'],inputs['probes'],inputs['observed'],*[x['ref'] for x in inputs['originals']]]:host.check_ref(workspace,ref)
    for selected in inputs['checkpoints'].values():host.check_ref(confined(workspace,selected['root']),selected['descriptor']['weights'])
    machine=host_factory(args)
    if machine.engine.name!=parent['engine']:raise ValueError('original engine required for reservoir amendment')
    images=Images(cap=86400)
    pair=[images.ensure(machine,target,destination/'build'/target,seconds=43200) for target in ('qualification','gpu')]
    plan=amended_plan(workspace,destination,parent,evidence,pair)
    write(destination/'plan.json',plan)
    print('reservoir amendment: frozen 7664 draws; original probe evidence preserved; qualifying current CPU/native and HIP scopes',flush=True)
    qualified=produce(machine,workspace,destination/'qualification',images=pair,contract=plan['identity'],pac_seconds=7200)
    if qualified['status']!='PASS':raise ValueError('New amended-source qualification failed: '+json.dumps(qualified)+'; inspect '+str(destination/'qualification'))
    verify(workspace,destination/'plan.json',plan)
    print(json.dumps(dict(status='FROZEN',plan=str(destination/'plan.json'),identity=plan['identity'],draws=7664,
        probes='original successful evidence with verified unchanged inference closure; no new probe generation')),flush=True)
    return 0


def capture_admission(output):
    """Retain bounded public quota/resource records in failure ZIPs, never truth."""
    output=Path(output);records=[];omissions=[];members=[]
    import re
    for filename in ('availability.json','preparation-resources.json'):
        relative='publication/role/'+filename;candidate=output/relative
        if not candidate.exists() and not candidate.is_symlink():continue
        try:
            path=confined(output,relative)
            if not path.is_file() or path.stat().st_size>256<<10:raise ValueError('bounded admission record required')
            payload=path.read_bytes();value=json.loads(payload)
            if len(payload)>256<<10:raise ValueError('admission byte cap')
            if filename=='availability.json':
                fields={'version','identity','protocol','draws','available','deficits','reference_rejections','exclusion_reasons'}
                if (set(value) not in (fields,fields|{'resources'}) or value!=seal(value)
                        or value['version']!='pac-selector-availability/v1' or type(value['draws']) is not int or not 0<=value['draws']<=8192
                        or not isinstance(value['protocol'],str) or not re.fullmatch(r'sha256:[0-9a-f]{64}',value['protocol'])):
                    raise ValueError('closed public availability required')
                for key in ('available','deficits','reference_rejections','exclusion_reasons'):
                    if not isinstance(value[key],dict) or any(not isinstance(k,str) or len(k)>512 or type(n) is not int or not 0<=n<=8192 for k,n in value[key].items()):
                        raise ValueError('public availability counters required')
                if 'resources' in value:resource_fields=value['resources']
                else:resource_fields=None
            else:resource_fields=value
            if resource_fields is not None:
                names={'peak_rss_bytes','memory_ceiling_bytes','reference_and_exclusion_seconds','index_rows'}
                if set(resource_fields)!=names or any(type(n) not in (int,float) or not 0<=n<=1<<44 for n in resource_fields.values()):
                    raise ValueError('public preparation resource counters required')
            destination=output/'admission-diagnostics'/filename;destination.parent.mkdir(exist_ok=True)
            destination.write_bytes(payload);name=str(destination.relative_to(output));members.append(name)
            records.append(dict(source=relative,copy=name,reference=reference(destination,output)))
        except (OSError,ValueError,TypeError):omissions.append(dict(source=relative,reason='unsafe, changed or oversized public admission record'))
    if records or omissions:
        write(output/'admission-diagnostics/index.json',seal(dict(version='pac-selector-admission-diagnostics/v1',
            qualification_evidence=False,files=records,omissions=omissions)))
        members.append('admission-diagnostics/index.json')
    return members


def admission_attachment(workspace,plan,output,publication):
    from oeis_learn.generalization.selector_reservoir import MEMORY_CEILING
    role=output/'publication/role'
    availability=host.read(host.check_ref(role,publication['availability']),sealed=True,cap=256<<10)
    if (availability['protocol']!=plan['protocol']['identity'] or availability['draws']!=7664 or availability['deficits']
            or availability['resources']['memory_ceiling_bytes']!=MEMORY_CEILING
            or availability['resources']['peak_rss_bytes']>MEMORY_CEILING):
        raise ValueError('complete bounded reservoir admission required')
    write(output/'admission.json',seal(dict(version='pac-selector-reservoir-admission/v1',plan=plan['identity'],
        scope='public preparation counts/resources; no replacement for qualification or model scores',
        producer=reference(role/'result.json',workspace),availability=availability)))
