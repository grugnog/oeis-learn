"""Rebind validated round016 panels and continue the unchanged qualified runner.

Stdlib host orchestration, copied into run-state. Existing qualification, timing
and calibration measurements are replayed, never regenerated. Canonical prepare,
freeze, campaign acceptance and run/resume retain execution authority.
"""
import argparse
import copy
import importlib.util
from pathlib import Path
import shutil
import signal
import sys
import threading

SESSION='run-state/discovery/013-research-round-016'
REBIND_ERROR='panel rebind changed scientific tasks, settings, tokens, time or learning compatibility'
ORIGINAL_HELPER_SHA='sha256:3812c4c1214bd2c1379336cdc347a69915d75df6dda425083d55679425691b62'
SEED_PHASES={'proposal','training','generation','scoring','retention'}


def checked_step(session, key):
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    row=session.journal['steps'].get(key,{})
    if row.get('status')!='COMPLETE':raise ValueError('completed original step required: '+key)
    for ref in row.get('outputs',[]):checked_file(session.root,ref)
    return row.get('value')


def authenticate(session):
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    receipt=checked_step(session,'prepare-research/reservation-repair')
    path=checked_file(session.root,receipt['script'])
    spec=importlib.util.spec_from_file_location('original_round016_reservation_repair',path)
    repair=importlib.util.module_from_spec(spec);spec.loader.exec_module(repair)
    repair.verify_original(session.root,session)
    row=checked_step(session,'panel-revision/regroup')
    if (row.get('status')!='PANELS_READY' or row.get('deficits') or not row.get('training_support_unchanged')
            or row.get('source')!=session.binding['source'] or row.get('model_calls')!=0 or row.get('new_draws')!=0):
        raise ValueError('original complete outcome-blind panel revision required')
    materials=read(checked_file(session.root,row['materials']))
    old=read(checked_file(session.root,row['original_materials']))
    if materials['status']!='COMPLETE' or materials['diagnostic'] or materials['deficits']:
        raise ValueError('complete revised research materials required')
    for name in ('support','support_summary'):
        checked_file(session.root,materials[name]);checked_file(session.root,old[name])
        if (materials[name]['sha256'],materials[name]['bytes'])!=(old[name]['sha256'],old[name]['bytes']):
            raise ValueError('original training inputs differ')
    return repair,row,materials


def remaining_setup(session, script, repair_receipt):
    """Share only unused original preparation time across the metadata rebind."""
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    from oeis_learn.generalization.progressive.contracts import seal,reference
    base=session.folder/'prepare-research'
    prior=read(base/'reservation-repair/prepare-allowance.json')
    if prior['active'] is not None or prior['seconds']!=repair_receipt['remaining_seconds']:
        raise ValueError('closed original preparation allowance required')
    extra=sum(session.journal['steps'].get(k,{}).get('elapsed_seconds',0) for k in
              ('prepare-research/freeze','panel-audit/inspect','panel-audit/inspect-v2','panel-revision/regroup'))
    remaining=prior['seconds']-prior['spent_seconds']-extra
    if remaining<=0:raise ValueError('original preparation allowance exhausted')
    return seal(dict(version='round016-panel-continuation/v1',source=session.binding['source'],
        script=reference(session.root,script),prior_allowance=reference(session.root,base/'reservation-repair/prepare-allowance.json'),
        additional_original_seconds=extra,remaining_seconds=remaining,
        qualification_repeated=False,calibration_execution_repeated=False,scientific_budgets_changed=False,
        scope='one remaining original preparation allowance shared by metadata, prepare and freeze; acceptance retains its separate original reservation'))


def invariant_tasks(old,new,*,report=None):
    """Keep scientific settings fixed; verify canonical study-derived seeds."""
    from oeis_learn.generalization.progressive.contracts import digest,seal
    changes=[];errors=[]
    def normalized(plan):
        value=copy.deepcopy(plan)
        for key in ('policy','output'):value.pop(key,None)
        meta=value.get('trial')
        if meta:
            if meta['phase'] in SEED_PHASES:
                expected=int(digest([meta['study'],meta['ordinal']])[7:15],16)
                if value['seed']!=expected:
                    errors.append(dict(task=value['id'],field='noncanonical_trial_seed'))
                value['seed']='canonical-study-and-ordinal'
            meta.pop('study',None)
        for choice in value.get('trial_choices',[]):
            choice.pop('policy',None)
            if isinstance(choice.get('runtime'),dict):choice['runtime'].pop('path',None)
        if isinstance(value.get('trial_runtime'),dict):value['trial_runtime'].pop('path',None)
        return value
    if len(old)!=len(new):errors.append(dict(field='task_count'))
    for before,after in zip(old,new):
        left,right=normalized(before),normalized(after)
        for key in sorted(left.keys()|right.keys()):
            if key not in left or key not in right or left[key]!=right[key]:
                errors.append(dict(task=before['id'],field=key))
        if before['seed']!=after['seed']:
            changes.append(dict(task=before['id'],before=before['seed'],after=after['seed'],
                original_study=before.get('trial',{}).get('study'),revised_study=after.get('trial',{}).get('study')))
    value=seal(dict(version='round016-task-rebind/v2',status='PASS' if not errors else 'FAIL',
        tasks=len(new),derived_seed_changes=changes,mismatches=errors,
        policy='Only canonical study-and-ordinal seeds may change; fixed and confirmation seeds, settings and budgets remain exact'))
    if report is not None:
        from oeis_learn.pipeline_validation.preparation_recovery import retain
        retain(report,value)
    if errors:raise ValueError(REBIND_ERROR+'; see task-rebind.json' if report else REBIND_ERROR)
    return value


def bind_calibration(root, old_blueprint, new_blueprint, output, *, bind_reservation):
    from oeis_learn.pipeline_validation.progressive_calibration_reuse import raw_prepared,named,NAME
    from oeis_learn.pipeline_validation.progressive_research_inputs import _json,checked_file
    from oeis_learn.pipeline_validation.preparation_recovery import retain
    from oeis_learn.pipeline_validation.progressive_trajectory import reservation
    from oeis_learn.generalization.progressive.contracts import seal,reference
    old=raw_prepared(root,old_blueprint);new=raw_prepared(root,new_blueprint)
    invariant_tasks(old['payload']['task_plans'],new['payload']['task_plans'],report=output/'task-rebind.json')
    if old['payload']['graph']!=new['payload']['graph'] or old['payload']['caps']!=new['payload']['caps']:
        raise ValueError('original scientific graph/reservations changed')
    payload=copy.deepcopy(new['payload']);payload.pop('blueprint')
    transfer={'research_resource_policy',NAME,'training_trajectory'}
    payload['references']=[r for r in payload['references'] if r['name'] not in transfer]
    for name in sorted(transfer):
        named(old,name)  # Reject missing or duplicate original authorities.
        payload['references'].extend(copy.deepcopy(r) for r in old['payload']['references'] if r['name']==name)
    from oeis_learn.pipeline_validation.progressive_preparation_preflight import reservation as headroom
    budget=named(dict(payload=payload),'research-reservation-v2')
    retain(output/'preparation-headroom.json',headroom(root,budget))
    payload['references'].append(dict(name='preparation_headroom',role='custodian',kind='policy',
        artifact=reference(root,output/'preparation-headroom.json')))
    trajectory=_json(checked_file(root,named(old,'training_trajectory')))
    request=_json(checked_file(root,trajectory['request']))
    retain(output/'trajectory-reservation.json',reservation(root,payload['references'],request,research=True))
    payload['references'].append(dict(name='trajectory_reservation',role='custodian',kind='policy',
        artifact=reference(root,output/'trajectory-reservation.json')))
    retain(output/'preparation-plan.json',dict(version='progressive-preparation-plan/v1',payload=payload))
    result=copy.deepcopy(new_blueprint)
    for ref in result['payload']['references']:
        if ref['name']=='preparation_plan':ref['artifact']=reference(root,output/'preparation-plan.json')
    result=bind_reservation(root,seal(result),_json(checked_file(root,named(old,NAME))))
    retain(output/'blueprint.json',result)
    return result


def archive_failed_freeze(session, output):
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    from oeis_learn.pipeline_validation.preparation_recovery import retain
    from oeis_learn.generalization.progressive.contracts import reference,seal
    path=session.folder/'prepare-research/frozen'
    result=read(path/'result.json')
    row=session.journal['steps'].get('prepare-research/freeze',{})
    failed=row.get('status')=='FAILED' or (row.get('status')=='BLOCKED' and any(h.get('status')=='FAILED' for h in row.get('history',[])))
    if ((path/'campaign.json').exists() or result.get('producer_error',{}).get('message')
            !='cross-panel detectable related group leakage'
            or not failed):
        raise ValueError('only the original failed pre-publication freeze may be archived')
    target=output/'original-failed-freeze'
    if target.exists():raise ValueError('original freeze archive already exists')
    before=[reference(session.root,p) for p in sorted(path.rglob('*')) if p.is_file()]
    target.parent.mkdir(parents=True,exist_ok=True);shutil.move(path,target)
    after=[reference(session.root,target/Path(r['path']).relative_to(path.relative_to(session.root))) for r in before]
    if [(r['bytes'],r['sha256']) for r in before]!=[(r['bytes'],r['sha256']) for r in after]:
        raise ValueError('archived freeze bytes differ')
    value=seal(dict(version='round016-failed-freeze-archive/v1',originals=before,archived=after))
    retain(output/'freeze-archive.json',value)
    return value,[output/'freeze-archive.json',*[session.root/r['path'] for r in after]]


def retry_seed_rebind(session,script,enrolled,folder):
    """Retry the known v1 metadata rejection without resetting its allowance."""
    from oeis_learn.generalization.progressive.contracts import reference
    row=session.journal['steps'].get('prepare-research/panel-campaign-calibration',{})
    if row.get('status')!='FAILED' or row.get('reason')!=REBIND_ERROR:return False
    if (row.get('history') or enrolled['script']['sha256']!=ORIGINAL_HELPER_SHA
            or enrolled['script']==reference(session.root,script)
            or any((folder/n).exists() for n in ('calibration-reuse','prepared','original-failed-freeze'))
            or (session.folder/'prepare-research/frozen/campaign.json').exists()
            or (session.folder/'prepare-research/acceptance/acceptance.json').exists()
            or (session.folder/'run-research').exists()):
        raise ValueError('only the original unexecuted v1 seed rebind rejection may resume')
    return True


def continue_campaign(session,args,script):
    from oeis_learn.generalization.progressive.contracts import load,reference,seal,CAMPAIGN_DOCUMENT_BYTES
    from oeis_learn.generalization.progressive.research_roster import ROSTER_DOCUMENT_BYTES
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow,command
    from oeis_learn.pipeline_validation.progressive_workstation_state import read,Blocked
    from oeis_learn.pipeline_validation.preparation_recovery import retain,revise,Allowance
    from oeis_learn.pipeline_validation import progressive_research_bindings as bindings
    from oeis_learn.pipeline_validation import progressive_research_feasibility as feasibility
    from oeis_learn.pipeline_validation import supervision
    root=session.root;base=session.folder/'prepare-research';folder=base/'panel-campaign'
    repair,panel_report,materials=authenticate(session)
    original=read(base/'reservation-repair/reservation-repair.json')
    def enroll():
        value=remaining_setup(session,script,original)
        retain(folder/'continuation-original-journal.json',seal(session.journal))
        retain(folder/'continuation.json',value)
        return value,[folder/'continuation-original-journal.json',folder/'continuation.json']
    previous=session.journal['steps'].get('prepare-research/panel-campaign-enroll',{})
    if previous.get('status')=='COMPLETE':
        from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
        checked_file(root,previous['value']['script'])
        if previous['binding']!=previous['value']['script']:
            raise ValueError('original continuation enrollment binding differs')
    enrolled=session.step('panel-campaign-enroll',enroll,dependencies=('panel-revision/regroup',),
        binding=previous['binding'] if previous.get('status')=='COMPLETE' else reference(root,script))
    if enrolled is None:return
    def attempt():
        return Allowance(folder/'continuation-allowance.json',enrolled['identity'],enrolled['remaining_seconds']).attempt()
    prefix='prepare-research/'
    retry=retry_seed_rebind(session,script,enrolled,folder)
    def helper():
        value=seal(dict(version='round016-continuation-helper/v2',original_script=enrolled['script'],
            script=reference(root,script),original_allowance=enrolled['identity'],
            seed_policy='canonical study-and-ordinal derivation; fixed seeds unchanged'))
        retain(folder/'continuation-helper.json',value)
        return value,[folder/'continuation-helper.json']
    active=session.step('panel-campaign-helper',helper,dependencies=(prefix+'panel-campaign-enroll',),binding=reference(root,script))
    if active is None:return
    data=load(base/'registered/registered-data.json');design=load(base/'research-round-design.json')
    qroot=base/'qualification';original_timing=base/'evaluation-timing'
    def roster():
        with attempt():
            value=bindings.publish_roster(root,folder/'roster',data,design,materials)
            return value,sorted((folder/'roster').rglob('*.json'))
    session.step('panel-campaign-roster',roster,dependencies=(prefix+'panel-campaign-enroll',))
    def estimates():
        with attempt():
            value=load(folder/'roster/private/role-roster.json',maximum=ROSTER_DOCUMENT_BYTES)
            report=feasibility.estimates(original_timing,load(original_timing/'request.json'),value,publish=False)
            retain(folder/'evaluation-feasibility.json',report)
            if report['status']!='READY':raise Blocked('revised panel evaluation estimates do not fit original budgets')
            feasibility.execution_contract(root,folder/'execution-contract.json',request_path=original_timing/'request.json',
                roster_path=folder/'roster/private/role-roster.json',report_path=folder/'evaluation-feasibility.json')
            return report,[folder/'evaluation-feasibility.json',folder/'execution-contract.json']
    session.step('panel-campaign-estimates',estimates,dependencies=(prefix+'panel-campaign-roster',))
    def construction():
        with attempt() as (left,ordinal):
            argv=[sys.executable,'-S','-c',
                'import sys;sys.path.insert(0,sys.argv.pop(1));from oeis_learn.pipeline_validation.progressive_research_bindings import main;main()',
                str(root/'src'),'--root',str(root),'--output',str((folder/'construction').relative_to(root)),
                '--data',str((base/'registered/registered-data.json').relative_to(root)),
                '--design',str((base/'research-round-design.json').relative_to(root)),
                '--materials',panel_report['materials']['path'],'--qualification',str(qroot.relative_to(root)),
                '--execution-contract',str((folder/'execution-contract.json').relative_to(root))]
            supervision.launch(argv,folder/'logs/construction'/str(ordinal),seconds=left,stream_cap=4<<20)
            value=load(folder/'construction/construction-summary.json')
            return value,sorted((folder/'construction').rglob('*.json'))
    session.step('panel-campaign-construction',construction,dependencies=(prefix+'panel-campaign-estimates',))
    def calibration():
        with attempt():
            value=bind_calibration(root,load(base/'reservation-repair/blueprint.json'),
                load(folder/'construction/blueprint.json'),folder/'calibration-reuse',bind_reservation=repair.bind_reservation)
            return value,sorted((folder/'calibration-reuse').glob('*.json'))
    session.step('panel-campaign-calibration',calibration,dependencies=(prefix+'panel-campaign-construction',prefix+'panel-campaign-helper'),recover=retry)
    def archive():
        with attempt():return archive_failed_freeze(session,folder)
    session.step('panel-campaign-archive',archive,
        dependencies=(prefix+'panel-campaign-calibration',))
    def canonical(name,operation,dependencies,**arguments):
        bound=seal(dict(operation=operation,arguments=arguments))
        old=session.journal['steps'].get(prefix+name,{})
        # Exactly one metadata revision of the old COMPLETE prepare/FAILED freeze.
        # Subsequent unknown or failed executions are retained, not automatically retried.
        replacing=old.get('binding')!=bound and name in {'prepare','freeze'}
        def execute():
            with attempt() as (left,ordinal):
                logs=folder/'logs'/name/str(ordinal)
                supervision.launch(command(root,args,operation,**arguments),logs,seconds=left,
                    stream_cap=4<<20,cleanup_seconds=45,grace_seconds=40,allowed_returncodes=(0,1,2,3))
                process=load(logs/'process.json');code=process['returncode']
                if code:raise (Blocked if code==2 else RuntimeError)(operation+' exited '+str(code)+'; see '+str(logs))
                result=root/arguments['output']/'result.json'
                if load(result).get('mechanical_status')!='COMPLETE':
                    raise ValueError('canonical '+operation+' did not publish a complete result')
                return dict(operation=operation,output=arguments['output'],exit_code=0),[logs/'process.json',result]
        return session.step(name,execute,dependencies=dependencies,binding=bound,recover=replacing)
    relative=lambda path:str(path.relative_to(root))
    canonical('prepare','prepare',(prefix+'panel-campaign-archive',),blueprint=relative(folder/'calibration-reuse/blueprint.json'),
        qualification=relative(qroot),output=relative(folder/'prepared'))
    canonical('freeze','freeze',(prefix+'prepare',),blueprint=relative(folder/'calibration-reuse/blueprint.json'),
        prepared=relative(folder/'prepared'),qualification=relative(qroot),output=relative(base/'frozen'))
    w=Workflow(session,args);w.base=folder
    setup=read(base/'preparation-headroom.json')['setup_seconds']
    w.run_command('acceptance','acceptance',campaign=relative(base/'frozen/campaign.json'),output=relative(base/'acceptance'),
        seconds=setup['acceptance'],dependencies=(prefix+'freeze',))
    def readiness():
        from oeis_learn.generalization.progressive.campaign import validate_campaign
        from oeis_learn.pipeline_validation.progressive_device_acceptance import consume_receipt
        campaign=validate_campaign(load(base/'frozen/campaign.json',maximum=CAMPAIGN_DOCUMENT_BYTES))
        accepted=read(base/'acceptance/acceptance.json');consume_receipt(base/'acceptance',accepted)
        if campaign['purpose']!='research' or accepted['campaign']!=campaign['identity']:
            raise ValueError('revised campaign acceptance differs')
        refs={name:reference(root,path) for name,path in [('campaign',base/'frozen/campaign.json'),
            ('acceptance',base/'acceptance/acceptance.json'),('qualification',qroot/'qualification.json'),
            ('resources',folder/'prepared/resource-estimate.json'),('continuation',folder/'continuation.json')]}
        value=seal(dict(version='round016-panel-campaign-readiness/v1',status='READY',research_ready=True,
            deficits=[],authorities=refs,panels=panel_report['identity']))
        revise(base/'readiness.json',value)
        return value,[base/'readiness.json']
    session.step('readiness',readiness,dependencies=(prefix+'acceptance',))
    w.run_after_prepare()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.');args=p.parse_args(argv)
    root=Path(args.root).resolve();script=Path(__file__).resolve()
    if not script.is_relative_to(root/'run-state'):p.error('copy this host continuation into run-state; keep the qualified checkout unchanged')
    sys.path.insert(0,str(root/'src'))
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session,read,NAMES,TelemetryCollector,telemetry
    binding=read(root/SESSION/'session.json')['binding']
    options=SimpleNamespace(phase='prepare-research',engine=binding['engine'],memory_gib=binding['memory_gib'],
        cpus=binding['cpus'],render_node=binding.get('render_node'))
    session=Session(root,SESSION,binding,'prepare-research');session.observer=TelemetryCollector(session.folder)
    NAMES.update({'continuation.json','continuation-original-journal.json','continuation-allowance.json',
        'freeze-archive.json','preparation-headroom.json','reservation-repair.json','original-journal.json',
        'panel-revision.json','panel-revision-allocation.json','panel-revision-progress.json','panel-revision-census.json',
        'panel-audit.json','panel-audit-progress.json','task-rebind.json','continuation-helper.json'})
    stop=threading.Event()
    def observe():
        while not stop.wait(30):
            try:session.observer.observe(telemetry(root))
            except OSError as exc:session.observer.failed(exc)
            print('Panel campaign continuation: '+str(session.folder),flush=True)
    def interrupt(*_):raise KeyboardInterrupt()
    previous={s:signal.signal(s,interrupt) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    try:
        with session.own():
            thread=threading.Thread(target=observe,daemon=True);thread.start()
            continue_campaign(session,options,script)
            return session.code()
    except KeyboardInterrupt:return 130
    finally:
        stop.set()
        for s,handler in previous.items():signal.signal(s,handler)


if __name__=='__main__':raise SystemExit(main())
