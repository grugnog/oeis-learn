#!/usr/bin/env python3
"""Resume round016 under its exact qualified checkout; no build/Q/timing replay.

Copy this script into run-state before checking out QUALIFIED_COMMIT. This is
run-specific stdlib orchestration over the unchanged canonical installed roles.
The reusable reservation fix lives in progressive_calibration_reuse.py.
"""
import argparse
import copy
from pathlib import Path
import signal
import subprocess
import sys
import threading

QUALIFIED_COMMIT = '2faa8e43fb772b7ad5d4232fe99ff93b3d96b81d'
SESSION = 'run-state/discovery/013-research-round-016'
ERROR = 'full original calibration debit must remain reserved'


def bind_reservation(root, blueprint, proof):
    # Same metadata projection as the package fix; kept here so the qualified
    # source itself stays byte-identical. Canonical prepare replays all authority.
    from oeis_learn.pipeline_validation.progressive_calibration_reuse import named, raw_prepared
    from oeis_learn.pipeline_validation.progressive_research_inputs import _json, checked_file
    from oeis_learn.pipeline_validation.progressive_handoff import named_plan
    from oeis_learn.generalization.progressive.contracts import CAMPAIGN_DOCUMENT_BYTES, validate, seal
    old = validate('blueprint', _json(checked_file(root, proof['blueprint']), maximum=CAMPAIGN_DOCUMENT_BYTES))
    request = _json(checked_file(root, proof['request']))
    fields = {'seconds','caps','checkpoint_bytes','maximum_inodes'}
    prior = named_plan(root, old, 'iteration_calibration_plan', 'progressive-calibration-plan/v1', fields, optional={'checkpoint_retention'})
    current = named_plan(root, blueprint, 'iteration_calibration_plan', 'progressive-calibration-plan/v1', fields, optional={'checkpoint_retention'})
    if (prior is None or current is None or prior['caps'] != request['caps']
            or prior['seconds'] * 1000000 != prior['caps']['elapsed_us'] or prior['seconds'] != current['seconds']):
        raise ValueError('original calibration debit and time reservation required')
    raw = raw_prepared(root, blueprint)
    result = copy.deepcopy(blueprint)
    for ref in result['payload']['references']:
        if ref['name']=='iteration_calibration_plan': ref['artifact']=named(old,'iteration_calibration_plan')
    result['payload']['proposed_caps'] = {k:max(result['payload']['proposed_caps'][k],raw['payload']['caps'][k]+v) for k,v in prior['caps'].items()}
    return validate('blueprint',seal(result))


def verify_original(root, session):
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.generalization.progressive.contracts import digest
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file,_json
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    if head != QUALIFIED_COMMIT or digest(collect_snapshot(root)) != session.binding['source']:
        raise ValueError('exact round016 qualified checkout required; do not pull or edit source during this campaign')
    owner=read(session.folder/'preparation-owner.json')
    receipt=owner['receipt']
    current=root/'run-state/discovery-preparation-revisions'/('runtime-'+receipt['census'][7:]+'.json')
    if read(current)!=owner or receipt['recipient']!=str(session.folder.relative_to(root)):
        raise ValueError('round016 is no longer the active preparation owner')
    for key,row in session.journal['steps'].items():
        if row['status']=='COMPLETE':
            for ref in row.get('outputs',[]): checked_file(root,ref)
        if row['status'] in {'RUNNING','INTERRUPTED'} and not key.startswith('run-research/'):
            raise ValueError('reconcile interrupted operation before continuation: '+key)
    q=read(session.folder/'prepare-research/qualification/qualification.json')
    if q['images'] != [_json(session.folder/'prepare-research'/('image-'+n)/'image.json')['image'] for n in ('qualification','gpu')]:
        raise ValueError('original qualified image bindings differ')


def repair_plan(root, session, script):
    from oeis_learn.generalization.progressive.contracts import reference, seal, CAMPAIGN_DOCUMENT_BYTES
    from oeis_learn.pipeline_validation.progressive_research_inputs import _json, checked_file
    from oeis_learn.pipeline_validation.progressive_workstation_state import read
    from oeis_learn.pipeline_validation.preparation_recovery import retain
    from oeis_learn.pipeline_validation.progressive_process_result import stdout_result
    base=session.folder/'prepare-research'; output=base/'reservation-repair'
    row=session.journal['steps']['prepare-research/prepare']
    if row['status']!='FAILED' or row.get('history') or any((base/n).exists() for n in ('frozen','acceptance','construction/execution','prepared/prepared.json')):
        raise ValueError('one original pre-freeze failed prepare required')
    result=read(base/'prepared/result.json');error=result.get('producer_error') or {}
    if (result.get('mechanical_status')!='INVALID' or result.get('reason')!='CONTRACT_MISMATCH'
            or error.get('type')!='ValueError' or error.get('message')!=ERROR):
        raise ValueError('only round016 original reservation mismatch may be repaired')
    trace=checked_file(base/'prepared',error['local_traceback'])
    if error['local_traceback']['bytes']>64<<10 or not trace.read_text().rstrip().endswith('ValueError: '+ERROR):
        raise ValueError('original bounded reservation traceback required')
    process=_json(base/'logs/prepare/0000/process.json')
    if (process.get('returncode')!=3 or process.get('outcome')!='COMPLETE' or process.get('failure')
            or process.get('cleanup_failure') or process.get('omitted_bytes',{}).get('stdout',0)
            or stdout_result(base/'logs/prepare/0000/stdout.log')!=result):
        raise ValueError('original failed prepare must be fully terminated')
    allowance=read(base/'prepare-allowance.json')
    expected=seal(dict(scope=session.binding,operation='prepare',arguments=dict(
        blueprint=str((base/'calibration-reuse/blueprint.json').relative_to(root)),
        qualification=str((base/'qualification').relative_to(root)),output=str((base/'prepared').relative_to(root)))))
    if (allowance.get('active') is not None or allowance['binding']!=expected
            or not 0<allowance['spent_seconds']<allowance['seconds']):
        raise ValueError('closed original preparation allowance with remaining time required')
    blueprint=_json(base/'calibration-reuse/blueprint.json',maximum=CAMPAIGN_DOCUMENT_BYTES)
    proof=read(base/'calibration-reuse/calibration-reuse.json')
    revised=bind_reservation(root,blueprint,proof)
    if revised==blueprint: raise ValueError('original mismatch is not reproducible')
    retain(output/'original-journal.json',copy.deepcopy(session.journal))
    retain(output/'blueprint.json',revised)
    value=seal(dict(version='round016-reservation-repair/v1',source=session.binding['source'],
        qualified_commit=QUALIFIED_COMMIT,script=reference(root,script),
        original_blueprint=reference(root,base/'calibration-reuse/blueprint.json'),
        revised_blueprint=reference(root,output/'blueprint.json'),original_result=reference(root,base/'prepared/result.json'),
        original_journal=reference(root,output/'original-journal.json'),
        original_allowance=reference(root,base/'prepare-allowance.json'),
        remaining_seconds=allowance['seconds']-allowance['spent_seconds'],
        qualification=reference(root,base/'qualification/qualification.json'),
        no_new_qualification=True,no_calibration_execution=True,original_costs_retained=True))
    retain(output/'reservation-repair.json',value)
    return value,[output/n for n in ('original-journal.json','blueprint.json','reservation-repair.json')]


def continue_preparation(root, session, args, script):
    from oeis_learn.pipeline_validation.progressive_workstation import Workflow
    from oeis_learn.pipeline_validation.progressive_workstation_state import read,write
    from oeis_learn.generalization.progressive.contracts import reference,seal,load,CAMPAIGN_DOCUMENT_BYTES
    verify_original(root,session)
    prefix='prepare-research/'
    dependencies=tuple(prefix+n for n in ('qualify','evaluation-feasibility','construction','training-trajectory','calibration-reuse-plan','sensitivity'))
    value=session.step('reservation-repair',lambda:repair_plan(root,session,script),dependencies=dependencies)
    if value is None:return
    from oeis_learn.pipeline_validation.progressive_research_inputs import checked_file
    if checked_file(root,value['script'])!=script:raise ValueError('original repair script changed')
    base=session.folder/'prepare-research';repair=base/'reservation-repair'
    w=Workflow(session,args);w.base=repair
    relative=lambda p:str(p.relative_to(root))
    w.run_command('prepare','prepare',resume=True,blueprint=relative(repair/'blueprint.json'),
        qualification=relative(base/'qualification'),output=relative(repair/'prepared'),
        seconds=value['remaining_seconds'],dependencies=(prefix+'reservation-repair',))
    setup=read(base/'preparation-headroom.json')['setup_seconds']
    w.run_command('freeze','freeze',blueprint=relative(repair/'blueprint.json'),prepared=relative(repair/'prepared'),
        qualification=relative(base/'qualification'),output=relative(base/'frozen'),
        seconds=setup['freeze'],dependencies=(prefix+'prepare',))
    w.run_command('acceptance','acceptance',campaign=relative(base/'frozen/campaign.json'),
        output=relative(base/'acceptance'),seconds=setup['acceptance'],dependencies=(prefix+'freeze',))
    def readiness():
        from oeis_learn.generalization.progressive.campaign import validate_campaign
        from oeis_learn.pipeline_validation.progressive_device_acceptance import consume_receipt
        campaign=validate_campaign(load(base/'frozen/campaign.json',maximum=CAMPAIGN_DOCUMENT_BYTES))
        accepted=read(base/'acceptance/acceptance.json');consume_receipt(base/'acceptance',accepted)
        if campaign['purpose']!='research' or accepted['campaign']!=campaign['identity']:raise ValueError('original campaign acceptance differs')
        refs={n:reference(root,p) for n,p in [('campaign',base/'frozen/campaign.json'),('acceptance',base/'acceptance/acceptance.json'),
            ('qualification',base/'qualification/qualification.json'),('resources',repair/'prepared/resource-estimate.json'),
            ('repair',repair/'reservation-repair.json')]}
        result=seal(dict(version='round016-repair-readiness/v1',status='READY',research_ready=True,deficits=[],authorities=refs))
        from oeis_learn.pipeline_validation.preparation_recovery import revise
        revise(base/'readiness.json',result)
        return result,[base/'readiness.json']
    session.step('readiness',readiness,dependencies=(prefix+'acceptance',))
    w.run_after_prepare()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.');args=p.parse_args(argv)
    root=Path(args.root).resolve();script=Path(__file__).resolve()
    if not script.is_relative_to(root/'run-state'):p.error('copy this script into run-state before selecting the qualified checkout')
    sys.path.insert(0,str(root/'src'))
    from oeis_learn.pipeline_validation.progressive_workstation_state import Session,read,TelemetryCollector,telemetry,NAMES
    binding=read(root/SESSION/'session.json')['binding']
    from types import SimpleNamespace
    options=SimpleNamespace(phase='prepare-research',engine=binding['engine'],memory_gib=binding['memory_gib'],
        cpus=binding['cpus'],render_node=binding.get('render_node'))
    s=Session(root,SESSION,binding,'prepare-research');s.observer=TelemetryCollector(s.folder)
    NAMES.update({'original-journal.json','reservation-repair.json'})
    stop=threading.Event()
    def observe():
        while not stop.wait(30):
            try:s.observer.observe(telemetry(root))
            except OSError as exc:s.observer.failed(exc)
            print('Round016 repair progress: '+str(s.folder),flush=True)
    def interrupt(*_):raise KeyboardInterrupt()
    previous={sig:signal.signal(sig,interrupt) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    try:
        with s.own():
            thread=threading.Thread(target=observe,daemon=True);thread.start()
            continue_preparation(root,s,options,script)
            return s.code()
    except KeyboardInterrupt:return 130
    finally:
        stop.set()
        for sig,handler in previous.items():signal.signal(sig,handler)


if __name__=='__main__':raise SystemExit(main())
