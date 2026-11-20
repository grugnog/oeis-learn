"""Registered research preparation through the canonical installed workflow."""
from pathlib import Path
import sys
import time
import traceback

from oeis_learn.generalization.progressive.contracts import CAMPAIGN_DOCUMENT_BYTES,load,reference,seal
from .progressive_research_inputs import checked_file,_json
from .progressive_workstation_state import Blocked,read,write
from oeis_learn.generalization.progressive.research_budget import SETUP_SECONDS
from oeis_learn.generalization.progressive.research_roster import ROSTER_DOCUMENT_BYTES


def prepare(workflow,data,design,materials):
    w=workflow;s=w.s;root=w.root;base=w.base
    setup=design['operational_reservation']['setup_seconds']
    storage_preflight = None
    retry_acceptance = False
    if getattr(getattr(w,'args',None),'reuse_calibration_session',None):
        from .progressive_preparation_preflight import prepare as preflight
        checked=s.step('execution-preflight',lambda:preflight(w),
            dependencies=('prepare-research/research-prerequisites','prepare-research/role-roster'))
        if checked is None:return
        setup=checked['setup_seconds']
        from . import progressive_round019_successor as round019
        from . import progressive_round020_successor as round020
        owner_proof = read(s.folder / 'preparation-owner-proof.json')
        successor = {round019.VERSION: round019, round020.VERSION: round020}.get(owner_proof['version'])
        retry_acceptance = successor is round020
        if successor is not None:
            setup = successor.setup_limits(setup, owner_proof)
            storage_preflight = lambda: round019.check_storage(root, base)
        elif (owner_proof['version'] == 'progressive-preparation-owner-proof/v1'
              and owner_proof['predecessor']['recipient'] == round020.ROUND021):
            setup = round020.replacement_limits(root, setup, owner_proof)
            retry_acceptance = True
            storage_preflight = lambda: round019.check_storage(root, base)
    from .progressive_research_qualification import qualify
    qualify(w,data,design,setup=setup)
    qroot=base/'qualification';timing=base/'evaluation-timing'
    def timing_request():
        from .progressive_research_feasibility import request
        manifest=_json(checked_file(root,data['manifest']))
        retained=manifest['retained']['checkpoints']['D']
        checkpoint=reference(root,checked_file(root/retained['root'],retained['descriptor']['weights']))
        value=request(root,timing,_json(checked_file(root,materials['support']),maximum=256<<20),
            load(base/'matrix/profiles.json')['profiles'],reference(root,qroot/'qualification.json'),checkpoint=checkpoint,
            native=materials['native'])
        files=[timing/'request.json',timing/'private/training-samples.json']
        if 'scoring_services' in value:files.extend([timing/'scoring-services/request.json',timing/'scoring-services/private/training-services.json'])
        return value,files
    req=s.step('evaluation-plan',timing_request,dependencies=('prepare-research/qualify','prepare-research/role-roster'))
    def measure():
        from .progressive_handoff import stage
        from .supervision import diagnostic_program
        elapsed=0
        if 'scoring_image' in req:
            from .progressive_service_calibration import SECONDS as service_seconds
            started=time.monotonic()
            services=timing/'scoring-services'
            service_error=None
            try:
                stage(w.host(),root,req['scoring_image'],services,
                    ['python','-c',diagnostic_program("from oeis_learn.pipeline_validation.progressive_service_calibration import run_inside;run_inside('/work/output',"+str(service_seconds)+")",'research-scoring-services')],
                    gpu=False,seconds=service_seconds+30)
            except (OSError,ValueError,RuntimeError,TimeoutError):
                service_error=traceback.format_exc()
                write(services/'failures.json',dict(traceback=traceback.format_exc()))
                # An ordinary CPU failure keeps independent GPU measurements.
                # Unacknowledged cleanup must not launch another worker owner.
                cleanup=_json(services/'cleanup.json')
                if cleanup.get('acknowledged') is not True or cleanup.get('within_bound') is not True or cleanup.get('failure'):raise
            elapsed=time.monotonic()-started
            write(services/'service-stage.json',seal(dict(status='FAIL' if service_error else 'PASS',
                request=load(services/'request.json')['identity'],source=req['source'],image=req['scoring_image'],
                process=reference(services,services/'process.json'),cleanup=reference(services,services/'cleanup.json'),
                elapsed_seconds=elapsed,error=service_error)))
        if elapsed>=req['seconds']:raise TimeoutError('aggregate evaluation timing deadline')
        stage(w.host(),root,req['image'],timing,
            ['python','-c',diagnostic_program("from oeis_learn.pipeline_validation.progressive_research_feasibility import run_inside;run_inside('/work/output',elapsed_before="+repr(elapsed)+")",'research-evaluation-timing')],
            gpu=True,seconds=req['seconds']-elapsed+30,
            readonly_inputs=[(checked_file(root,req['checkpoint']),'/work/inputs/'+req['checkpoint']['path'])])
        value=load(timing/'evaluation-cases.json')
        files=[timing/'evaluation-cases.json']
        files.extend(checked_file(timing,r['producer']) for r in value['cases'])
        if 'scoring_services' in req:
            services=timing/'scoring-services'
            files.append(services/'service-cases.json')
            if 'scoring_image' in req:files.append(services/'service-stage.json')
            files.extend(checked_file(services,r['producer']) for r in load(services/'service-cases.json')['cases'])
        return value,files
    s.step('evaluation-timing',measure,dependencies=('prepare-research/evaluation-plan',))
    def feasibility():
        from .progressive_research_feasibility import estimates,execution_contract
        roster=load(base/'roster/private/role-roster.json',maximum=ROSTER_DOCUMENT_BYTES)
        value=estimates(timing,req,roster)
        if value['status']!='READY':
            raise Blocked('Measured evaluation cannot fit the prospective role budgets; see evaluation-feasibility.json. No screening looks or members were removed.')
        execution_contract(root,base/'execution-contract.json',request_path=timing/'request.json',
            roster_path=base/'roster/private/role-roster.json',report_path=timing/'evaluation-feasibility.json')
        return value,[timing/'evaluation-feasibility.json',base/'execution-contract.json']
    s.step('evaluation-feasibility',feasibility,dependencies=('prepare-research/evaluation-timing',))
    def construction():
        from .supervision import launch
        argv=[sys.executable,'-S','-c',
            'import sys;sys.path.insert(0,sys.argv.pop(1));from oeis_learn.pipeline_validation.progressive_research_bindings import main;main()',
            str(root/'src'),'--root',str(root),'--output',str((base/'construction').relative_to(root)),
            '--data',str((base/'registered/registered-data.json').relative_to(root)),
            '--design',str((base/'research-round-design.json').relative_to(root)),
            '--materials',str((base/'materials/materials.json').relative_to(root)),
            '--qualification',str(qroot.relative_to(root)),
            '--execution-contract',str((base/'execution-contract.json').relative_to(root))]
        launch(argv,base/'logs/construction',seconds=setup['construction'],stream_cap=4<<20)
        result=load(base/'construction/construction-summary.json')
        # Bind every constructed immutable channel/request. A resumed wrapper
        # cannot treat changed private inputs as completed preparation.
        return result,sorted((base/'construction').rglob('*.json'))
    s.step('construction',construction,dependencies=('prepare-research/evaluation-feasibility',))
    blueprint=str((base/'construction/blueprint.json').relative_to(root));q=str(qroot.relative_to(root))
    preparation_dependencies=('prepare-research/construction','prepare-research/sensitivity')
    if getattr(getattr(w,'args',None),'reuse_calibration_session',None):
        from .progressive_calibration_reuse_workstation import prepare as reuse_calibration
        blueprint=reuse_calibration(w)
        preparation_dependencies+=('prepare-research/calibration-reuse-plan',)
    w.run_command('prepare','prepare',resume=True,blueprint=blueprint,qualification=q,
        output=str((base/'prepared').relative_to(root)),seconds=setup['prepare'],
        dependencies=preparation_dependencies)
    w.run_command('freeze','freeze',blueprint=blueprint,qualification=q,
        prepared=str((base/'prepared').relative_to(root)),output=str((base/'frozen').relative_to(root)),
        seconds=setup['freeze'],preflight=storage_preflight,dependencies=('prepare-research/prepare',))
    if retry_acceptance:
        from .progressive_acceptance_retry import run
        run(w, setup['acceptance'])
    else:
        w.run_command('acceptance','acceptance',campaign=str((base/'frozen/campaign.json').relative_to(root)),
            output=str((base/'acceptance').relative_to(root)),seconds=setup['acceptance'],dependencies=('prepare-research/freeze',))


def readiness(workflow,data,census,sensitivity,materials):
    """Publish complete original authorities, or retain every independent deficit."""
    from .progressive_research_bridge import readiness as data_readiness
    w=workflow;s=w.s;base=w.base
    value=data_readiness(base,data,census,sensitivity,materials)
    needed=('qualify','evaluation-feasibility','construction','prepare','freeze','acceptance')
    missing=[name for name in needed if s.journal['steps'].get('prepare-research/'+name,{}).get('status')!='COMPLETE']
    if not missing and materials and materials.get('status')=='COMPLETE' and sensitivity and sensitivity.get('status')=='COMPLETE':
        from oeis_learn.generalization.progressive.campaign import validate_campaign
        from .progressive_device_acceptance import consume_receipt
        campaign=validate_campaign(load(base/'frozen/campaign.json',maximum=CAMPAIGN_DOCUMENT_BYTES))
        if campaign['purpose']!='research':raise ValueError('research-purpose campaign required')
        from .progressive_acceptance_retry import accepted_folder
        acceptance = accepted_folder(w.root, base, s.journal)
        accepted=load(acceptance/'acceptance.json')
        consume_receipt(acceptance,accepted)
        if accepted['campaign']!=campaign['identity']:raise ValueError('original acceptance campaign differs')
        refs={name:reference(w.root,path) for name,path in [
            ('qualification',base/'qualification/qualification.json'),('campaign',base/'frozen/campaign.json'),
            ('acceptance',acceptance/'acceptance.json'),('resources',base/'prepared/resource-estimate.json'),
            ('evaluation',base/'evaluation-timing/evaluation-feasibility.json')]}
        value=seal(dict(value,status='READY',research_ready=True,deficits=[],authorities=refs,
            stage_dispositions=dict(value['stage_dispositions'],capability='READY',diversity='READY',tuning='READY'),
            qualification='current source and model scope; original campaign-bound acceptance'))
        write(base/'readiness.json',value)
        return value,[base/'readiness.json',*(Path(w.root)/r['path'] for r in refs.values())]
    value=seal(dict(value,deficits=sorted(set(value['deficits']+['incomplete:'+name for name in missing]))))
    write(base/'readiness.json',value)
    raise Blocked('Research preparation retained; '+', '.join(value['deficits'])+'; share prepare-research.zip')
