"""Registered research preparation through the canonical installed workflow."""
from pathlib import Path
import sys

from oeis_learn.generalization.progressive.contracts import CAMPAIGN_DOCUMENT_BYTES,load,reference,seal
from .progressive_research_inputs import checked_file,_json
from .progressive_workstation_state import Blocked,write
from oeis_learn.generalization.progressive.research_budget import SETUP_SECONDS
from oeis_learn.generalization.progressive.research_roster import ROSTER_DOCUMENT_BYTES


def prepare(workflow,data,design,materials):
    w=workflow;s=w.s;root=w.root;base=w.base
    from .progressive_research_qualification import qualify
    qualify(w,data,design)
    setup=design['operational_reservation']['setup_seconds']
    qroot=base/'qualification';timing=base/'evaluation-timing'
    def timing_request():
        from .progressive_research_feasibility import request
        manifest=_json(checked_file(root,data['manifest']))
        retained=manifest['retained']['checkpoints']['D']
        checkpoint=reference(root,checked_file(root/retained['root'],retained['descriptor']['weights']))
        value=request(root,timing,_json(checked_file(root,materials['support']),maximum=256<<20),
            load(base/'matrix/profiles.json')['profiles'],reference(root,qroot/'qualification.json'),checkpoint=checkpoint)
        return value,[timing/'request.json',timing/'private/training-samples.json']
    req=s.step('evaluation-plan',timing_request,dependencies=('prepare-research/qualify','prepare-research/role-roster'))
    def measure():
        from .progressive_handoff import stage
        from .supervision import diagnostic_program
        stage(w.host(),root,req['image'],timing,
            ['python','-c',diagnostic_program("from oeis_learn.pipeline_validation.progressive_research_feasibility import run_inside;run_inside('/work/output')",'research-evaluation-timing')],
            gpu=True,seconds=req['seconds']+30,
            readonly_inputs=[(checked_file(root,req['checkpoint']),'/work/inputs/'+req['checkpoint']['path'])])
        value=load(timing/'evaluation-cases.json')
        files=[timing/'evaluation-cases.json']
        files.extend(checked_file(timing,r['producer']) for r in value['cases'])
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
    w.run_command('prepare','prepare',resume=True,blueprint=blueprint,qualification=q,
        output=str((base/'prepared').relative_to(root)),seconds=setup['prepare'],
        dependencies=('prepare-research/construction','prepare-research/sensitivity'))
    w.run_command('freeze','freeze',blueprint=blueprint,qualification=q,
        prepared=str((base/'prepared').relative_to(root)),output=str((base/'frozen').relative_to(root)),
        seconds=setup['freeze'],dependencies=('prepare-research/prepare',))
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
        accepted=load(base/'acceptance/acceptance.json')
        consume_receipt(base/'acceptance',accepted)
        if accepted['campaign']!=campaign['identity']:raise ValueError('original acceptance campaign differs')
        refs={name:reference(w.root,path) for name,path in [
            ('qualification',base/'qualification/qualification.json'),('campaign',base/'frozen/campaign.json'),
            ('acceptance',base/'acceptance/acceptance.json'),('resources',base/'prepared/resource-estimate.json'),
            ('evaluation',base/'evaluation-timing/evaluation-feasibility.json')]}
        value=seal(dict(value,status='READY',research_ready=True,deficits=[],authorities=refs,
            stage_dispositions=dict(value['stage_dispositions'],capability='READY',diversity='READY',tuning='READY'),
            qualification='current source and model scope; original campaign-bound acceptance'))
        write(base/'readiness.json',value)
        return value,[base/'readiness.json',*(Path(w.root)/r['path'] for r in refs.values())]
    value=seal(dict(value,deficits=sorted(set(value['deficits']+['incomplete:'+name for name in missing]))))
    write(base/'readiness.json',value)
    raise Blocked('Research preparation retained; '+', '.join(value['deficits'])+'; share prepare-research.zip')
