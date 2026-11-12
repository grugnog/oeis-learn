"""Registered-data preparation, with independent CPU cases and explicit readiness."""
import runpy
import sys

from oeis_learn.generalization.progressive.research_budget import SETUP_SECONDS
from .identity import confined, seal
from .progressive_workstation_state import Blocked, write
from .progressive_research_inputs import checked_file
from oeis_learn.generalization.progressive.contracts import load


def prepare(workflow):
    from . import progressive_research_bridge as bridge
    from .supervision import diagnostic_program
    w = workflow; s = w.s; root = w.root; base = w.base
    reuse=None
    if getattr(w.args,'reuse_preparation_data',None):
        from .progressive_preparation_reuse import adopt
        donor=load(confined(root,w.args.reuse_preparation_data)/'session.json')
        if donor['binding'].get('registration_session')!=w.args.registration_session:
            raise ValueError('data reuse requires the original registration session')
        reuse=adopt(root,w.args.reuse_preparation_data,s.folder)
        write(base/'preparation-data-reuse.json',reuse[2])
    def registered():
        if reuse:
            data=reuse[0]['registered-data']
            write(base/'registered/registered-data.json',data)
            return data,[base/'registered/registered-data.json',base/'preparation-data-reuse.json']
        data,files=bridge.adopt(root,w.args.registration_session,base/'registered')
        donor=getattr(w.args,'reuse_census_session',None)
        if donor:
            from .progressive_research_revision import adopt
            reuse=adopt(root,donor,s.folder,data)
            data=seal(dict(data,census_reuse=reuse))
            write(base/'registered/registered-data.json',data)
            write(base/'census-reuse.json',reuse);files.append(base/'census-reuse.json')
        return data,files
    data = s.step('registered-data', registered)
    s.step('rehearsal-reference',lambda: bridge.rehearsal(root,w.args.rehearsal_session,base,s.binding['source']))
    def plan():
        request=bridge.request(data)
        write(base/'census/request.json',request)
        from oeis_learn.generalization.progressive.prepared_support import policy
        write(base/'support-policy.json',policy(seed=request['seed'],target_tokens=request['token_target'],batch_cadence='reuse' in request))
        from .progressive_research_design import scientific_round
        design=scientific_round(data,expanded=getattr(w.args,'expanded_runtime_budgets',False))
        write(base/'research-round-design.json',design)
        updated=seal(dict(version='progressive-research-preflight/v1',registration=data['identity'],
            source=data['source'], source_resolved=True, executable=False, research_ready=False,
            selected_initial_stages=design['selected_stages'],
            phases=['native_support_and_panel_census','outcome_blind_sensitivity',
                    'prospective_support_and_budget_revision_if_needed','selected_model_HIP_qualification',
                    'exact_input_calibration','finite_campaign_freeze','scoped_acceptance'],
            limits=request, qualification_scope='current source, image, device and selected update policy',
            subsequent_stages='conditional scaling/OEIS/component/source branches require explicit finite reservations'))
        write(base/'research-preflight-plan.json',updated)
        return request,[base/'census/request.json',base/'research-preflight-plan.json',base/'support-policy.json',
                        base/'research-round-design.json']
    req=s.step('census-plan',plan,dependencies=('prepare-research/registered-data',))
    def environment():
        host=w.host()
        from .progressive_workstation_state import telemetry
        value=dict(engine=host.engine.facts(), resources=host.policy,
                   measurements=telemetry(root), gpu_requested=False)
        write(base/'environment.json',value)
        return value,[base/'environment.json']
    s.step('environment',environment,dependencies=('prepare-research/registered-data',))
    image=w.images(targets=('qualification',))['qualification']
    def execute(name, body, *, seconds, inputs=(), outputs=()):
        folder=base/name
        runner=runpy.run_path(str(root/'scripts/qualify_bigint_phase5.py'))
        host=w.host()
        host.policy['memory_bytes']=min(host.policy['memory_bytes'],64 << 30)
        from .preparation_recovery import preserve_process
        preserve_process(folder)
        runner['container_stage'](host,image,folder,
            ['python','-c',diagnostic_program(body,name)],gpu=False,seconds=seconds,
            readonly_inputs=[(checked_file(root,r),'/work/inputs/'+r['path']) for r in inputs])
        return load(folder/outputs[0]),[folder/n for n in outputs]
    def census_action(seconds):
        if reuse:return reuse[0]['native-census'],reuse[1]['native-census']
        if (base/'census/census.json').exists():
            result=load(base/'census/census.json')
            if result['request']!=req['identity']:raise ValueError('original census request differs')
            files=[base/'census/census.json',*[checked_file(base/'census',r) for r in result['artifacts']]]
            return result,files
        result, files=execute('census',
            'from oeis_learn.pipeline_validation.progressive_research_census import main;main()',
            seconds=seconds,inputs=[req[k] for k in ('candidates','exposure','recipe')]+(
                [req['reuse']['census'],req['reuse']['request'],*req['reuse']['artifacts']] if 'reuse' in req else []),
            outputs=('census.json',))
        # Hash-bound private outputs are reusable data, but never included in ZIPs.
        for ref in result['artifacts']:
            files.append(checked_file(base/'census',ref))
        return result,files
    def census():
        from .preparation_recovery import Allowance
        with Allowance(base/'census-host-allowance.json',dict(request=req['identity'],scope=s.binding),req['seconds']+120-req.get('reuse',{}).get('prior_elapsed_seconds',0)).attempt() as (left,ordinal):
            return census_action(left)
    counted=s.step('native-census',census,recover=True,dependencies=('prepare-research/census-plan','prepare-research/image-qualification'))
    def materials():
        if reuse:return reuse[0]['materials'],reuse[1]['materials']
        from .supervision import launch
        # Stdlib byte projection only; no model, native execution, download or
        # original exposure import runs outside the installed container roles.
        argv=[sys.executable,'-S','-c',
            "import sys;sys.path.insert(0,sys.argv.pop(1));from oeis_learn.pipeline_validation.progressive_research_materials import main;main()",
            str(root/'src'),'--root',str(root),
            '--census',str((base/'census/census.json').relative_to(root)),
            '--request',str((base/'census/request.json').relative_to(root)),
            '--policy',str((base/'support-policy.json').relative_to(root)),
            '--output',str((base/'materials').relative_to(root))]
        launch(argv,base/'logs/materials',seconds=SETUP_SECONDS['materials'],stream_cap=4<<20)
        result=load(base/'materials/materials.json')
        return result,sorted((base/'materials').rglob('*.json'))
    projected=s.step('materials',materials,dependencies=('prepare-research/native-census','prepare-research/census-plan'))
    def role_roster():
        if projected['status']!='COMPLETE':
            raise Blocked('Outcome-blind materials are incomplete: '+', '.join(projected['deficits']))
        from .supervision import launch
        argv=[sys.executable,'-S','-c',
            'import sys;sys.path.insert(0,sys.argv.pop(1));from oeis_learn.pipeline_validation.progressive_research_bindings import main;main()',
            str(root/'src'),'--roster-only','--root',str(root),
            '--data',str((base/'registered/registered-data.json').relative_to(root)),
            '--design',str((base/'research-round-design.json').relative_to(root)),
            '--materials',str((base/'materials/materials.json').relative_to(root)),
            '--output',str((base/'roster').relative_to(root))]
        launch(argv,base/'logs/role-roster',seconds=SETUP_SECONDS['roster'],stream_cap=4<<20)
        return load(base/'roster/roster-summary.json'),[base/'roster/roster-summary.json',
            base/'roster/private/role-roster.json',base/'roster/research-reservation.json']
    s.step('role-roster',role_roster,dependencies=('prepare-research/materials',))
    def sensitivity():
        if reuse:return reuse[0]['sensitivity'],reuse[1]['sensitivity']
        # Original registered assumptions, not real panel/model outcomes.
        original=confined(root,data['donor_session']['path']).parent/'register-inputs/design/sensitivity-design.json'
        design=load(original)
        if 'census_reuse' in data:
            from .progressive_research_revision import reuse_sensitivity
            reused=reuse_sensitivity(root,data['census_reuse'],design,base/'sensitivity')
            if reused:return reused
        write(base/'sensitivity/design.json',design)
        return execute('sensitivity',
            "from oeis_learn.pipeline_validation.progressive_research_sensitivity import main;main()",
            seconds=7200,outputs=('sensitivity.json',))
    measured=s.step('sensitivity',sensitivity,dependencies=('prepare-research/registered-data','prepare-research/image-qualification'))
    def prerequisites():
        if projected['status']!='COMPLETE' or measured['status']!='COMPLETE':
            raise Blocked('Original support/panel and sensitivity evidence must be complete before HIP preparation')
        return dict(status='COMPLETE'),[base/'materials/materials.json',base/'sensitivity/sensitivity.json']
    ready=s.step('research-prerequisites',prerequisites,
        dependencies=('prepare-research/role-roster','prepare-research/sensitivity'))
    if ready:
        from .progressive_research_launch import prepare as prepare_execution
        prepare_execution(w,data,load(base/'research-round-design.json'),projected)
    def outcome():
        census_result=counted or seal(dict(version='progressive-research-census/v1',status='UNAVAILABLE',
                                          deficits=['native_census_not_completed']))
        from .progressive_research_launch import readiness
        return readiness(w,data,census_result,measured,projected)
    s.step('readiness',outcome,dependencies=('prepare-research/registered-data',))
