"""Registered-data preparation, with independent CPU cases and explicit readiness."""
import runpy
import sys

from .identity import confined, seal
from .progressive_workstation_state import Blocked, write
from .progressive_research_inputs import checked_file
from oeis_learn.generalization.progressive.contracts import load


def prepare(workflow):
    from . import progressive_research_bridge as bridge
    from .supervision import diagnostic_program
    w = workflow; s = w.s; root = w.root; base = w.base
    data = s.step('registered-data', lambda: bridge.adopt(root,w.args.registration_session,base/'registered'))
    s.step('rehearsal-reference',lambda: bridge.rehearsal(root,w.args.rehearsal_session,base,s.binding['source']))
    def plan():
        request=bridge.request(data)
        write(base/'census/request.json',request)
        from oeis_learn.generalization.progressive.prepared_support import policy
        write(base/'support-policy.json',policy(seed=request['seed'],target_tokens=request['token_target']))
        design=data['design']
        updated=seal(dict(version='progressive-research-preflight/v1',registration=data['identity'],
            source=data['source'], source_resolved=True, executable=False, research_ready=False,
            selected_initial_stages=design['selected_stages'],
            phases=['native_support_and_panel_census','outcome_blind_sensitivity',
                    'prospective_support_and_budget_revision_if_needed','selected_model_HIP_qualification',
                    'exact_input_calibration','finite_campaign_freeze','scoped_acceptance'],
            limits=request, qualification_scope='current source, image, device and selected update policy',
            subsequent_stages='conditional scaling/OEIS/component/source branches require explicit finite reservations'))
        write(base/'research-preflight-plan.json',updated)
        return request,[base/'census/request.json',base/'research-preflight-plan.json',base/'support-policy.json']
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
        runner['container_stage'](host,image,folder,
            ['python','-c',diagnostic_program(body,name)],gpu=False,seconds=seconds,
            readonly_inputs=[(checked_file(root,r),'/work/inputs/'+r['path']) for r in inputs])
        return load(folder/outputs[0]),[folder/n for n in outputs]
    def census():
        result, files=execute('census',
            'from oeis_learn.pipeline_validation.progressive_research_census import main;main()',
            seconds=req['seconds']+120,inputs=[req[k] for k in ('candidates','exposure','recipe')],
            outputs=('census.json',))
        # Hash-bound private outputs are reusable data, but never included in ZIPs.
        for ref in result['artifacts']:
            files.append(checked_file(base/'census',ref))
        return result,files
    counted=s.step('native-census',census,dependencies=('prepare-research/census-plan','prepare-research/image-qualification'))
    def materials():
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
        launch(argv,base/'logs/materials',seconds=3600,stream_cap=4<<20)
        result=load(base/'materials/materials.json')
        return result,sorted((base/'materials').rglob('*.json'))
    projected=s.step('materials',materials,dependencies=('prepare-research/native-census','prepare-research/census-plan'))
    def sensitivity():
        # Original registered assumptions, not real panel/model outcomes.
        original=confined(root,data['donor_session']['path']).parent/'register-inputs/design/sensitivity-design.json'
        design=load(original)
        write(base/'sensitivity/design.json',design)
        return execute('sensitivity',
            "from oeis_learn.pipeline_validation.progressive_research_sensitivity import main;main()",
            seconds=7200,outputs=('sensitivity.json',))
    measured=s.step('sensitivity',sensitivity,dependencies=('prepare-research/registered-data','prepare-research/image-qualification'))
    def outcome():
        census_result=counted or seal(dict(version='progressive-research-census/v1',status='UNAVAILABLE',
                                          deficits=['native_census_not_completed']))
        value=bridge.readiness(base,data,census_result,measured,projected)
        # Persist the full deficit report even when a producer failed independently.
        raise Blocked('Research preparation evidence collected; '+', '.join(value['deficits'])+
                      '; retain private reservoirs and share prepare-research.zip')
    s.step('readiness',outcome,dependencies=('prepare-research/registered-data',))
