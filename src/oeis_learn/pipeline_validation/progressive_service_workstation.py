"""CPU-only continuation of original preparation for service/budget diagnosis."""
from .identity import seal
from .progressive_workstation_state import write, read, telemetry, Blocked
from .progressive_research_inputs import checked_file
from . import progressive_service_calibration as services


def measure(workflow):
    w=workflow;s=w.s;base=w.base;phase=s.phase
    folder=base/'services'
    def inputs():
        result=services.gather(w.root,w.args.preparation_session,folder)
        return result,[folder/'service-inputs.json',folder/'private/training-services.json']
    source=s.step('original-inputs',inputs)
    def environment():
        host=w.host()
        value=dict(engine=host.engine.facts(),resources=host.policy,
            measurements=telemetry(w.root),gpu_requested=False,scope='training-only service diagnostic')
        write(base/'environment.json',value)
        return value,[base/'environment.json']
    s.step('environment',environment,dependencies=(phase+'/original-inputs',))
    image=w.images(targets=('qualification',))['qualification']
    def request():
        value=services.plan(w.root,folder,source,image,s.folder)
        return value,[folder/'request.json']
    req=s.step('service-plan',request,dependencies=(phase+'/original-inputs',phase+'/image-qualification'))
    def execute():
        from .preparation_recovery import Allowance,preserve_process
        from .progressive_handoff import stage
        from .supervision import diagnostic_program
        with Allowance(base/'service-allowance.json',dict(request=req['identity'],scope=s.binding),
                       services.HOST_SECONDS).attempt() as (left,ordinal):
            preserve_process(folder)
            seconds=min(services.SECONDS,left-120)
            if seconds<=0:raise Blocked('no service continuation time remains after cleanup reserve')
            body=('from oeis_learn.pipeline_validation.progressive_service_calibration import run_inside;'
                  "run_inside('/work/output',"+repr(seconds)+')')
            stage(w.host(),w.root,image,folder,['python','-c',diagnostic_program(body,'service-calibration')],
                  gpu=False,seconds=left)
        result=read(folder/'service-cases.json')
        if result['status']!='PASS':
            raise Blocked('Service measurement incomplete; independent cases and original diagnostics retained')
        files=[folder/'service-cases.json',*[checked_file(folder,r['producer']) for r in result['cases']]]
        return result,files
    s.step('service-measurements',execute,recover=True,
        binding=seal(dict(request=req['identity'])) if req else None,
        dependencies=(phase+'/service-plan',))
    # Reporting is independent of a successful case batch and never grants READY.
    def review():
        result=services.report(folder,source)
        if result['status']!='MEASURED':
            raise Blocked('Partial budget review retained; missing service measurements remain explicit')
        return result,[folder/'service-budget-review.json']
    s.step('budget-review',review,recover=True,dependencies=(phase+'/service-plan',))
