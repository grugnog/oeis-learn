"""Full-learner diagnostic through the standard host/Podman failure collector."""
from . import progressive_runtime_timing as timing
from .progressive_workstation_state import write, read, telemetry, Blocked


def measure(w):
    s=w.s;base=w.base;phase=s.phase;folder=base/'timing'
    def inputs():
        value=timing.gather(w.root,w.args.service_session,w.args.training_session,folder)
        return value,[folder/n for n in ('runtime-inputs.json','private/training-input.json',
                       'service-inputs.json','service-budget-review.json','prior-training.json')]
    source=s.step('original-inputs',inputs)
    def environment():
        host=w.host()
        value=dict(engine=host.engine.facts(),resources=host.policy,measurements=telemetry(w.root),
                   gpu_requested=True,scope='full-learner training-only diagnostic')
        write(base/'environment.json',value)
        return value,[base/'environment.json']
    s.step('environment',environment,dependencies=(phase+'/original-inputs',))
    image=w.images(targets=('gpu',))['gpu']
    def request():
        value=timing.plan(w.root,folder,source,image,s.folder)
        return value,[folder/'request.json']
    req=s.step('timing-plan',request,dependencies=(phase+'/original-inputs',phase+'/image-gpu'))
    def execute():
        from .preparation_recovery import Allowance
        from .progressive_handoff import stage
        from .supervision import diagnostic_program
        with Allowance(base/'runtime-allowance.json',dict(request=req['identity'],scope=s.binding),
                       timing.HOST_SECONDS).attempt() as (left,_):
            body=('from oeis_learn.pipeline_validation.progressive_runtime_timing import run_inside;'
                  f"run_inside('/work/output',{max(1,left-120)!r})")
            stage(w.host(),w.root,image,folder,['python','-c',diagnostic_program(body,'runtime-timing')],gpu=True,seconds=left)
        value=read(folder/'runtime-cases.json')
        if value['status']!='PASS':raise RuntimeError('runtime cases incomplete; original results retained')
        paths=[folder/'runtime-cases.json',folder/'hardware.json']
        paths.extend(folder/r['producer']['path'] for r in value['cases'])
        paths.extend(folder/r['diagnostics']['path'] for r in value['cases'] if r.get('diagnostics'))
        return value,paths
    s.step('runtime-measurements',execute,dependencies=(phase+'/timing-plan',))
    def review():
        value=timing.report(folder,source)
        if value['status']!='MEASURED':raise Blocked('partial runtime review retained')
        return value,[folder/'runtime-review.json']
    s.step('runtime-review',review,recover=True,dependencies=(phase+'/timing-plan',))
