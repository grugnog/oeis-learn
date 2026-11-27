"""One finite diagnostic HIP owner, independent reporting and existing ZIP flow."""
from . import progressive_training_timing as timing
from .progressive_workstation_state import write, read, telemetry, Blocked


def measure(w):
    s=w.s;base=w.base;phase=s.phase;folder=base/'timing'
    def inputs():
        value=timing.gather(w.root,w.args.service_session,folder)
        return value,[folder/'training-timing-inputs.json',folder/'private/training-input.json']
    source=s.step('original-inputs',inputs)
    def environment():
        host=w.host()
        value=dict(engine=host.engine.facts(),resources=host.policy,measurements=telemetry(w.root),
                   gpu_requested=True,scope='training-only timing diagnostic')
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
        with Allowance(base/'training-timing-allowance.json',dict(request=req['identity'],scope=s.binding),
                       timing.HOST_SECONDS).attempt() as (left,_):
            body="from oeis_learn.pipeline_validation.progressive_training_timing import run_inside;run_inside('/work/output')"
            stage(w.host(),w.root,image,folder,['python','-c',diagnostic_program(body,'training-timing')],gpu=True,seconds=left)
        value=read(folder/'measurement/producer.json')
        if value['status']!='PASS':raise Blocked('original training timing incomplete')
        return value,[folder/'measurement/producer.json',folder/'hardware.json']
    # A complete step is reused. Failed training is retained, never restarted
    # from random weights under a fresh debit or another session path.
    s.step('training-measurements',execute,dependencies=(phase+'/timing-plan',))
    def review():
        value=timing.report(folder,source)
        if value['status']!='MEASURED':raise Blocked('partial runtime review retained')
        return value,[folder/'training-runtime-review.json']
    s.step('runtime-review',review,recover=True,dependencies=(phase+'/timing-plan',))
