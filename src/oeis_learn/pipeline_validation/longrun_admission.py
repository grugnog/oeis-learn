"""Main-CLI preparation/approval of one exact pilot; never starts research."""
import copy
from dataclasses import replace,asdict
import json
from pathlib import Path
import shutil
from .identity import seal,digest,file_hash,confined,write
from .longrun_contract import PROFILE,requirements,decision,cost_authority,parse_decision


def registry(engine):
    from .registry import registry as original,Stage
    stages=[s for s in original([engine],engine) if s.id in requirements(engine)]
    stages.insert(1,Stage('pilot.compatibility',1,('host.contracts',),seconds=300,adapter='longrun-import'))
    stages=[replace(s,prerequisites=(*s.prerequisites,'pilot.compatibility')) if s.adapter=='engine' else s for s in stages]
    stages.append(Stage('pilot.operations.'+engine,5,('image.'+engine+'.qualification',),seconds=180,adapter='soak-operations'))
    return stages


class Admission:
    def __init__(self,adapters):self.a=adapters;self.imported=None;self.receipts={}

    def stage_binding(self,stage,base):
        result=self.a.stage_binding(stage,base)
        if stage.adapter=='cpu' and self.imported and result.get('hardware')!=self.imported['compatibility']['environment']['hardware']:
            raise ValueError('calibrated CPU/host identity changed')
        return result

    def __call__(self,stage,output):
        self.a.receipts=self.receipts
        if stage.id!='pilot.compatibility':
            result=self.a(stage,output)
            if stage.adapter=='engine':
                from .resources import execution_identity
                proof=self.imported['compatibility'];host=self.a.host(self.a.args.engine)
                if (result!=proof['environment']['engine'] or host.policy['cpus']!=proof['cpus']
                    or execution_identity(host.policy['pipeline_resources'])!=proof['resources']):
                    raise ValueError('calibrated engine/resources/CPU changed before image work')
            return result
        from .longrun_import import inspect
        self.imported=inspect(confined(self.a.root,self.a.args.calibration),confined(self.a.root,self.a.args.inputs),self.a.root,engine=self.a.args.engine)
        write(output/'compatibility.json',self.imported['compatibility'])
        write(output/'calibration-cost.json',self.imported['candidate'])
        return dict(work=self.imported['manifest']['identity'],retained_cost=True,fresh_initialization_required=True)

    def finalize(self,output,invocation,receipts):
        a=self.a;required=sorted(requirements(a.args.engine)|{'pilot.compatibility'})
        passed=all(receipts.get(k,{}).get('state')=='PASS' for k in required)
        result=dict(primary_plumbing='PASS' if passed else 'BLOCKED',research='NOT_EXECUTED',
            long_run_admitted=False,automatic_run_authorized=False,portability='INDEPENDENT_DOCKER_PENDING' if a.args.engine=='podman' else 'SELECTED_ENGINE_ONLY')
        if not passed:
            write(output/'admission.json',result);return result
        from .longrun_import import environment
        from .resources import execution_identity
        imported=self.imported;compat=imported['compatibility'];host=a.host(a.args.engine)
        if environment(invocation,a.args.engine)!=compat['environment']:raise ValueError('calibrated hardware/toolchain/native/environment changed')
        if execution_identity(host.policy['pipeline_resources'])!=compat['resources'] or host.policy['cpus']!=compat['cpus']:
            raise ValueError('calibrated resource envelope/CPU quota changed')
        write(output/'environment.json',compat['environment'])
        ref=lambda p:dict(path=str(Path(p).relative_to(a.root)),sha256=file_hash(p))
        proof=invocation/'stages/pilot.compatibility/compatibility.json'
        cost=cost_authority(imported['candidate'],source=compat['source'],compatibility=compat['identity'])
        write(output/'cost.json',cost)
        required=sorted(requirements(a.args.engine))
        q=seal(dict(version='pipeline-pilot-qualification/v2',source=compat['source'],profile=PROFILE['identity'],work=compat['work'],
            image={k:a.images.images[(a.args.engine,k)] for k in ('gpu','qualification')},engine=a.args.engine,
            resources=compat['resources'],cpus=compat['cpus'],required=required,
            receipts={k:{f:receipts[k][f] for f in ('state','identity')} for k in required},
            receipt_refs={k:ref(invocation/'stages'/k/'receipt.json') for k in required},
            state='PASS',compatibility=compat['identity']))
        write(output/'qualification.json',q)
        d=decision(source=compat['source'],prepared=ref(imported['prepared']/'prepared.json'),
            qualification=ref(output/'qualification.json'),cost=ref(output/'cost.json'),compatibility=ref(proof))
        if a.args.approve:
            from .pilot_contract import decision_payload
            auth=seal(dict(version='pipeline-study-authority/v2',decision_payload=decision_payload(d),approved=True,reviewed_by=a.args.reviewed_by))
            write(output/'authority.json',auth);d=seal(dict(d,state='ADMITTED',authority=ref(output/'authority.json')))
        write(output/'decision.json',d)
        verdict=parse_decision(d,root=a.root,source=compat['source'])
        if a.args.approve and verdict['status']!='ADMITTED':raise ValueError('pilot admission blocked: '+', '.join(verdict['missing']))
        result.update(long_run_admitted=verdict['status']=='ADMITTED',decision=ref(output/'decision.json'),
            projected_seconds=cost['projected_seconds'],status=verdict['status'],
            warm_run_only=True,preparation_cost='Prior preparation/build/reference costs remain separately charged; offline reference total UNKNOWN.')
        write(output/'admission.json',seal(result));return result


def main(args,root,host_factory):
    if args.approve and (not args.reviewed_by or not args.reviewed_by.strip()):raise ValueError('--approve requires explicit nonempty --reviewed-by')
    if not args.approve and args.reviewed_by:raise ValueError('--reviewed-by requires --approve')
    if getattr(args,'cpus',None) not in (None,16):raise ValueError('calibrated pilot requires --cpus 16')
    root=Path(root);output=confined(root,args.output,exists=False)
    if output.exists() or output.with_suffix('.zip').exists():raise ValueError('fresh admission output required')
    engine=args.engine
    if engine=='auto':engine=next((n for n in ('docker','podman') if shutil.which(n)),'docker')
    options=copy.copy(args);options.engine=engine;options.pipeline_engines=[engine]
    # Existing adapters share all installed-package, CPU/HIP, policy and operation
    # qualification. No corpus selection, native readmission, soak or calibration.
    options.reuse=[];options.run=None
    from .pilot_contract import preflight_manifest
    contract=preflight_manifest()
    from .adapters import Adapters
    from .runner import execute
    from .closures import dependency_closure
    from oeis_learn.experiments.source_snapshot import collect_snapshot
    snapshot=collect_snapshot(root);stages=registry(engine)
    maximum=43200
    if args.max_seconds is not None:
        from .identity import number
        if not 0<number(args.max_seconds)<=maximum:raise ValueError('admission ceiling may only be lowered')
        maximum=args.max_seconds
    if args.scope_report:
        print(json.dumps(dict(profile=PROFILE,stages=[asdict(s) for s in stages],maximum_seconds=maximum,
            training=False,calibration=False,approval_requested=args.approve),indent=2));return 0
    return execute(stages,output,adapter=Admission(Adapters(options,root,host_factory,contract)),
        binding=dict(source=digest(snapshot),contract=contract['identity'],dependency_closure=dependency_closure(snapshot,prerequisites={})),
        cap=maximum,reserve=1200,contract=contract)
