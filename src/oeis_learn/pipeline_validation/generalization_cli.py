"""Prospective routing only; not registered with the portable host CLI until Phase C."""
from types import MappingProxyType

from oeis_learn.generalization.interfaces import (
    Command, CommandRequest, CommandResult, ScientificStatus, SoftwareStatus,
)


def _unimplemented(request: CommandRequest) -> CommandResult:
    return CommandResult(
        request.operation, SoftwareStatus.BLOCKED, ScientificStatus.UNMEASURED,
        "generalization_phase_c_not_implemented",
    )


HANDLERS = MappingProxyType({command: _unimplemented for command in Command})


def dispatch(request: CommandRequest) -> CommandResult:
    """Always blocked in Phase A; unknown operations fail before any side effects."""
    return HANDLERS[Command(request.operation)](request)


def register(commands):
    group=commands.add_parser('generalization',help='controlled generalization evidence operations')
    operations=group.add_subparsers(dest='operation',required=True)
    verify=operations.add_parser('verify-bundle',help='verify safe archive and independently reconcile evidence')
    verify.add_argument('--archive',required=True)
    verify.add_argument('--output',required=True)
    report=operations.add_parser('report',help='reconcile and package an explicit public evidence inventory')
    report.add_argument('--manifest',required=True)
    report.add_argument('--inputs',required=True)
    report.add_argument('--output',required=True)
    report.add_argument('--reuse-run')
    for name in ('preflight','qualify','prepare','evaluate','train','replay'):
        command=operations.add_parser(name,help='execute frozen generalization '+name)
        command.add_argument('--manifest',required=True)
        command.add_argument('--runtime',help='sealed runtime sidecar; default beside the study manifest')
        command.add_argument('--output',required=True)
        command.add_argument('--reuse-run')
        if name not in {'preflight','qualify'}:command.add_argument('--qualification',required=True)
        if name=='train':
            command.add_argument('--arms',nargs='+',required=True)
            command.add_argument('--seeds',nargs='+',type=int,required=True)
            command.add_argument('--resume')
        if name=='evaluate':
            command.add_argument('--checkpoints',nargs='+',required=True)
            command.add_argument('--proposal-bank')


def verify_main(args,root,host_factory):
    """Host stays stdlib-only; schema/reconciliation execute in an isolated CPU role."""
    import json,re
    from pathlib import Path
    from .identity import confined,write,seal
    from .generalization_preparation import reference
    from .generalization_roles import project_and_invoke
    from .bundle import pack
    from oeis_learn.container_validation import inspect_current_image
    if args.operation!='verify-bundle':raise ValueError('unregistered generalization endpoint')
    root=Path(root)
    archive=confined(root,args.archive,exists=False)
    output=confined(root,args.output,exists=False)
    if not archive.is_file() or archive.stat().st_size>512<<20:raise ValueError('bounded archive required')
    if output.exists() or output.with_suffix('.zip').exists():raise ValueError('fresh verification output required')
    if not args.image or not re.fullmatch(r'sha256:[0-9a-f]{64}',args.image):
        raise ValueError('--image must identify a prebuilt immutable qualification image')
    host=host_factory(args)
    if host.engine.image(args.image)!=args.image:raise ValueError('immutable qualification image changed')
    output.mkdir(parents=True,exist_ok=False)
    inspect_current_image(host,args.image,output/'inspection',seconds=120)
    ref=reference(archive,root)
    result=project_and_invoke(host,args.image,root,'report',{'public_evidence':[ref]},
        output=str((output/'verification').relative_to(root)),operation='verify-archive',
        settings=dict(archive=ref),seconds=120)
    status=result.get('verification',{}).get('software_status',result['status'])
    write(output/'summary.json',seal(dict(selected_scope=status,archive=ref,
        integrity=result.get('verification',{}).get('integrity','FAIL'),
        scientific_status=result.get('verification',{}).get('scientific_status','UNMEASURED'))))
    pack(output,mandatory=['summary.json'])
    from oeis_learn.generalization.interfaces import EXIT_CODES
    return EXIT_CODES[status]


def report_main(args,root,host_factory):
    import json,re,shutil
    from pathlib import Path
    from .identity import confined,file_hash
    from .generalization_preparation import reference
    from .generalization_roles import project_and_invoke
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.generalization.interfaces import EXIT_CODES
    from oeis_learn.container_validation import inspect_current_image
    root=Path(root);inputs=confined(root,args.inputs)
    manifest=confined(root,args.manifest);evidence_path=confined(inputs,'evidence.json')
    if evidence_path.stat().st_size>1<<20:raise ValueError('evidence manifest size cap')
    record=json.loads(evidence_path.read_text())
    if record['study']['sha256']!=file_hash(manifest) or record['study']['bytes']!=manifest.stat().st_size:
        raise ValueError('report study differs from selected manifest')
    refs=[reference(evidence_path,inputs),*record['inventory']]
    if len(refs)>9997 or sum(r['bytes'] for r in refs)>512<<20:raise ValueError('report inventory cap')
    for ref in refs:verify_artifact(inputs,ref,max_bytes=512<<20)
    output=confined(root,args.output,exists=False)
    if output.exists() or output.with_suffix('.zip').exists():raise ValueError('fresh report output required')
    if not args.image or not re.fullmatch(r'sha256:[0-9a-f]{64}',args.image):
        raise ValueError('--image must identify a prebuilt immutable qualification image')
    host=host_factory(args)
    if host.engine.image(args.image)!=args.image:raise ValueError('immutable qualification image changed')
    output.mkdir(parents=True,exist_ok=False)
    inspect_current_image(host,args.image,output/'inspection',seconds=120)
    result=project_and_invoke(host,args.image,root,'report',{'public_evidence':refs},
        source_roots={'public_evidence':inputs},output=str((output/'packaging').relative_to(root)),
        operation='package',settings={'evidence':refs[0]},seconds=480)
    if result['status'] not in {'PASS','PARTIAL','INCONCLUSIVE','FAIL','BLOCKED','INTERRUPTED'}:raise ValueError('terminal report status required')
    if 'archive' not in result:return EXIT_CODES[result['status']]
    role_root=output/'packaging/role'
    verify_artifact(role_root,result['archive'],max_bytes=512<<20)
    with confined(role_root,result['archive']['path']).open('rb') as source,output.with_suffix('.zip').open('xb') as destination:
        shutil.copyfileobj(source,destination,length=1<<20)
    return EXIT_CODES[result['status']]


def main(args,root,host_factory):
    if args.operation=='report':return report_main(args,root,host_factory)
    if args.operation=='verify-bundle':return verify_main(args,root,host_factory)
    from .generalization_commands import main as execute
    return execute(args,root,host_factory)
