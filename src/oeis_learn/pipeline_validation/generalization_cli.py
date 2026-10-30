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


def main(args,root,host_factory):
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
