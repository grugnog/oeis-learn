"""Emergency reporting space and a compact, explicitly failed recovery archive."""
import errno
import os
from pathlib import Path
from .identity import seal,write,file_hash,digest


def reserve_space(root, size=16<<20):
    path=Path(root)/'private/emergency.reserve';path.parent.mkdir(parents=True,exist_ok=True)
    try:
        with path.open('xb') as stream:
            try:
                if not hasattr(os,'posix_fallocate'):raise OSError(errno.ENOSYS,'fallocate unavailable')
                os.posix_fallocate(stream.fileno(),0,int(size))
            except OSError as exc:
                if exc.errno not in {errno.ENOSYS,errno.EOPNOTSUPP,errno.EINVAL}:raise
                remaining=int(size);block=b'\0'*(1<<20)
                while remaining:
                    count=min(remaining,len(block));stream.write(block[:count]);remaining-=count
            stream.flush();os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def minimal_archive(output,*,contract,binding,receipts,error,cap):
    from .bundle import pack
    from .receipts import Ledger
    from .report import summarize
    output=Path(output);root=output/'private/failure-bundle';root.mkdir(parents=True)
    write(root/'contract.json',contract or binding)
    write(root/'invocation.json',seal(dict(version='pipeline-recovery-invocation/v1',binding=binding,
        parent={'path':str(output/'invocation.json'),'sha256':file_hash(output/'invocation.json')},
        purpose='failed export recovery; incomplete evidence cannot qualify any stage')))
    write(root/'stage-index.json',seal(dict(version='pipeline-incomplete-stage-index/v1',
        stages={name:dict(state=row['state'],producer=row['identity'],reason=row.get('reason','')[:512],
            file_count=len(row['files']),files_identity=digest(row['files'])) for name,row in receipts.items()},
        omissions='full stage/job details remain in the original output tree and are absent from this failed recovery archive')))
    write(root/'export-failure.json',dict(error=str(error),original_output=str(output),qualification='FAIL'))
    ledger=Ledger(root/'stages/export','export',binding)
    ledger.transition('READY');ledger.transition('RUNNING');ledger.transition('FAIL',reason=str(error))
    summary=summarize({'export':ledger.read()})
    if any(set(r.get('binding',{}))&{'study','batch'} for r in receipts.values()):
        summary['research']='INCOMPLETE_EVIDENCE'
    write(root/'summary.json',summary)
    archive=pack(root,mandatory=['contract.json','invocation.json','stage-index.json','export-failure.json',
        'summary.json','stages/export/receipt.json','stages/export/transitions.jsonl'],cap=min(cap,16<<20),optional_cap=0)
    destination=output.with_suffix('.zip')
    os.link(archive,destination)  # Exclusive; never overwrite a concurrently created file.
    archive.unlink()
    return destination,root/'summary.json'
