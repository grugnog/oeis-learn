"""Serial topological stage dispatch with independent failure continuation."""
import time
from pathlib import Path
from .identity import seal, write, file_hash
from .receipts import Ledger
from .report import summarize, EXIT
from .bundle import pack


def execute(stages, output, *, adapter, binding, cap, reserve, contract=None, unavailable=()):
    output=Path(output)
    if output.exists() or output.with_suffix('.zip').exists():
        raise ValueError('fresh output and archive paths required')
    output.mkdir(parents=True)
    write(output/'contract.json',contract or binding)
    start=time.monotonic()
    write(output/'invocation.json',seal(dict(version='pipeline-invocation/v1',binding=binding,
        started=time.time(),monotonic_origin=start,cap=cap,reserve=reserve,selected=[s.id for s in stages])))
    receipts={};interrupted=False;global_stop=''
    mandatory=['contract.json','invocation.json','summary.json']
    for stage in stages:
        root=output/'stages'/stage.id
        ledger=Ledger(root,stage.id,binding)
        prerequisites={p:receipts[p]['identity'] for p in stage.prerequisites}
        blocked=[p for p in stage.prerequisites if receipts[p]['state'] not in {'PASS','REUSED'}]
        reason=global_stop or ('failed/missing prerequisites: '+','.join(blocked) if blocked else '')
        if not stage.available:
            ledger.transition('UNAVAILABLE',reason=stage.reason)
        elif reason:
            ledger.transition('BLOCKED',reason=reason,prerequisites=prerequisites)
        elif time.monotonic()-start+stage.seconds+reserve>cap:
            ledger.transition('BLOCKED',reason='insufficient remaining global reservation',prerequisites=prerequisites)
        else:
            ledger.transition('READY');ledger.transition('RUNNING',prerequisites=prerequisites)
            begin=time.monotonic()
            print(f'START {stage.id}; log: {root}; remaining {cap-(begin-start):.1f}s',flush=True)
            try:
                result=adapter(stage,root) or {}
                files={str(p.relative_to(root)):file_hash(p) for p in root.rglob('*') if p.is_file() and not p.is_symlink() and p.name not in {'receipt.json','transitions.jsonl'}}
                ledger.transition('PASS',files=files,observations=dict(elapsed_seconds=time.monotonic()-begin,result=result))
            except KeyboardInterrupt:
                interrupted=True;global_stop='interrupted'
                ledger.transition('PARTIAL',reason='user interruption')
            except CapabilityError as exc:
                ledger.transition('PARTIAL',reason=str(exc))
            except BaseException as exc:
                if isinstance(exc,(MemoryError,ResourceBreach)):
                    global_stop='global resource/audit breach: '+str(exc)
                ledger.transition('FAIL',reason=f'{type(exc).__name__}: {exc}')
            print(f"END {stage.id}: {ledger.read()['state']}; elapsed {time.monotonic()-begin:.1f}s",flush=True)
        receipts[stage.id]=ledger.read()
        mandatory.extend([f'stages/{stage.id}/receipt.json',f'stages/{stage.id}/transitions.jsonl'])
    summary=summarize(receipts,interrupted=interrupted,registry_unavailable=unavailable)
    summary.update(elapsed_seconds=time.monotonic()-start,cap_seconds=cap,cleanup_archive_reserve_seconds=reserve)
    write(output/'summary.json',summary)
    code=EXIT[summary['selected_scope']]
    try:
        archive=pack(output,mandatory=mandatory,cap=(contract or {}).get('budgets',{}).get('zip_bytes',512<<20))
        print(f'SUMMARY {output}/summary.json; ARCHIVE {archive}',flush=True)
    except (OSError,ValueError) as exc:
        print(f'ARCHIVE FAILED: {exc}; surviving evidence: {output}/summary.json',flush=True)
        write(output/'export-failure.json',dict(error=str(exc),surviving=str(output/'summary.json')))
        code=code or 1
    return code


class CapabilityError(RuntimeError):
    pass


class ResourceBreach(RuntimeError):
    pass
