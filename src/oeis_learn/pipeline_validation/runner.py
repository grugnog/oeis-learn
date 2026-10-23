"""Serial topological stage dispatch with independent failure continuation."""
import time
from pathlib import Path
from .identity import seal, write, file_hash
from .receipts import Ledger
from .report import summarize, EXIT
from .bundle import pack,public


def execute(stages, output, *, adapter, binding, cap, reserve, contract=None, unavailable=(), reuse_roots=()):
    output=Path(output)
    if output.exists() or output.with_suffix('.zip').exists():
        raise ValueError('fresh output and archive paths required')
    output.mkdir(parents=True)
    write(output/'contract.json',contract or binding)
    if reuse_roots:
        from .reuse import validate_import
        write(output/'reuse.json',[dict(path=str(root),manifest=validate_import(root)) for root in reuse_roots])
    start=time.monotonic()
    write(output/'invocation.json',seal(dict(version='pipeline-invocation/v1',binding=binding,
        started=time.time(),monotonic_origin=start,cap=cap,reserve=reserve,selected=[s.id for s in stages])))
    receipts={};interrupted=False;global_stop='';phase_costs={}
    mandatory=['contract.json','invocation.json','summary.json']
    for stage in stages:
        root=output/'stages'/stage.id
        ledger=Ledger(root,stage.id,binding)
        prerequisites={p:receipts[p].get('reused_from',receipts[p]['identity']) for p in stage.prerequisites}
        blocked=[p for p in stage.prerequisites if receipts[p]['state'] not in {'PASS','REUSED'}]
        reason=global_stop or ('failed/missing prerequisites: '+','.join(blocked) if blocked else '')
        from .reuse import find_stage
        imported=find_stage(reuse_roots,stage.id,binding,prerequisites) if stage.adapter=='host' and not reason else None
        if imported is not None:
            import shutil
            producer, previous=imported
            files={}
            for name,sha in previous['files'].items():
                from .identity import confined
                target=root/'imported'/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(confined(producer,name),target)
                files[str(target.relative_to(root))]=sha
            ledger.transition('REUSED',reason='complete compatible PASS; original producer retained',files=files,
                reused_from=previous['identity'],prerequisites=prerequisites)
        elif not stage.available:
            ledger.transition('UNAVAILABLE',reason=stage.reason)
        elif reason:
            ledger.transition('BLOCKED',reason=reason,prerequisites=prerequisites)
        elif contract and 'budgets' in contract and str(stage.phase) in contract['budgets'].get('phase_seconds',{}) and stage.adapter != 'image' and phase_costs.get(stage.phase,0)+stage.seconds>contract['budgets']['phase_seconds'][str(stage.phase)]:
            ledger.transition('BLOCKED',reason='owning phase reservation cannot fit',prerequisites=prerequisites)
        elif time.monotonic()-start+stage.seconds+reserve>cap:
            ledger.transition('BLOCKED',reason='insufficient remaining global reservation',prerequisites=prerequisites)
        else:
            ledger.transition('READY');ledger.transition('RUNNING',prerequisites=prerequisites)
            begin=time.monotonic()
            print(f'START {stage.id}; log: {root}; remaining {cap-(begin-start):.1f}s',flush=True)
            try:
                if hasattr(adapter,'receipts'):
                    adapter.receipts=dict(receipts)
                from .disk import DiskWatchdog
                budgets=(contract or {}).get('budgets',{})
                with DiskWatchdog(output,root,cap=budgets.get('transient_bytes',8<<30),
                        trace_cap=budgets.get('trace_bytes',128<<20),
                        free_reserve=budgets.get('mandatory_reserve_bytes',16<<20)):
                    result=adapter(stage,root) or {}
                files=public_files(root)
                ledger.transition(result.get('stage_state','PASS'),files=files,observations=dict(elapsed_seconds=time.monotonic()-begin,result=result))
            except KeyboardInterrupt:
                interrupted=True;global_stop='interrupted'
                ledger.transition('PARTIAL',reason='user interruption',files=public_files(root))
            except CapabilityError as exc:
                ledger.transition('PARTIAL',reason=str(exc),files=public_files(root))
            except BaseException as exc:
                if isinstance(exc,(MemoryError,ResourceBreach)):
                    global_stop='global resource/audit breach: '+str(exc)
                ledger.transition('FAIL',reason=f'{type(exc).__name__}: {exc}',files=public_files(root))
            if stage.adapter != 'image':
                phase_costs[stage.phase]=phase_costs.get(stage.phase,0)+time.monotonic()-begin
            print(f"END {stage.id}: {ledger.read()['state']}; elapsed {time.monotonic()-begin:.1f}s",flush=True)
        receipts[stage.id]=ledger.read()
        mandatory.extend([f'stages/{stage.id}/receipt.json',f'stages/{stage.id}/transitions.jsonl'])
        mandatory.extend(f'stages/{stage.id}/'+name for name in ledger.read()['files'])
    summary=summarize(receipts,interrupted=interrupted,registry_unavailable=unavailable)
    summary.update(phase_consumed_seconds=phase_costs,elapsed_seconds=time.monotonic()-start,cap_seconds=cap,cleanup_archive_reserve_seconds=reserve)
    write(output/'summary.json',summary)
    code=EXIT[summary['selected_scope']]
    try:
        archive=pack(output,mandatory=mandatory,cap=(contract or {}).get('budgets',{}).get('zip_bytes',512<<20))
        print(f'SUMMARY {output}/summary.json; ARCHIVE {archive}',flush=True)
    except (OSError,ValueError) as exc:
        print(f'ARCHIVE FAILED: {exc}; surviving evidence: {output}/summary.json',flush=True)
        try:
            write(output/'export-failure.json',dict(error=str(exc),surviving=str(output/'summary.json')))
        except OSError:
            pass
        code=code or 1
    return code


class CapabilityError(RuntimeError):
    pass


class ResourceBreach(RuntimeError):
    pass


def public_files(root):
    files={}
    for path in root.rglob('*'):
        if (path.is_file() and not path.is_symlink() and public(path.relative_to(root))
                and path.suffix in {'.json','.xml','.jsonl'} and path.name not in {'receipt.json','transitions.jsonl'}):
            try:files[str(path.relative_to(root))]=file_hash(path)
            except OSError:
                # The failed stage remains failed; stderr names any inaccessible
                # record. Export cannot invent missing bytes or terminal outcomes.
                print('UNREADABLE EVIDENCE: '+str(path),flush=True)
    return files
