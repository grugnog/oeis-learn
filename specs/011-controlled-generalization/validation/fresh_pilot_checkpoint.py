#!/usr/bin/env python3
"""Read-only checkpoint diagnostics; stdlib only, never loads tensor/optimizer state."""
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from oeis_learn.pipeline_validation.identity import confined,seal
from oeis_learn.generalization.contracts import artifact_refs
from oeis_learn.generalization.budgets import Ledger


def scheduler_seconds(payload):
    """Decode only the scalar scheduler dict; reject globals and persistent IDs."""
    class Scalars(pickle.Unpickler):
        def find_class(self,*args):raise ValueError('scheduler must contain scalars only')
        def persistent_load(self,*args):raise ValueError('scheduler persistent IDs forbidden')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        names=[n for n in archive.namelist() if n.endswith('/data.pkl')]
        if len(names)!=1 or archive.getinfo(names[0]).file_size>1<<20:raise ValueError('bounded scheduler pickle required')
        value=Scalars(io.BytesIO(archive.read(names[0]))).load()
    if not isinstance(value,dict) or set(value)!={'learner_seconds'}:raise ValueError('scalar learner clock required')
    seconds=value['learner_seconds']
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or not 0<=seconds<=1e9:
        raise ValueError('invalid persisted learner seconds')
    return seconds


def document(path):
    if path.is_symlink() or not path.is_file() or path.stat().st_size>16<<20:raise ValueError('bounded checkpoint metadata required')
    return json.loads(path.read_bytes())


def inspect(command):
    """Verify original receipt, latest/pins and referenced bytes; export aggregates."""
    command=Path(command);receipt=document(confined(command,'result.json'))
    if seal(receipt)!=receipt or receipt.get('version')!='generalization-command-result/v2' or receipt.get('stage')!='train':
        raise ValueError('original canonical training receipt required')
    inventory=receipt['inventory']
    if not isinstance(inventory,list) or not 0<len(inventory)<=10000:raise ValueError('bounded original inventory required')
    by_path={r['path']:r for r in inventory}
    if len(by_path)!=len(inventory):raise ValueError('duplicate inventory path')
    pointers=[p for p in by_path if p.endswith('/checkpoints/generalization-latest.json')]
    if len(pointers)!=1:raise ValueError('exactly one published learner pointer required')
    checked={};total=0
    def check(path,expected=None):
        nonlocal total
        relative=str(path.relative_to(command));ref=by_path.get(relative)
        if ref is None or set(ref)!={'path','bytes','sha256'}:raise ValueError('checkpoint missing from original inventory')
        if expected is not None and any(ref[k]!=expected[k] for k in ('bytes','sha256')):raise ValueError('checkpoint reference differs from inventory')
        if relative in checked:return path
        path=confined(command,relative)
        size=ref['bytes']
        if type(size) is not int or size<0 or not path.is_file() or path.stat().st_size!=size:raise ValueError('checkpoint size changed')
        total+=size
        if total>32<<30:raise ValueError('checkpoint verification byte cap')
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
        if 'sha256:'+h.hexdigest()!=ref['sha256']:raise ValueError('checkpoint bytes changed')
        checked[relative]=ref
        return path
    pointer_path=check(command/pointers[0]);base=pointer_path.parent
    pointer=document(pointer_path)
    pins=document(check(base/'generalization-pins.json'))
    if not isinstance(pins,dict) or len(pins)>64:raise ValueError('bounded checkpoint pin set required')
    refs={pointer['path']:pointer}
    for ref in pins.values():
        if ref['path'] in refs and refs[ref['path']]!=ref:raise ValueError('conflicting checkpoint pin')
        refs[ref['path']]=ref
    latest=None;generations=set()
    from oeis_learn.generalization.state import _complete
    for ref in refs.values():
        record=document(check(confined(base,ref['path']),ref))
        if seal(record)!=record or record.get('version')!='generalization-learner-state/v1' or record.get('kind')!='full_state':
            raise ValueError('complete sealed learner state required')
        if set(record)!={'version','kind','contract','states','ledger','boundary','identity'} or record['boundary']!='before_or_after_complete_update':raise ValueError('invalid checkpoint boundary')
        Ledger.restore(record['ledger'],cap=record['ledger']['cap'])
        _complete(record['states'])
        for child in artifact_refs(record['states']):check(confined(base,child['path']),child)
        if ref==pointer:latest=record
        generations.add(Path(ref['path']).parts[0])
    for ref in refs.values():
        record=document(confined(base,ref['path']))
        if record['contract']!=latest['contract'] or record['ledger']['cap']!=latest['ledger']['cap']:raise ValueError('incompatible pinned checkpoint')
    account=Ledger.restore(latest['ledger'],cap=latest['ledger']['cap'])
    if account.pending and latest['states']['data']['pending_batch'] is None:raise ValueError('pending debit without batch')
    scheduler=latest['states']['scheduler']
    if scheduler['bytes']>1<<20:raise ValueError('scheduler size cap')
    elapsed=scheduler_seconds(confined(base,scheduler['path']).read_bytes())
    all_generations={Path(p).relative_to(base.relative_to(command)).parts[0] for p in by_path
        if Path(p).is_relative_to(base.relative_to(command)) and len(Path(p).relative_to(base.relative_to(command)).parts)>1}
    return seal(dict(version='fresh-pilot-checkpoint-diagnostic/v1',command=receipt['identity'],
        original_status=receipt['status'],study=receipt['study'],state=pointer,
        contract=latest['contract'],target_token_cap=account.cap,
        committed_tokens=account.consumed-(account.pending[1] if account.debited else 0),
        charged_tokens=account.consumed,updates=account.updates,pending_tokens=account.pending[1] if account.pending else 0,
        pending_debited=account.debited,learner_seconds=elapsed,
        pinned_generations=len(pins),unreferenced_generation_count=len(all_generations-generations),
        verified_bytes=total,verified_files=len(checked),resume_authorized=False,
        scope='saved-state integrity and accounting only; no scientific or resume approval'))


def collect(root):
    root=Path(root);runs=[]
    for layout in ('execution','research'):
        for arm in ('static','refreshed'):
            relative=layout+'/train-'+arm;command=confined(root,relative,exists=False)
            if not (command/'result.json').is_file() or not (command/'training').is_dir():continue
            row=dict(arm=arm,command_path=relative)
            try:row.update(status='VERIFIED',checkpoint=inspect(command))
            except Exception as exc:row.update(status='UNVERIFIED',error_type=type(exc).__name__)
            runs.append(row)
    return seal(dict(version='fresh-pilot-checkpoint-summary/v1',runs=runs,
        status='PASS' if runs and all(r['status']=='VERIFIED' for r in runs) else 'BLOCKED',
        qualification_evidence=False,resume_authorized=False))


def main(argv=None):
    import argparse
    from oeis_learn.pipeline_validation.identity import write
    from oeis_learn.pipeline_validation.bundle import pack,verify
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,help='original stopped pilot root; read only')
    parser.add_argument('--output',required=True,help='fresh sibling diagnostic output root')
    args=parser.parse_args(argv)
    try:
        root=confined(ROOT,args.run);output=confined(ROOT,args.output,exists=False)
        if output.exists() or output.with_suffix('.zip').exists() or root==output or root in output.parents or output in root.parents:
            raise ValueError('fresh disjoint diagnostic output required')
        result=collect(root)
        write(output/'checkpoints.json',result)
        write(output/'summary.json',dict(selected_scope=result['status'],scientific_status='UNMEASURED',
            scope='read-only saved checkpoint integrity/accounting diagnostic; no learning or resume authorization'))
        archive=pack(output,mandatory=['summary.json','checkpoints.json']);verify(archive)
        print(json.dumps(dict(status=result['status'],archive=str(archive.relative_to(ROOT)),runs=result['runs']),indent=2))
        return 0 if result['status']=='PASS' else 2
    except Exception as exc:
        print('Checkpoint diagnostic failed: '+type(exc).__name__,file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
