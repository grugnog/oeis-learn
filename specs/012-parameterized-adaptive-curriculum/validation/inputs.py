"""Read-only original-input inventory and installed control-audit adapter.

Host inventory hashes bytes without interpreting protected rows. Native admission
and row counts remain the installed custodian's responsibility.
"""
import json
import os
from pathlib import Path
from oeis_learn.pipeline_validation.identity import confined,seal
from oeis_learn.generalization.contracts import verify_artifact
from oeis_learn.generalization.structural import original_cells

ORIGINAL_ROOTS=(
    'run-state/generalization/fresh-p1-v2-20261103T035534Z',
    'run-state/generalization/fresh-p1-hourly-20261104T093555Z',
    'run-state/generalization/observable-eval-20261104T193226Z/acceptance/report/qualified',
    'run-state/generalization/uniform-import',
)


def discover(root,name,*,max_entries=4096):
    """Bounded optional discovery; explicit roots/inventories take precedence."""
    if type(max_entries) is not int or not 1<=max_entries<=100000:raise ValueError('finite search budget required')
    if not name or Path(name).name!=name:raise ValueError('single inventory filename required')
    root=Path(root)
    if root.is_symlink() or not root.is_dir():raise ValueError('regular explicit root required')
    count=0;found=[]
    for directory,folders,files in os.walk(root,followlinks=False):
        folders.sort();files.sort()
        for entry in folders+files:
            count+=1
            if count>max_entries:raise ValueError('bounded discovery exhausted; supply an explicit original root or inventory')
            path=Path(directory)/entry
            if path.is_symlink():raise ValueError('symlink in original input search; supply regular explicit roots')
            if entry==name and path.is_file():found.append(path)
    if len(found)!=1:raise ValueError('unique original input missing/ambiguous; supply an explicit root or inventory')
    return found[0]


def inventory(root,originals,producer):
    """Verify all sixteen original references against the immutable producer."""
    root=Path(root)
    if not isinstance(originals,list) or len(originals)!=16:raise ValueError('all 16 original exclusion cells required')
    metadata=[]
    for item in originals:
        if set(item)!={'name','role','holdout','count','ref'}:raise ValueError('closed original cell reference required')
        metadata.append({k:v for k,v in item.items() if k!='ref'})
    if sorted(metadata,key=lambda c:c['name'])!=sorted(original_cells(),key=lambda c:c['name']):raise ValueError('all 16 original exclusion identities/counts required')
    receipt=json.loads(verify_artifact(root,producer,max_bytes=16<<20))
    if (seal(receipt)!=receipt or receipt.get('version')!='generalization-command-result/v2'
        or receipt.get('status')!='PASS' or receipt.get('original') is not None):
        raise ValueError('immutable original PASS preparation producer required; failed history stays failed')
    indexed={}
    for ref in receipt.get('inventory',[]):
        if ref['path'] in indexed:raise ValueError('duplicate producer inventory path')
        parent=Path(producer['path']).parent
        path=str(parent/ref['path'])
        indexed[path]=dict(ref,path=path)
    for item in originals:
        ref=item['ref']
        if indexed.get(ref['path'])!=ref:raise ValueError('original cell absent from producer inventory: '+item['name'])
        verify_artifact(root,ref,max_bytes=2<<30)
    return seal(dict(version='pac-input-inventory/v1',status='PASS',producer=producer,originals=originals,
                     native_admission='PENDING',parameter='DEFERRED'))


def original_chain(workspace,*,prepared,pilot,protocol):
    """Reuse reviewed original lineage/retained checkpoint verification, read-only."""
    import importlib.util
    import sys
    prior=Path(__file__).resolve().parents[2]/'011-controlled-generalization/validation'
    path=prior/'prospective_evaluation_policy.py'
    spec=importlib.util.spec_from_file_location('pac_original_lineage',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Its original checkpoint verifier imports the adjacent fresh_pilot helper.
    sys.path.insert(0,str(prior))
    try:return module.lineage(workspace,pilot,prepared,protocol)
    finally:sys.path.pop(0)


def audit_control(original,expected,**identities):
    """Installed schema-validated audit; missing original proof remains unavailable."""
    from oeis_learn.generalization.execution import pac_control
    return pac_control(original,expected,**identities)
