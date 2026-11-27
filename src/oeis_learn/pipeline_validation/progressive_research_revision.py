"""Explicit data-only reuse of the reviewed, unscored preparation v2 census.

Original results are immutable. A separate revision owns the remaining census
clock and its additional fixed counter range; sibling output paths cannot create
new entitlement. No image/HIP/calibration/acceptance authority is transferred.
"""
from pathlib import Path
import json
import os
import sqlite3
import shutil
from .identity import confined, seal
from .progressive_workstation_state import read, write
from .progressive_research_inputs import checked_file, _json
from oeis_learn.generalization.progressive.contracts import reference, digest

# Reviewed producer of 013-research-round-001: data verification and exclusions
# are unchanged by the prospective panel/schedule amendment. Not a Q allowlist.
COMPATIBLE_CENSUS_SOURCES={'sha256:f5076f15ec651e9c2bf9b92f0550ba7ae5ff9475c0de1d57be25b4a0668db481'}


def adopt(root,session,recipient,data):
    root=Path(root);folder=confined(root,session)
    if not folder.is_relative_to(root/'run-state') or folder==Path(recipient):
        raise ValueError('distinct original preparation session required')
    binding=read(folder/'session.json');journal=read(folder/'journal.json')
    if binding['binding']['source'] not in COMPATIBLE_CENSUS_SOURCES:
        raise ValueError('original census producer has not been reviewed for data reuse')
    forbidden={'freeze','acceptance','qualify','qualification','calibrate','evaluation-timing','execution-contract','construction','prepare'}
    if any(k.startswith(('run-research/','assess/')) or
           (k.startswith('prepare-research/') and k.split('/')[1] in forbidden)
           for k in journal['steps']):
        raise ValueError('revision requires an unscored pre-HIP census; later execution exists')
    if any((folder/'prepare-research'/name).exists() for name in ('frozen','acceptance','evaluation-timing','construction','prepared')):
        raise ValueError('later preparation artifacts preclude outcome-blind revision')
    step=journal['steps'].get('prepare-research/native-census',{})
    if step.get('status')!='COMPLETE':raise ValueError('terminal original census producer required')
    originals={r['path']:r for r in step.get('outputs',[])}
    census_path=folder/'prepare-research/census/census.json'
    census_ref=reference(root,census_path)
    if originals.get(census_ref['path'])!=census_ref:raise ValueError('original census not journal-bound')
    census=_json(census_path,maximum=32<<20)
    request_path=census_path.parent/'request.json';request_ref=reference(root,request_path)
    plan=journal['steps'].get('prepare-research/census-plan',{})
    if plan.get('status')!='COMPLETE' or request_ref not in plan.get('outputs',[]):
        raise ValueError('original census request not journal-bound')
    from .progressive_research_census import validate_request
    request=validate_request(_json(request_path))
    if (request['version']!='progressive-research-census-request/v2' or request['diagnostic']
            or seal(census)!=census or census['request']!=request['identity']
            or census['diagnostic'] or census['training_started'] or census['assessment_exposed']
            or census['counts'].get('TRAINING')!=request['training_draws']
            or census['counts'].get('FAILED',0) or census['counts'].get('unattempted',0)):
        raise ValueError('complete original native draw inventory without model outcomes required')
    if any(request[k]!=data[k] for k in ('candidates','exposure','recipe')):
        raise ValueError('original registered census inputs changed')
    artifacts=[]
    for ref in census['artifacts']:
        path=checked_file(census_path.parent,ref);current=reference(root,path)
        if originals.get(current['path'])!=current:raise ValueError('original native artifact not journal-bound')
        artifacts.append(current)
    # Debit both installed work and enclosing process time; never refund startup.
    prior=max(float(census['elapsed_seconds']),float(step['elapsed_seconds']))
    if not 0<=prior<request['seconds']-600:raise ValueError('original census allowance has no usable remainder')
    result=seal(dict(version='progressive-census-data-reuse/v1',census=census_ref,
        request=request_ref,artifacts=artifacts,native=census['native'],
        original_source=binding['binding']['source'],prior_elapsed_seconds=prior,
        original_draws=request['training_draws']+request['evaluation_draws'],
        additional_draws=32768,seconds_remaining=request['seconds']-prior,
        original_session=reference(root,folder/'session.json'),original_journal=reference(root,folder/'journal.json'),
        hardware_authority_transferred=False,assessment_exposed=False))
    # Stable donor-derived owner, outside the historical session. Atomic claim
    # precedes any new work; a crash cannot grant a second session the remainder.
    claim=root/'run-state/discovery-preparation-revisions'/(census['identity'][7:]+'.json')
    claim.parent.mkdir(parents=True,exist_ok=True)
    owner=seal(dict(version='progressive-census-revision-owner/v1',
        original=census['identity'],recipient=str(Path(recipient).relative_to(root)),reuse=result['identity']))
    if claim.exists():
        if claim.is_symlink() or read(claim)!=owner:raise ValueError('census revision already belongs to another session; no sibling budget reset')
    else:
        with claim.open('x') as f:
            json.dump(owner,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
    fd=os.open(claim.parent,os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return result


def initialize(root,output,request,client):
    """Copy verified immutable native data; never execute an original draw again."""
    reuse=request['reuse'];root=Path(root);output=Path(output)
    if reuse['native']!=client.runtime_identity('bigint-body/v1'):
        raise ValueError('original native data verifier differs from current runtime')
    census=_json(checked_file(root,reuse['census']),maximum=32<<20)
    original=_json(checked_file(root,reuse['request']))
    if (seal(reuse)!=reuse or seal(census)!=census or census['request']!=original['identity']
            or census['native']!=reuse['native'] or census['diagnostic']!=request['diagnostic']):
        raise ValueError('original census reuse scope differs')
    if (any(original[k]!=request[k] for k in ('candidates','exposure','recipe','training_draws','seed'))
            or reuse['original_draws']!=original['training_draws']+original['evaluation_draws']):
        raise ValueError('original data/seed/counter commitment changed')
    base=Path(reuse['census']['path']).parent
    expected=[dict(r,path=str(base/r['path'])) for r in census['artifacts']]
    if reuse['artifacts']!=expected:raise ValueError('original census artifact closure differs')
    paths={Path(r['path']).name:checked_file(root,r) for r in reuse['artifacts']}
    if set(paths)!={'census.sqlite','training-exposure.sqlite','training-reservoir.json','panel-reservations.json'}:
        raise ValueError('complete native data closure required')
    private=output/'private';private.mkdir(parents=True,exist_ok=True,mode=0o700)
    if (private/'census.sqlite').exists():return
    donor=sqlite3.connect(paths['census.sqlite'].as_uri()+'?mode=ro',uri=True)
    try:
        if json.loads(donor.execute('SELECT value FROM binding').fetchone()[0])!=original:
            raise ValueError('original native database request changed')
        count,low,high=donor.execute('SELECT count(*),min(ordinal),max(ordinal) FROM draws').fetchone()
        if (count,low,high)!=(reuse['original_draws'],0,reuse['original_draws']-1):
            raise ValueError('complete original draw denominator required')
        statuses=dict(donor.execute('SELECT status,count(*) FROM draws GROUP BY status'))
        if any(statuses.get(k,0)!=census['counts'].get(k,0) for k in set(statuses)|{'TRAINING','EVALUATION','REJECTED'}):
            raise ValueError('original native database counts changed')
        for name in ('training-exposure.sqlite','census.sqlite'):
            temp=private/(name+'.copy')
            shutil.copyfile(paths[name],temp)
            if name=='census.sqlite':
                with sqlite3.connect(temp) as target:
                    target.execute('UPDATE binding SET value=?',(json.dumps(request),))
            with temp.open('rb') as f:os.fsync(f.fileno())
            temp.replace(private/name)
    finally:donor.close()


def reuse_sensitivity(root,reuse,design,output):
    """Reuse unchanged synthetic assumptions/results, never model measurements."""
    root=Path(root);output=Path(output)
    journal=read(checked_file(root,reuse['original_journal']))
    step=journal['steps'].get('prepare-research/sensitivity',{})
    if step.get('status')!='COMPLETE':return None
    folder=checked_file(root,reuse['original_session']).parent/'prepare-research/sensitivity'
    if _json(folder/'design.json')!=design:raise ValueError('original sensitivity assumptions changed')
    refs=step.get('outputs',[])
    for ref in refs:checked_file(root,ref)
    path=folder/'sensitivity.json'
    if reference(root,path) not in refs:raise ValueError('original sensitivity result not journal-bound')
    result=read(path)
    if result.get('status')!='COMPLETE':return None
    cases=read(folder/'case-outcomes.json')
    if result['design']!=digest(design) or cases['design']!=digest(design) or result['cases']!=digest(cases['cases']):
        raise ValueError('original sensitivity case evidence changed')
    write(output/'design.json',design)
    write(output/'sensitivity.json',result)
    # Keep original case tables and their identity, without a second bootstrap.
    for name in ('case-outcomes.json',):
        if (folder/name).exists():shutil.copyfile(folder/name,output/name)
    write(output/'reuse.json',seal(dict(version='progressive-sensitivity-reuse/v1',
        original=reference(root,path),source=reuse['original_source'],
        original_elapsed_seconds=step.get('elapsed_seconds'),hardware_authority_transferred=False)))
    return result,list(output.glob('*.json'))
