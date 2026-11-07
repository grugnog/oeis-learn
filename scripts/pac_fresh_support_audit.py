#!/usr/bin/env python3
"""Read-only, stdlib support census for the blocked PAC structural protocol.

Run from the original checkout, including when this script is copied into its
ignored run-state directory. No package imports, engine, model or GPU is used.
Only counts and input identities leave the exclusion-data reader.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import zipfile

PROTOCOL='sha256:f5b92efafe8394952de1ff1b00429288fbcf461f66aaeade44608ee9ee496441'
ORIGINALS=('selection-parameter','selection-structural','assessment-parameter','assessment-structural',
           'static',*(f'refresh-{i:02}' for i in range(1,12)))
FILE_CAP=256<<20
TOTAL_CAP=4<<30
SECONDS=1800


def digest(data):return 'sha256:'+hashlib.sha256(data).hexdigest()


def seal(value):
    value={k:v for k,v in value.items() if k!='identity'}
    return dict(value,identity=digest(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()))


def confined(root,name):
    relative=Path(name)
    if relative.is_absolute() or '..' in relative.parts:raise ValueError('relative confined path required')
    target=root
    for part in relative.parts:
        target=target/part
        if target.is_symlink():raise ValueError('symlink input/output')
    target.resolve().relative_to(root.resolve())
    return target


def read_ref(root,ref):
    if set(ref)!={'path','sha256','bytes'} or type(ref['bytes']) is not int or not 0<=ref['bytes']<=FILE_CAP:
        raise ValueError('bounded original reference required')
    path=confined(root,ref['path'])
    if not path.is_file() or path.stat().st_size!=ref['bytes']:raise ValueError('reference mismatch: '+ref['path'])
    data=path.read_bytes()
    if len(data)!=ref['bytes'] or digest(data)!=ref['sha256']:raise ValueError('reference mismatch: '+ref['path'])
    return json.loads(data)


def witnesses(protocol):
    """One guaranteed 20-term window per coefficient class of every assignment.

    Complete coverage is sufficient to rule out the entire family under this
    exclusion rule. An uncovered window establishes neither an admissible task
    nor support elsewhere in the joint parameter grid.
    """
    if protocol!=seal(protocol) or protocol['identity']!=PROTOCOL:
        raise ValueError('exact reviewed protocol required for these witness formulas')
    result={}
    for family in protocol['families']:
        prefix=family[7:15]
        if prefix=='3f7f434b':
            result[family]={p:tuple(str(p-n) for n in range(20)) for p in range(-1024,1025)}
        elif prefix=='5c475cec':
            # n=13..32 is wholly in the second branch, independently of p0.
            result[family]={p:tuple(str(2*p*n*n) for n in range(13,33)) for p in range(-1024,1025)}
        elif prefix=='83a8423d':
            # n=0..19 is wholly in the first branch, independently of p1.
            result[family]={p:tuple(str(n*n+p*n) for n in range(20)) for p in range(-1024,1025)}
        elif prefix=='ee89096d':
            # p0+p1 spans every integer in [-2048,2048].
            result[family]={s:tuple(str(n-s) for n in range(20)) for s in range(-2048,2049)}
    if len(result)!=4:raise ValueError('four reviewed families required')
    return result


def run(workspace,plan_name,output,expected_plan_sha256):
    started=time.monotonic();deadline=started+SECONDS
    workspace=Path(workspace).resolve(strict=True)
    plan_path=confined(workspace,plan_name)
    if plan_path.stat().st_size>4<<20:raise ValueError('plan byte cap')
    raw=plan_path.read_bytes()
    if digest(raw)!=expected_plan_sha256:raise ValueError('original plan hash mismatch')
    plan=json.loads(raw)
    if plan!=seal(plan) or plan.get('phase')!='structural':raise ValueError('sealed frozen structural plan required')
    cases=witnesses(plan['protocol']);inputs=plan['inputs']
    originals=inputs['originals']
    if len(originals)!=16 or sorted(r['name'] for r in originals)!=sorted(ORIGINALS):
        raise ValueError('all sixteen original exclusion cells required')
    sources=[(r['name'],r['ref'],r['count']) for r in originals]
    sources.extend([('observed-PAC',inputs['observed'],128),('training-probes',inputs['probes'],256)])
    if any(type(ref.get('bytes')) is not int or ref['bytes']<0 for _,ref,_ in sources):raise ValueError('bounded references required')
    if sum(ref['bytes'] for _,ref,_ in sources)>TOTAL_CAP:raise ValueError('total input byte cap')
    destination=confined(workspace,output)
    if destination==workspace or destination.exists():raise ValueError('fresh diagnostic output required')
    destination.mkdir(parents=True)
    lookup=defaultdict(list)
    for family,windows in cases.items():
        for coefficient,window in windows.items():lookup[window].append((family,coefficient))
    covered={f:set() for f in cases};cells=[]
    for name,ref,count in sources:
        if time.monotonic()>=deadline:raise TimeoutError('support audit deadline')
        print('support-audit: scanning '+name,flush=True)
        rows=read_ref(workspace,ref)
        if not isinstance(rows,list) or len(rows)!=count:raise ValueError('original cell count changed: '+name)
        matched={f:set() for f in cases}
        for ordinal,row in enumerate(rows):
            if ordinal%512==0 and time.monotonic()>=deadline:raise TimeoutError('support audit deadline')
            values=row['values']
            if len(values)!=100 or any(type(v) is not str for v in values):raise ValueError('original full-horizon string values required')
            if name=='training-probes' and row.get('role')!='training':raise ValueError('training-only probe exclusion required')
            for start in range(81):
                for family,coefficient in lookup.get(tuple(values[start:start+20]),()):matched[family].add(coefficient)
        del rows
        counts={}
        for family in cases:
            counts[family]=dict(matched_classes=len(matched[family]),new_classes=len(matched[family]-covered[family]))
            covered[family].update(matched[family])
        cells.append(dict(name=name,input=ref,rows=count,families=counts))
    families={f:dict(category=plan['protocol']['families'][f]['category'],
        witness_classes=len(w),covered_classes=len(covered[f]),uncovered_classes=len(w)-len(covered[f]),
        all_assignments_blocked=len(w)==len(covered[f]),
        conclusion='NO_SUPPORT_UNDER_CURRENT_WINDOW_EXCLUSION' if len(w)==len(covered[f]) else 'UNDETERMINED_NOT_AN_ELIGIBILITY_WITNESS')
        for f,w in cases.items()}
    if time.monotonic()>=deadline:raise TimeoutError('support audit deadline')
    result=seal(dict(version='pac-fresh-support-audit/v1',status='PASS',scientific_status='DIAGNOSTIC_ONLY',
        freeze_eligible=False,training_tokens=0,assessment='EXCLUSION_ONLY',
        limits=dict(file_bytes=FILE_CAP,total_input_bytes=TOTAL_CAP,seconds=SECONDS),elapsed_seconds=time.monotonic()-started,
        plan=dict(path=plan_name,sha256=digest(raw),bytes=len(raw),identity=plan['identity']),
        protocol=plan['protocol']['identity'],audit_source=digest(Path(__file__).read_bytes()),
        families=families,cells=cells,
        interpretation='complete witness coverage proves every allowed assignment has an excluded window; uncovered classes do not prove admissibility'))
    (destination/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    lines=['# Fresh structural support audit','','Diagnostic only; no cohort, model or qualification result.','',
        '| Family | Covered coefficient classes | All assignments blocked? |','|---|---:|---|']
    for f,r in families.items():lines.append(f"| {f[7:19]} | {r['covered_classes']}/{r['witness_classes']} | {r['all_assignments_blocked']} |")
    lines.extend(['','An uncovered class does not establish an eligible fresh task. No original values or source programs are exported.'])
    (destination/'summary.md').write_text('\n'.join(lines)+'\n')
    with zipfile.ZipFile(destination/'report.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for name in ('result.json','summary.md'):archive.write(destination/name,name)
    print(json.dumps(dict(status='PASS',scientific_status='DIAGNOSTIC_ONLY',report=str(destination/'report.zip'))),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',default='.')
    parser.add_argument('--plan',required=True)
    parser.add_argument('--expected-plan-sha256',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try:run(args.workspace,args.plan,args.output,args.expected_plan_sha256)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps(dict(status='FAIL',error=type(exc).__name__,message=str(exc))),file=sys.stderr)
        traceback.print_exc();return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
