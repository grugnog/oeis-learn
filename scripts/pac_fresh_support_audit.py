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
import math
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
N=2049
MAX_OUTPUT=2*1024*99**2


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


class AllWindows:
    """Exact inverse images of 20-term windows in the four bounded families.

    A quadratic segment blocks a row/column of the two-parameter grid;
    a segment across a branch boundary blocks a point. Sparse byte rows bound
    the point storage by 2 * 2049**2 bytes without enumerating target programs.
    """
    def __init__(self,families):
        self.ids={f[7:15]:f for f in families}
        self.minus=bytearray(N);self.sums=bytearray(4097)
        self.columns13=bytearray(N);self.rows42=bytearray(N);self.columns42=bytearray(N)
        self.points={'5c475cec':{},'83a8423d':{}}

    def point(self,family,p0,p1):
        if not (-1024<=p0<=1024 and -1024<=p1<=1024):return
        rows=self.points[family];i=p0+1024
        if i not in rows:rows[i]=bytearray(N)
        rows[i][p1+1024]=1

    @staticmethod
    def interval(mask,lo,hi,offset):
        lo=max(lo,-offset);hi=min(hi,len(mask)-offset-1)
        if lo<=hi:mask[lo+offset:hi+offset+1]=b'\x01'*(hi-lo+1)

    def project(self,w,first=None,last=None):
        if len(w)!=20:return
        if first is None:
            if any(abs(x)>MAX_OUTPUT for x in w):return
            bad=[j for j in range(17) if w[j+3]-3*w[j+2]+3*w[j+1]-w[j]]
            first,last=(bad[0],bad[-1]) if bad else (-1,-1)
        if first<0:
            self.quadratic(w)
            return
        # In a piecewise quadratic the third difference can be nonzero only
        # across the join: positions m-3..m-1, where m is its window offset.
        if last-first>2:return
        for m in range(last+1,min(19,first+3)+1):
            if m<=13 and w[m]%338==0:
                p1=w[m]//338;k=13-m;j=0 if k else 1;divisor=(k+j)**2
                if -1024<=p1<=1024 and w[j]%divisor==0:
                    p0=w[j]//divisor
                    if -1024<=p0<=1024 and all(value==(p0*n*n if n<=12 else 2*p1*n*n)
                            for n,value in zip(range(k,k+20),w)):
                        self.point('5c475cec',p0,p1)
            if w[m]%3528==0:
                p1=w[m]//3528;k=42-m
                if -1024<=p1<=1024 and (w[0]-k*k)%k==0:
                    p0=(w[0]-k*k)//k
                    if -1024<=p0<=1024 and all(value==(n*n+p0*n if n<42 else 2*p1*n*n)
                            for n,value in zip(range(k,k+20),w)):
                        self.point('83a8423d',p0,p1)

    def quadratic(self,w):
        second=w[2]-2*w[1]+w[0]
        if second%2:return
        a=second//2;b=w[1]-w[0]-a;c=w[0]  # a*j*j+b*j+c
        if a==0:
            if b==-1:self.interval(self.minus,c,c+80,1024)
            if b==1:self.interval(self.sums,-c,80-c,2048)
            if b==0 and c==0:self.columns13[1024]=self.columns42[1024]=1
            return
        if a==1:
            disc=b*b-4*c
            if disc>=0:
                root=math.isqrt(disc)
                if root*root==disc:
                    for numerator in (b+root,b-root):
                        if numerator%2==0:
                            k=numerator//2;p0=b-2*k
                            if 0<=k<=22 and -1024<=p0<=1024:self.rows42[p0+1024]=1
            # A globally quadratic window can cross the 42 join with just
            # one suffix term (start 23). Two suffix terms would force p1=1/2.
            p0=b-46
            if c==23*(23+p0) and (p0+42)%84==0:self.point('83a8423d',p0,(p0+42)//84)
        if a%2==0 and abs(a)<=2048 and b%(2*a)==0:
            k=b//(2*a);p1=a//2
            if c!=a*k*k:return
            if 13<=k<=80:self.columns13[p1+1024]=1
            if 42<=k<=80:self.columns42[p1+1024]=1
            # Equal coefficients across the 13 join; or just one prefix term
            # before the 42 join (start 41).
            if 0<=k<=12:self.point('5c475cec',a,p1)
            if k==41:self.point('83a8423d',41*(a-1),p1)

    def observe(self,values):
        numbers=[]
        for value in values:
            # Longer strings cannot equal any output in these bounded families.
            if len(value)>10:numbers.append(None);continue
            n=int(value)
            if str(n)!=value:raise ValueError('canonical integer outputs required')
            numbers.append(n if abs(n)<=MAX_OUTPUT else None)
        invalid=[0]
        for n in numbers:invalid.append(invalid[-1]+(n is None))
        bad=[False]*100
        for j in range(97):
            part=numbers[j:j+4]
            bad[j]=any(n is None for n in part) or bool(part[3]-3*part[2]+3*part[1]-part[0])
        next_bad=[100]*100;previous_bad=[-1]*100;position=100
        for j in range(99,-1,-1):
            if bad[j]:position=j
            next_bad[j]=position
        position=-1
        for j in range(100):
            if bad[j]:position=j
            previous_bad[j]=position
        for start in range(81):
            if invalid[start+20]!=invalid[start]:continue
            first=next_bad[start]
            if first>start+16:self.project(numbers[start:start+20],-1,-1)
            else:self.project(numbers[start:start+20],first-start,previous_bad[start+16]-start)

    def blocked(self,family,p0,p1=0):
        prefix=family[7:15];i=p0+1024;j=p1+1024
        if prefix=='3f7f434b':return bool(self.minus[i])
        if prefix=='ee89096d':return bool(self.sums[p0+p1+2048])
        columns=self.columns13 if prefix=='5c475cec' else self.columns42
        row=self.points[prefix].get(i)
        return bool(columns[j] or prefix=='83a8423d' and self.rows42[i] or row is not None and row[j])

    def summary(self):
        result={}
        free_minus=N-sum(self.minus);free_sums=4097-sum(self.sums)
        free_sum_assignments=sum((N-abs(s)) for s in range(-2048,2049) if not self.sums[s+2048])
        for prefix,total,free,outputs in [('3f7f434b',N,free_minus,free_minus),
                                         ('ee89096d',N*N,free_sum_assignments,free_sums)]:
            result[self.ids[prefix]]=dict(allowed_assignments=total,window_free_assignments=free,window_free_output_classes=outputs)
        for prefix,columns in [('5c475cec',self.columns13),('83a8423d',self.columns42)]:
            free_columns=[j for j in range(N) if not columns[j]]
            free_rows=N if prefix=='5c475cec' else N-sum(self.rows42)
            free=free_rows*len(free_columns)
            for i,row in self.points[prefix].items():
                if prefix=='83a8423d' and self.rows42[i]:continue
                free-=sum(row[j] for j in free_columns)
            result[self.ids[prefix]]=dict(allowed_assignments=N*N,window_free_assignments=free,window_free_output_classes=free)
        for value in result.values():
            value.update(window_blocked_assignments=value['allowed_assignments']-value['window_free_assignments'],
                all_assignments_blocked=value['window_free_assignments']==0,
                conclusion='FULL_DOMAIN_WINDOW_BLOCKED' if value['window_free_assignments']==0 else 'WINDOW_FREE_ASSIGNMENTS_REQUIRE_FULL_ADMISSION')
        return result


def run(workspace,plan_name,output,expected_plan_sha256,*,all_windows=False):
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
    census=AllWindows(cases) if all_windows else None
    previous={f:0 for f in cases}
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
            if census:census.observe(values)
            else:
                for start in range(81):
                    for family,coefficient in lookup.get(tuple(values[start:start+20]),()):matched[family].add(coefficient)
        del rows
        counts={}
        if census:
            for family,record in census.summary().items():
                blocked=record['window_blocked_assignments']
                counts[family]=dict(newly_blocked_assignments=blocked-previous[family],cumulative_window_free_assignments=record['window_free_assignments'])
                previous[family]=blocked
        else:
            for family in cases:
                counts[family]=dict(matched_classes=len(matched[family]),new_classes=len(matched[family]-covered[family]))
                covered[family].update(matched[family])
        cells.append(dict(name=name,input=ref,rows=count,families=counts))
    families={f:dict(category=plan['protocol']['families'][f]['category'],
        witness_classes=len(w),covered_classes=len(covered[f]),uncovered_classes=len(w)-len(covered[f]),
        all_assignments_blocked=len(w)==len(covered[f]),
        conclusion='NO_SUPPORT_UNDER_CURRENT_WINDOW_EXCLUSION' if len(w)==len(covered[f]) else 'UNDETERMINED_NOT_AN_ELIGIBILITY_WITNESS')
        for f,w in cases.items()}
    if census:
        families={f:dict(v,category=plan['protocol']['families'][f]['category']) for f,v in census.summary().items()}
    if time.monotonic()>=deadline:raise TimeoutError('support audit deadline')
    interpretation=('exact support under all 81 finite-window positions only; window-free assignments still require full frozen admission'
        if census else 'complete witness coverage proves every allowed assignment has an excluded window; uncovered classes do not prove admissibility')
    result=seal(dict(version='pac-fresh-support-audit/v2' if census else 'pac-fresh-support-audit/v1',status='PASS',scientific_status='DIAGNOSTIC_ONLY',
        freeze_eligible=False,training_tokens=0,assessment='EXCLUSION_ONLY',
        limits=dict(file_bytes=FILE_CAP,total_input_bytes=TOTAL_CAP,seconds=SECONDS),elapsed_seconds=time.monotonic()-started,
        plan=dict(path=plan_name,sha256=digest(raw),bytes=len(raw),identity=plan['identity']),
        protocol=plan['protocol']['identity'],audit_source=digest(Path(__file__).read_bytes()),
        families=families,cells=cells,
        interpretation=interpretation))
    (destination/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
    lines=['# Fresh structural support audit','','Diagnostic only; no cohort, model or qualification result.','',
        '| Family | Covered coefficient classes | All assignments blocked? |','|---|---:|---|']
    if census:lines[4:]=['| Family | Window-free / allowed assignments | All assignments blocked? |','|---|---:|---|']
    for f,r in families.items():
        counts=(f"{r['window_free_assignments']}/{r['allowed_assignments']}" if census else f"{r['covered_classes']}/{r['witness_classes']}")
        lines.append(f"| {f[7:19]} | {counts} | {r['all_assignments_blocked']} |")
    lines.extend(['',interpretation+'. No original values or source programs are exported.'])
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
    parser.add_argument('--all-windows',action='store_true',help='exact full-law window support using all 81 candidate positions')
    args=parser.parse_args()
    try:run(args.workspace,args.plan,args.output,args.expected_plan_sha256,all_windows=args.all_windows)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(json.dumps(dict(status='FAIL',error=type(exc).__name__,message=str(exc))),file=sys.stderr)
        traceback.print_exc();return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
