"""Bounded public control copies, with metadata-only discovery of nested roles."""
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
from .identity import seal,write

JSON_CAP=256<<10
XML_CAP=2<<20
LOG_CAP=64<<10
TOTAL_CAP=4<<20
CASES=('parameter-targets','refresh','retained-eval','replay-resume')
CONTROL=('process.json','cleanup.json','stdout.log','stderr.log')
EXPORTED=frozenset({'COPIED','TRUNCATED','PROJECTED'})
MESSAGE_CAP=2048
ROLE_SCAN_ENTRIES=8192
ROLE_SCAN_DIRS=2048
ROLE_SCAN_ROLES=256
ROLE_SCAN_DEPTH=32
ROLE_SKIP=frozenset({'input','inputs','checkpoints','.runtime','native-cache',
                     'cache','caches','selected','cells'})


def terminal_fields(result):
    """Project bounded terminal diagnostics, excluding all other role payloads."""
    statuses={'PASS','FAIL','BLOCKED','PARTIAL','INCONCLUSIVE','INTERRUPTED'}
    if not isinstance(result,dict) or not isinstance(result.get('status'),str) or result['status'] not in statuses:
        raise ValueError('invalid role terminal status')
    fields={'status':result['status']}
    for key in ('reason','error'):
        value=result.get(key)
        if isinstance(value,str):
            marker='[truncated]'
            fields[key]=value if len(value)<=MESSAGE_CAP else value[:MESSAGE_CAP-len(marker)]+marker
    return fields



def control_fields(value):
    """Public numeric/status projection; command lines and host paths stay private."""
    import math
    if not isinstance(value,dict):raise ValueError('invalid control record')
    result={}
    numeric={'elapsed_seconds','seconds','allowance_seconds','returncode','peak_bytes','trace_peak_bytes',
             'cap_bytes','trace_cap_bytes','free_reserve_bytes','free_bytes','minimum_free_bytes'}
    for key in numeric:
        member=value.get(key)
        if key in value and (member is None or type(member) is int and -(1<<63)<=member<1<<63 or type(member) is float and math.isfinite(member)):
            result[key]=member
    for key in ('acknowledged','within_bound'):
        if type(value.get(key)) is bool:result[key]=value[key]
    if isinstance(value.get('outcome'),str) and value['outcome'] in {'COMPLETE','FAIL','INTERRUPTED'}:result['outcome']=value['outcome']
    for key in ('failure','cleanup_failure'):
        text=value.get(key)
        if isinstance(text,str):
            kind=text.split(':',1)[0]
            result[key+'_type']=kind if kind in {'TimeoutError','OSError','RuntimeError','ResourceBreach','KeyboardInterrupt','TimeoutExpired'} else 'UNACKNOWLEDGED'
    breach=value.get('breach')
    if breach:
        known=('invocation transient artifact cap exceeded','invocation trace cap exceeded',
               'invocation free-space reserve exhausted','invocation disk accounting unavailable',
               'disk accounting did not stop within cleanup deadline','symlink in study output')
        result['breach']=next((k for k in known if isinstance(breach,str) and breach.startswith(k)),'RESOURCE_BREACH')
        if isinstance(breach,str) and breach in {'study reservation exceeded: '+k for k in ('checkpoint_bytes','shard_bytes','raw_bytes')}:
            result['breach']=breach
    observation=value.get('breach_observation')
    if isinstance(observation,dict):
        result['breach_observation']={k:v for k,v in observation.items() if k in {'artifact_bytes','trace_bytes','free_bytes'} and type(v) is int}
    return result

def nested_controls(root):
    """Discover control paths and results for projection, without reading payloads."""
    pending=[];names=[];omissions=[];entries=directories=roles=0
    for case in (*CASES,'failure-packaging'):
        relative=Path(case)/'case/private'
        if any((root/Path(*relative.parts[:i])).is_symlink() for i in range(1,len(relative.parts)+1)):
            omissions.append(dict(source=str(relative),status='OMITTED',reason='symlink'))
        elif (root/relative).is_dir():pending.append((relative,0))
    bounded=False
    while pending and not bounded:
        relative,depth=pending.pop();directories+=1
        if directories>ROLE_SCAN_DIRS or depth>ROLE_SCAN_DEPTH:
            bounded=True;break
        try:
            # Recheck queued directories; never descend links or role payloads.
            if any((root/Path(*relative.parts[:i])).is_symlink() for i in range(1,len(relative.parts)+1)):
                omissions.append(dict(source=str(relative),status='OMITTED',reason='symlink'));continue
            with os.scandir(root/relative) as children:
                for child in children:
                    entries+=1
                    if entries>ROLE_SCAN_ENTRIES:bounded=True;break
                    if child.name in ROLE_SKIP:continue
                    if child.is_symlink():
                        omissions.append(dict(source=str(relative/child.name),status='OMITTED',reason='symlink'));continue
                    if not child.is_dir(follow_symlinks=False):continue
                    path=relative/child.name
                    if child.name=='role':
                        if roles>=ROLE_SCAN_ROLES:bounded=True;break
                        roles+=1
                        names.extend(str(path/'supervision'/name) for name in CONTROL)
                        names.append(str(path/'result.json'))
                        names.append(str(path/'disk.json'))
                    else:pending.append((path,depth+1))
        except OSError as exc:
            omissions.append(dict(source=str(relative),status='OMITTED',reason=type(exc).__name__))
    if bounded:
        omissions.append(dict(source='nested-role-discovery',status='OMITTED',reason='nested role discovery bound'))
    return sorted(names),omissions


def candidates(*,nested=()):
    # Fixed public paths plus controls and projected outcomes of bounded roles.
    # Prioritize terminal explanations before potentially large suite logs/XML.
    for stage in ('preflight','prepare',*CASES,'failure-packaging','report','verify-bundle'):
        for name in ('result.json','summary.json'):yield f'{stage}/{name}'
    from oeis_learn.generalization.feature_acceptance import OBLIGATIONS
    for case,(_,labels) in OBLIGATIONS.items():
        for name in ('receipt.json','case-result.json',*(label+'.json' for label in labels)):
            yield f'{case}/case/{name}'
    yield 'replay-resume/bootstrap.json'
    for stage in ('preflight','qualify','prepare',*CASES,'failure-packaging','report','verify-bundle'):
        for name in CONTROL:yield f'private/logs/{stage}/{name}'
        yield f'private/logs/{stage}/disk/disk.json'
    for position in range(2):
        yield f'qualify/current/image-{position}/cleanup.json'
        for name in CONTROL:yield f'qualify/current/image-{position}/logs/{name}'
    for name in CONTROL:yield 'qualify/current/device/logs/'+name
    yield 'qualify/authority-validation/role/result.json'
    yield 'qualify/authority-validation/role/disk.json'
    for name in CONTROL:yield 'qualify/authority-validation/role/supervision/'+name
    yield 'verify-bundle/inspection/cleanup.json'
    for name in CONTROL:yield 'verify-bundle/inspection/logs/'+name
    yield from nested
    for operation in ('report/packaging','verify-bundle/verification'):
        yield operation+'/role/result.json'
        for name in CONTROL:yield operation+'/role/supervision/'+name
    for stage in ('prepare',*CASES,'failure-packaging'):
        base=f'{stage}/qualification'
        for position in range(2):
            yield f'{base}/inspection/image-{position}/cleanup.json'
            for name in CONTROL:yield f'{base}/inspection/image-{position}/logs/{name}'
        for name in CONTROL:yield f'{base}/inspection/device/logs/{name}'
        yield f'{base}/verification/role/result.json'
        for name in CONTROL:yield f'{base}/verification/role/supervision/{name}'
    for stage in ('preflight','prepare'):
        for case in CASES:
            role=f'{stage}/{case}/validation/role'
            yield role+'/result.json'
            for name in CONTROL:yield role+'/supervision/'+name
    for case in CASES:
        role=f'{case}/validation/role'
        yield role+'/result.json'
        for name in CONTROL:yield role+'/supervision/'+name
    # The serial runner calls this directory qualify; selector probes use
    # qualification. Keep both fixed layouts without traversing private roles.
    for base in ('qualify','qualification'):
        yield f'{base}/pipeline/summary.json'
        for stage in ('image.qualification','cpu.semantic','image.gpu','hip.preflight'):
            yield f'{base}/pipeline/stages/{stage}/receipt.json'
            yield f'{base}/pipeline/stages/{stage}/inspection/cleanup.json'
            for name in CONTROL:yield f'{base}/pipeline/stages/{stage}/inspection/logs/{name}'
        for stage,suite in (('cpu.semantic','bigint-cpu'),('hip.preflight','bigint-gpu')):
            for step in ('environment',suite+'-regressions'):
                for name in (*CONTROL,'receipt.json','results.xml'):
                    yield f'{base}/pipeline/stages/{stage}/{step}/{name}'
    yield 'report/inspection/cleanup.json'
    for name in CONTROL:yield 'report/inspection/logs/'+name


def collect(root,destination,*,cap=TOTAL_CAP):
    """Snapshot controls with explicit omissions; originals and status are untouched."""
    if type(cap) is not int or not 0<=cap<=TOTAL_CAP:raise ValueError('diagnostic byte cap')
    root=Path(root).resolve(strict=True);destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    nested,omissions=nested_controls(root);nested=set(nested)
    rows=[];used=0
    names=list(dict.fromkeys(candidates(nested=sorted(nested))))
    names=list(dict.fromkeys([*names,*[str(Path(n).parent/'disk.json') for n in names if n.endswith('/role/result.json')]]))
    # Keep all terminal/process/disk records ahead of bulky logs and suite XML.
    names.sort(key=lambda n:Path(n).suffix in {'.log','.xml'})
    for name in names:
        relative=Path(name)
        outcome=relative.name=='result.json' and relative.parent.name=='role'
        control=relative.name in {'process.json','cleanup.json','disk.json'}
        projected=outcome or control
        control_parent=relative.parent/'supervision' if outcome else relative.parent
        # Archive paths must never expose private directories as public payloads.
        export_path=(str(Path('nested-roles')/hashlib.sha256(str(control_parent).encode()).hexdigest()/
                         ('outcome.json' if outcome else relative.name))
                     if name in nested or 'private' in relative.parts else name)
        path=root/name;row=dict(source=name,export_path=export_path);rows.append(row)
        try:
            if any((root/Path(*relative.parts[:i])).is_symlink() for i in range(1,len(relative.parts)+1)):
                raise ValueError('symlink')
            if not path.exists():row.update(status='MISSING');continue
            # Nonblocking/no-follow also rejects FIFO/link replacements at open.
            fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK|os.O_NOFOLLOW)
            with os.fdopen(fd,'rb') as stream:
                before=os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):raise ValueError('nonregular file')
                size=before.st_size;row['original_bytes']=size
                limit=LOG_CAP if path.suffix=='.log' else XML_CAP if path.suffix=='.xml' else JSON_CAP
                if path.suffix!='.log' and size>limit:raise ValueError(('XML' if path.suffix=='.xml' else 'JSON')+' byte cap')
                if min(size,limit)>cap-used:raise ValueError('total byte cap')
                if size>limit:
                    marker=b'\n...[diagnostic log truncated: opening and final tail]...\n'
                    half=(limit-len(marker))//2
                    data=stream.read(half)+marker
                    stream.seek(-(limit-half-len(marker)),os.SEEK_END)
                    data+=stream.read(limit-half-len(marker))
                    status='TRUNCATED'
                else:data=stream.read(limit+1);status='COPIED'
                after=os.fstat(stream.fileno())
                if len(data)>limit or (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                    raise ValueError('file changed during capture')
            if projected:
                row['original_sha256']='sha256:'+hashlib.sha256(data).hexdigest()
                data=(json.dumps((terminal_fields if outcome else control_fields)(json.loads(data)),sort_keys=True)+'\n').encode()
                status='PROJECTED'
                if len(data)>cap-used:raise ValueError('total byte cap')
            target=destination/export_path;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as stream:stream.write(data)
            used+=len(data)
            row.update(status=status,exported_bytes=len(data),sha256='sha256:'+hashlib.sha256(data).hexdigest())
        except (OSError,ValueError,RecursionError) as exc:
            row.update(status='OMITTED',reason=str(exc) if isinstance(exc,ValueError) else type(exc).__name__)
    rows.extend(omissions)
    result=seal(dict(version='generalization-diagnostics/v1',qualification_evidence=False,
        exported_bytes=used,payload_cap=cap,json_cap=JSON_CAP,xml_cap=XML_CAP,log_cap=LOG_CAP,files=rows))
    write(destination/'index.json',result)
    return result


def export(root,destination):
    """Package existing allowed records without executing or mutating the run."""
    from .bundle import pack,verify
    root=Path(root).resolve(strict=True);destination=Path(destination).resolve()
    archive=destination.with_suffix('.zip')
    if (not root.is_dir() or destination==root or root in destination.parents or
        destination in root.parents or destination.exists() or archive.exists()):
        raise ValueError('fresh disjoint diagnostic output and archive required')
    destination.mkdir(parents=True,exist_ok=False)
    index=collect(root,destination/'diagnostics')
    write(destination/'summary.json',seal(dict(selected_scope='PARTIAL',
        reason='DIAGNOSTIC_EXPORT_ONLY',feature_qualified=False,scientific_status='UNMEASURED')))
    mandatory=['summary.json','diagnostics/index.json']
    mandatory.extend('diagnostics/'+r['export_path'] for r in index['files'] if r['status'] in EXPORTED)
    archive=pack(destination,mandatory=mandatory)
    verify(archive)
    return dict(archive=str(archive),qualification_evidence=False,exported_bytes=index['exported_bytes'])


def main(argv=None):
    import argparse,json
    parser=argparse.ArgumentParser(description='Export bounded existing diagnostics; no container or HIP execution.')
    parser.add_argument('--run',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(argv)
    try:result=export(args.run,args.output)
    except (OSError,ValueError) as exc:
        parser.exit(1,str(exc)+'\n')
    print(json.dumps(result,sort_keys=True));return 0


if __name__=='__main__':raise SystemExit(main())


class PacObservers:
    """Model-bound monitoring and B shadow opportunities, never policy credit."""
    def __init__(self,batches,*,arm,seed,observe,semantic=None,snapshot=None,acceptance=False,coordinate_clock=None):
        self.batches=batches;self.arm=arm;self.seed=seed;self.observe=observe;self.semantic=semantic
        self.coordinate_clock=coordinate_clock
        from oeis_learn.generalization.retention import empty_history,validate_history,append_panel
        self.block=2 if acceptance else 8;self.cycle=4*self.block;self.common_period=8 if acceptance else 256
        self.families=sorted(batches.families);self.anchors=sorted(batches.anchors)
        self.state=dict(version='pac-acceptance-observers/v2' if acceptance else 'pac-observers/v2',arm=arm,seed=seed,inventory=batches.inventory['identity'],
            last_before=-1,last_common=-1,pending_shadow=None,shadow_blocks=0,shadow_grades=0,
            loss_panels=0,semantic_attempts=0,semantic_completed=0,semantic_success=0,
            allocation={f:{c:dict(updates=0,tokens=0,examples=0) for c in ('fixed','bandit','review','fallback')} for f in self.families},
            review_counts=dict(acquisitions=0,grades=0,successes=0,censored=0,incidental=0),reward_counts=dict(valid=0,censored=0),
            rotation={},missing_rotations={f:0 for f in self.families},retention_incomplete=[],recent=[],history=empty_history())
        if snapshot is not None:
            snapshot=copy.deepcopy(snapshot)
            if snapshot.get('version')==self.state['version'].replace('/v2','/v1'):
                snapshot['version']=self.state['version'];snapshot['history']=empty_history(legacy_panels=snapshot['loss_panels'])
            validate_history(snapshot.get('history',{}),self.families,snapshot['loss_panels'])
            if set(snapshot)!=set(self.state) or any(snapshot[k]!=self.state[k] for k in ('version','arm','seed','inventory')):raise ValueError('PAC observer restore identity')
            from oeis_learn.generalization.budgets import integer
            for key in ('shadow_blocks','shadow_grades','loss_panels','semantic_attempts','semantic_completed','semantic_success'):integer(snapshot[key])
            if not snapshot['semantic_success']<=snapshot['semantic_completed']<=snapshot['semantic_attempts']:raise ValueError('PAC observer semantic counters')
            if any(type(snapshot[k]) is not int or snapshot[k]<-1 for k in ('last_before','last_common')) or len(snapshot['recent'])>64:raise ValueError('PAC observer clock/history bounds')
            if set(snapshot['missing_rotations'])!=set(self.families) or not set(snapshot['rotation'])<=set(self.families) or not set(snapshot['retention_incomplete'])<=set(self.families):raise ValueError('PAC observer family coverage')
            for value in snapshot['missing_rotations'].values():integer(value)
            if any(type(v) is not bool for v in snapshot['rotation'].values()):raise ValueError('PAC observer rotation disposition')
            self.state=copy.deepcopy(snapshot)

    def event(self,value):
        self.state['recent'].append(value);self.state['recent']=self.state['recent'][-64:]

    def common(self,update,*,final=False):
        if self.state['last_common']==update or (update%self.common_period and not final):return
        sums=0.;tokens=0;complete=True;families={}
        for f in self.families:
            result=self.observe([self.batches.probe_rows[i] for i in self.batches.probes[f]])
            families[f]=None if result is None else dict(loss_sum=result['loss_sum'],tokens=result['tokens'])
            if result is None:complete=False
            else:sums+=result['loss_sum'];tokens+=result['tokens']
        self.state['loss_panels']+=1
        self.event(dict(kind='common_loss',update=update,loss_sum=sums,tokens=tokens,complete=complete,families=families,kernel_credit=False))
        start=((update+self.common_period-1)//self.common_period*4)%len(self.families)
        selected=[self.families[(start+i)%len(self.families)] for i in range(min(4,len(self.families)))]
        semantic={f:dict(semantic_attempts=0,semantic_completed=0,semantic_success=0) for f in self.families}
        for f in selected:
            completed=False
            for key in self.batches.probes[f]:
                self.state['semantic_attempts']+=1;semantic[f]['semantic_attempts']+=1
                result=self.semantic(self.batches.probe_rows[key],self.batches.probe_truth[key],seconds=5) if self.semantic else None
                if result is not None:
                    completed=True;self.state['semantic_completed']+=1;self.state['semantic_success']+=bool(result)
                    semantic[f]['semantic_completed']+=1;semantic[f]['semantic_success']+=int(bool(result))
            self.state['rotation'][f]=completed
        if len(self.state['rotation'])==len(self.families):
            for f,complete in self.state['rotation'].items():
                self.state['missing_rotations'][f]=0 if complete else self.state['missing_rotations'][f]+1
                if self.state['missing_rotations'][f]>=2 and f not in self.state['retention_incomplete']:self.state['retention_incomplete'].append(f)
            self.state['rotation']={}
        from oeis_learn.generalization.retention import append_panel
        allocation=[v for f in self.state['allocation'].values() for v in f.values()]
        coordinates=dict(update=update,tokens=sum(v['tokens'] for v in allocation),examples=sum(v['examples'] for v in allocation),seconds=self.coordinate_clock() if self.coordinate_clock else None)
        append_panel(self.state['history'],coordinates,{f:dict(loss=families[f],semantic=semantic[f]) for f in self.families})
        self.state['last_common']=update

    def before(self,update):
        if self.state['last_before']==update:return
        self.common(update)
        if self.arm=='B':
            import random
            from oeis_learn.generalization.curriculum import stream_seed
            cycle,pos=divmod(update,self.cycle)
            if pos in (0,self.block,2*self.block):
                number=cycle*3+pos//self.block
                f=self.families[random.Random(stream_seed(self.seed,'shadow-uniform-family',str(number))).randrange(len(self.families))]
                ids=self.batches.probes[f][(number%4)*2:(number%4)*2+2]
                self.state['pending_shadow']=dict(start=update,ids=ids,pre=self.observe([self.batches.probe_rows[i] for i in ids]))
            elif pos>=3*self.block:
                anchor=self.anchors[(cycle*self.block+pos-3*self.block)%len(self.anchors)]
                result=self.observe([self.batches.rows[anchor]]);self.state['shadow_grades']+=1
                self.event(dict(kind='shadow_recall',anchor=anchor,update=update,recall=None if result is None else result['recall'][0],anchor_promotion=False))
        self.state['last_before']=update

    def after(self,update):
        pending=self.state['pending_shadow']
        if pending and update==pending['start']+self.block:
            post=self.observe([self.batches.probe_rows[i] for i in pending['ids']])
            pre=pending['pre'];self.state['shadow_blocks']+=1
            self.event(dict(kind='shadow_reward',update=update,raw_gain=None if pre is None or post is None else pre['loss_sum']/pre['tokens']-post['loss_sum']/post['tokens'],kernel_credit=False))
            self.state['pending_shadow']=None
        self.common(update)

    def committed(self,batch,*,channel,anchor=None,grade=None,before=None):
        from oeis_learn.generalization.budgets import charge_tokens
        for row in batch:
            value=self.state['allocation'][row['family']][channel]
            value['tokens']+=charge_tokens([row['tokens']],bos=1,pad=0);value['examples']+=1
        # Adaptive batches are single-family; fixed batches may cross families.
        from fractions import Fraction
        for family in {r['family'] for r in batch}:
            value=self.state['allocation'][family][channel]
            value['updates']=str(Fraction(value['updates'])+Fraction(sum(r['family']==family for r in batch),len(batch)))
        if before is not None:
            counts=self.state['review_counts']
            for row in batch:
                a=before.get(row['id'])
                if a is not None:
                    counts['acquisitions']+=not a['acquired']
                    counts['incidental']+=row['id']!=anchor
            if anchor and before[anchor]['acquired']:
                counts['grades']+=grade is not None;counts['successes']+=grade is not None and grade['recall']
                counts['censored']+=grade is None

    def feedback(self,event,selection):
        self.state['reward_counts']['censored' if event['status']=='censored' else 'valid']+=1
        self.event(dict(kind='policy_reward',selection=dict(selection),**event))

    def snapshot(self):return copy.deepcopy(self.state)


def pac_training_projection(controller,ledger,*,discarded=0,timing=None):
    """Bounded public measurements, never a source of resumed policy authority."""
    if controller is None or ledger is None:return dict(status='UNAVAILABLE',reason='original_measurements_missing')
    from oeis_learn.generalization.budgets import Ledger
    account=Ledger.restore(ledger,cap=ledger['cap'])
    reserved=account.pending[1] if account.debited else 0
    tokens=dict(allowed=account.cap,useful=account.consumed-reserved,reserved=reserved,discarded=discarded,
                unspent=account.cap-account.consumed-discarded)
    if min(tokens.values())<0:raise ValueError('unreconciled PAC exposure')
    observer=controller['observer'];policy=controller.get('policy')
    allocation=observer.get('allocation',{})
    if allocation:
        if sum(v['tokens'] for f in allocation.values() for v in f.values())!=tokens['useful']:raise ValueError('allocation does not reconcile useful tokens')
        if sum(__import__('fractions').Fraction(v['updates']) for f in allocation.values() for v in f.values())!=account.updates:raise ValueError('allocation does not reconcile committed updates')
    return dict(status='RECORDED',tokens=tokens,updates=account.updates,channels=dict(controller['channels']),allocation=allocation,
        review=observer.get('review_counts'),rewards=observer.get('reward_counts'),
        clocks=dict(actions=policy['kernel']['action_count'],valid_rewards=policy['kernel']['valid_updates']) if policy else None,
        pending_anchors=sum(a['due_update']<=account.updates for a in policy['review']['anchors'].values()) if policy else None,
        pending_block=policy.get('block') if policy else None,last_reward=policy.get('last_observation') if policy else None,
        pauses=policy.get('pauses') if policy else [],observations=controller['observations'],observation_seconds=controller['observed_seconds'],
        costs={k:(timing or {}).get(k) for k in ('preparation_seconds','checkpoint_seconds','update_seconds','publication_seconds','invocation_seconds')},
        monitoring=observer,recent_observations_omitted=max(0,observer.get('loss_panels',0)+observer.get('shadow_blocks',0)-len(observer['recent'])))
