"""One immutable pilot allowance across invocations; interrupted slots stay charged."""
import json
from pathlib import Path
import time
from .identity import seal,digest,file_hash,confined,write
from oeis_learn.fresh.common import write as replace_json


def boot_identity():
    try:return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:raise ValueError('boot identity unavailable; cannot preserve monotonic pilot allowance') from None


class PilotLedger:
    def __init__(self,root,decision,*,clock=time.monotonic,boot=None):
        self.root=Path(root)/decision['identity'].split(':')[1];self.root.mkdir(parents=True,exist_ok=True)
        self.clock=clock;self.boot=boot or boot_identity();self.path=self.root/'ledger.json'
        if self.path.exists():
            self.value=json.loads(self.path.read_text())
            if seal(self.value)!=self.value:raise ValueError('pilot ledger integrity mismatch')
            if self.value['decision']!=decision or self.value['boot']!=self.boot:raise ValueError('changed pilot decision/boot cannot renew allowance')
        else:
            started=clock()
            self.value=dict(version='pipeline-pilot-ledger/v1',decision=decision,boot=self.boot,started=started,
                expires=started+172800,slots={},invocations=[])
            self.persist()

    def persist(self):
        self.value=seal(self.value)
        replace_json(self.path,self.value)

    def remaining(self):
        now=self.clock()
        if now<self.value['started']:raise ValueError('pilot monotonic clock domain changed')
        return max(0,self.value['expires']-now)

    def reserve(self,key,*,seconds,category,output):
        from .pilot_contract import SPEC
        from .identity import number
        number(seconds,positive=True)
        if self.remaining()<seconds+1800:raise ValueError('remaining original pilot wall allowance cannot fit slot/export')
        if category not in {'learner','evaluation','setup','qualification'}:raise ValueError('registered pilot budget category required')
        if key in self.value['slots']:raise ValueError('pilot slot already charged; no retry/refund')
        used=sum(r['seconds'] for r in self.value['slots'].values() if r['category']==category)
        if used+seconds>SPEC['reservations'][category]:raise ValueError('original cumulative pilot category exhausted')
        row=dict(key=key,seconds=seconds,category=category,output=str(Path(output).resolve()),state='ACTIVE',started=self.clock())
        self.value['slots'][key]=row;self.persist();return row

    def finish(self,key,state,*,result=None):
        if state not in ('COMPLETE','INTERRUPTED'):raise ValueError('explicit pilot terminal state required')
        row=self.value['slots'][key]
        if row['state']!='ACTIVE':raise ValueError('terminal pilot slots are immutable')
        ended=self.clock()
        late=ended-row['started']>row['seconds']
        if late:
            state='INTERRUPTED';result=dict(result or {},fatal=True,error='original wall reservation exceeded')
        row.update(state=state,ended=ended,result=result);self.persist()
        if late:
            from .runner import ResourceBreach
            raise ResourceBreach('pilot slot exceeded original wall reservation')

    def recover(self,key):
        """Only an acknowledged stopped role can support use of its retained state."""
        row=self.value['slots'][key];root=Path(row['output'])
        if (row.get('result') or {}).get('fatal'):raise ValueError('prior fatal audit/resource breach prevents pilot continuation')
        summary=root/'summary.json'
        if summary.exists() and json.loads(summary.read_text()).get('fatal'):raise ValueError('prior fatal role cannot be resumed')
        roles=[root] if (root/'command.json').exists() else [p.parent for p in root.glob('*/command.json')]
        if not roles:roles=[root]
        for role in roles:
            cleanup=role/'cleanup.json'
            if not cleanup.is_file() or json.loads(cleanup.read_text()).get('within_bound') is not True:
                raise ValueError('prior pilot role termination unacknowledged; no further compute')
            if key.startswith('pilot.'):
                if not (role/'summary.json').is_file():raise ValueError('missing prior role verdict; no automatic continuation')
                if json.loads((role/'summary.json').read_text()).get('fatal'):raise ValueError('prior fatal role prevents further compute')
        if row['state']=='ACTIVE':
            self.finish(key,'INTERRUPTED');row=self.value['slots'][key]
        if row['state']!='INTERRUPTED':raise ValueError('only interrupted slots need recovery')
        return row

    def invocation(self,output):
        if self.remaining()<=1800:raise ValueError('original pilot allowance expired')
        if len(self.value['invocations'])>=16:raise ValueError('finite pilot recovery invocation bound exhausted')
        self.value['invocations'].append(str(Path(output).resolve()));self.persist()
