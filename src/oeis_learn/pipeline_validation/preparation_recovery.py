"""Durable finite operational allowance; unknown intervals never refund time."""
import contextlib
import fcntl
import math
import time
from pathlib import Path
from .identity import seal
from .progressive_workstation_state import write,read,Blocked


class Allowance:
    def __init__(self,path,binding,seconds):
        self.path=Path(path);self.binding=binding;self.seconds=seconds
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<=0:
            raise ValueError('finite original recovery allowance required')

    @contextlib.contextmanager
    def attempt(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        if self.path.is_symlink() or self.path.with_suffix('.lock').is_symlink():
            raise ValueError('recovery allowance alias')
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            state=read(self.path) if self.path.exists() else dict(version='progressive-operational-allowance/v1',
                binding=self.binding,seconds=self.seconds,spent_seconds=0,active=None,attempts=[])
            if state['binding']!=self.binding or state['seconds']!=self.seconds:
                raise Blocked('original recovery scope/allowance changed')
            if state['active'] is not None:
                state['attempts'].append(dict(state['active'],outcome='UNKNOWN_INTERRUPTION',charged_seconds=state['active']['reserved_seconds']))
                state.update(spent_seconds=self.seconds,active=None)
                write(self.path,seal(state))
            remaining=self.seconds-state['spent_seconds']
            if remaining<=0 or len(state['attempts'])>=256:
                raise Blocked('original operational allowance exhausted; unknown crash intervals are fully charged')
            active=dict(ordinal=len(state['attempts']),reserved_seconds=remaining,started_unix=time.time())
            state['active']=active;write(self.path,seal(state));began=time.monotonic()
            outcome='COMPLETE';error=None;unsafe=False
            try:
                yield remaining,active['ordinal']
            except BaseException as exc:
                from .runner import ResourceBreach
                outcome=type(exc).__name__;error=str(exc)[:4096];unsafe=isinstance(exc,(ResourceBreach,KeyboardInterrupt,SystemExit))
                raise
            finally:
                charged=remaining if unsafe else min(remaining,max(0,time.monotonic()-began))
                state['attempts'].append(dict(active,outcome=outcome,error=error,charged_seconds=charged))
                state.update(spent_seconds=min(self.seconds,state['spent_seconds']+charged),active=None)
                write(self.path,seal(state))


def preserve_process(output):
    """Move only old supervisor records; keep immutable case/data publications."""
    import shutil
    output=Path(output)
    names=('command.json','process.json','stdout.log','stderr.log','cleanup.json','container-inspect.json','inspection-error.json')
    present=[output/n for n in names if (output/n).exists()]
    if not present:return
    process=output/'process.json'
    if not process.exists():raise Blocked('prior process termination is unproven')
    value=__import__('json').loads(process.read_bytes())
    if value.get('cleanup_failure') or value.get('returncode') is None:
        raise Blocked('prior process termination is unacknowledged')
    cleanup=output/'cleanup.json'
    if cleanup.exists() and __import__('json').loads(cleanup.read_bytes()).get('acknowledged') is not True:
        raise Blocked('prior container cleanup is unacknowledged')
    history=output/'recovery-processes';history.mkdir(exist_ok=True)
    target=history/str(len(list(history.iterdir()))).zfill(4);target.mkdir()
    for path in present:shutil.move(path,target/path.name)


def retain(path,value):
    """Reuse an identical immutable publication; changed originals are errors."""
    from oeis_learn.generalization.progressive.contracts import load,atomic_write,CAMPAIGN_DOCUMENT_BYTES
    path=Path(path)
    if path.exists():
        if load(path,maximum=CAMPAIGN_DOCUMENT_BYTES)!=value:raise ValueError('original recovery publication differs: '+str(path))
    else:atomic_write(path,value)


def revise(path,value):
    """Append a new operational summary while retaining its original bytes."""
    import shutil
    from .identity import file_hash
    path=Path(path)
    if path.exists():
        history=path.parent/'recovery-records';history.mkdir(exist_ok=True)
        original=history/(path.stem+'-'+file_hash(path)[7:]+path.suffix)
        if not original.exists():shutil.copyfile(path,original)
    write(path,value)


from .runner import ResourceBreach
class UnsafeRecovery(ResourceBreach,ValueError):
    """An unacknowledged device/process state consumes remaining allowance."""
