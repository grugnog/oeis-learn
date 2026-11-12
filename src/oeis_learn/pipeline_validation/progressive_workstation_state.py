"""Stdlib workstation journal and failure-safe, explicit diagnostic collection.

This archive is a support bundle, never qualification or scientific authority.
Tensor/private inputs remain local. Original public archives are retained intact.
"""
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time
import traceback
import zipfile

from .identity import confined, seal

LIMIT = 512 << 20
LOG_LIMIT = 4 << 20
NAMES = {'receipt.json','producer.json','qualification.json','acceptance.json',
         'result.json','summary.json','cases.json','case-outcomes.json','outcome.json',
         'failures.json','delta.json','reuse.json','qualification-run.json',
         'process.json','command.json','cleanup.json','container-inspect.json',
         'inspection.json','installed.json','acceptance-limits.json','verification.json',
         'environment.json','inputs.json','deficits.json','session.json','journal.json',
         'runtime.json','campaign.json','telemetry.jsonl','telemetry-previous.jsonl','telemetry-status.json','pipeline.zip'}
NAMES.update({'source.json','matrix.json','profile.json','recipe.json','measurements.json',
              'request.json','image.json','blueprint.json','prepared.json','selection.json',
              'inspection-error.json','partial.json','hardware-results.json','dispositions.json'})
NAMES.update({'registration.json', 'registration-progress.json', 'retained-inputs.json',
              'exposure-import.json', 'exposure-progress.json', 'research-design.json',
              'selected-profiles.json', 'sensitivity-design.json', 'source-candidates.json',
              'source-projection.json', 'source-progress.json', 'registration-status.json'})
NAMES.update({'registered-data.json','rehearsal-reference.json','research-preflight-plan.json',
              'census.json','census-reuse.json','census-progress.json','sensitivity.json','readiness.json'})
NAMES.update({'materials.json','support-policy.json','support-summary.json','research-round-design.json',
              'screening-summary.json','roster-summary.json'})
NAMES.update({'checkpoint-retirement.json','checkpoint-retirement-intent.json',
              'calibration-case.json','calibration-cases.json'})
NAMES.update({'attempt-index.json','portfolio-clocks.json','attempts.jsonl'})
NAMES.update({'preparation-allowance.json','prepare-allowance.json','qualify-allowance.json','qualification-allowance.json','calibration-allowance.json','hardware-allowance.json','census-host-allowance.json','reporting-allowance.json','census-allowance.json','research-reservation.json','construction-summary.json','execution-contract.json','evaluation-cases.json',
              'evaluation-feasibility.json','evaluation-progress.json','evaluation-diagnostics.json','resource-estimate.json','timing.json','running.json'})
PRIVATE = {'private','work','installed','dist','checkpoints','inputs','.runtime'}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.pending')
    with temp.open('w') as f:
        json.dump(value,f,sort_keys=True,indent=2,allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def read(path):
    if Path(path).is_symlink() or Path(path).stat().st_size > 32<<20:
        raise ValueError('bounded original session record required')
    value=json.loads(Path(path).read_bytes())
    if not isinstance(value,dict) or seal(value)!=value:
        raise ValueError('session record identity differs')
    return value


class Blocked(ValueError):
    pass


class Session:
    def __init__(self, root, folder, binding, phase):
        self.root=Path(root); self.folder=confined(root,folder,exists=False)
        if not self.folder.is_relative_to(self.root/'run-state'):
            raise ValueError('session must live under workspace run-state')
        self.folder.mkdir(parents=True,exist_ok=True)
        self.phase=phase; self.binding=binding
        self.journal=dict(version='progressive-workstation-journal/v1',steps={},errors=[])
        self.interrupted=False

    @contextlib.contextmanager
    def own(self):
        lock=self.folder/'session.lock'
        if lock.is_symlink(): raise ValueError('session lock alias')
        with lock.open('a') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                try:
                    if (self.folder/'journal.json').exists():
                        self.journal=read(self.folder/'journal.json')
                    for step in self.journal['steps'].values():
                        if step['status']=='RUNNING':
                            step.update(status='INTERRUPTED',reason='previous supervisor ended without a terminal receipt; no automatic budget reset')
                    path=self.folder/'session.json'
                    if path.exists():
                        old=read(path)
                        if old['binding']!=self.binding:
                            raise Blocked('SOURCE_OR_RESOURCE_SCOPE_CHANGED: retain old evidence; create a new session and requalify')
                    else:
                        write(path,seal(dict(version='progressive-workstation-session/v1',binding=self.binding)))
                    self.save()
                    yield self
                except BaseException as exc:
                    self.journal['errors'].append(dict(phase=self.phase,type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
                    self.save()
                    raise
            finally:
                # Originals survive a packaging error or power loss. A later
                # invocation reconstructs the archive, without rerunning work.
                import sys
                original_failure=sys.exc_info()[0] is not None
                try: self.package()
                except Exception as exc:
                    self.journal['errors'].append(dict(phase=self.phase,type='PACKAGING_FAILED',message=str(exc)))
                    self.save()
                    print('Diagnostic packaging failed; original files remain at '+str(self.folder),flush=True)
                    # A collector failure must not replace the producer error.
                    if not original_failure: raise

    def save(self):
        write(self.folder/'journal.json',seal(self.journal))

    def step(self,name,action,*,dependencies=(),recover=False,binding=None):
        key=self.phase+'/'+name
        steps=self.journal['steps']
        if key in steps:
            old=steps[key]
            if binding is not None and old.get('binding')!=binding and not recover:
                raise ValueError('completed step input binding changed')
            if old['status']=='COMPLETE' and (binding is None or old.get('binding')==binding):
                try:
                    for ref in old.get('outputs',[]):
                        path=confined(self.root,ref['path'])
                        if path.stat().st_size!=ref['bytes'] or sha(path)!=ref['sha256']:
                            raise ValueError('completed stage output changed: '+ref['path'])
                except (OSError,ValueError):
                    if not recover:raise
                else:
                    print('Reusing original stage: '+key,flush=True)
                    return old.get('value')
            if old['status']!='BLOCKED' and not recover:
                print('Retaining prior '+old['status']+': '+key,flush=True)
                return None
        missing=[d for d in dependencies if steps.get(d,{}).get('status')!='COMPLETE']
        row=dict(status='BLOCKED' if missing else 'RUNNING',dependencies=list(dependencies),started_unix=time.time(),reason=','.join(missing))
        if binding is not None:row['binding']=binding
        if key in steps:
            old=dict(steps[key]);history=list(old.pop('history',[]));row['history']=[*history,old]
        steps[key]=row; self.save()
        if missing: return None
        print('Starting '+key,flush=True)
        began=time.monotonic()
        try:
            value, files=action()
            row.update(status='COMPLETE',value=value,outputs=[dict(path=str(Path(p).relative_to(self.root)),bytes=Path(p).stat().st_size,sha256=sha(p)) for p in files])
            return value
        except KeyboardInterrupt:
            row.update(status='INTERRUPTED',reason='operator interruption')
            self.interrupted=True
            raise
        except Exception as exc:
            row.update(status='BLOCKED' if isinstance(exc,Blocked) else 'FAILED',reason=str(exc),traceback=traceback.format_exc())
            print(key+': '+row['status']+': '+str(exc),flush=True)
            return None
        finally:
            row['elapsed_seconds']=time.monotonic()-began
            self.save()
            self.package()

    def code(self):
        rows=[v for k,v in self.journal['steps'].items() if k.startswith(self.phase+'/')]
        if any(r['status']=='INTERRUPTED' for r in rows): return 130
        if any(r['status']=='FAILED' for r in rows): return 1
        return 2 if not rows or any(r['status']!='COMPLETE' for r in rows) else 0

    def package(self):
        observer=getattr(self,'observer',None)
        with observer.lock if observer is not None else contextlib.nullcontext():
            if observer is not None:observer.flush()
            return self._package()

    def _package(self):
        share=self.folder/'share'; share.mkdir(exist_ok=True)
        target=share/(self.phase+'.zip'); temp=share/(self.phase+'.pending.zip')
        files=[]; omissions=[]; total=0
        # Only the session's owned roots and named public/log outputs. Never
        # recurse into private truth, weights, or installed package directories.
        candidates=[]
        for directory, dirs, names in os.walk(self.folder,followlinks=False):
            dirs[:]=sorted(d for d in dirs if d not in PRIVATE|{'share'} and not (Path(directory)/d).is_symlink())
            for name in sorted(names):
                path=Path(directory)/name
                relative=path.relative_to(self.folder)
                if path.is_symlink():
                    omissions.append(dict(path=str(relative),reason='symlink')); continue
                import re
                if name in NAMES or path.suffix in {'.log','.xml'} or name=='diagnostics.zip' or re.fullmatch(r'execution-\d{4}\.zip',name):
                    candidates.append(path)
        # Put the complete first-failure journal before bulky evidence.
        candidates.sort(key=lambda p:(p.name!='journal.json',str(p)))
        with zipfile.ZipFile(temp,'w',compression=zipfile.ZIP_DEFLATED,allowZip64=True) as archive:
            for path in candidates:
                name=str(path.relative_to(self.folder)); size=path.stat().st_size
                cap=LOG_LIMIT if path.suffix=='.log' else LIMIT
                if size>cap or total+size>LIMIT-(1<<20):
                    omissions.append(dict(path=name,bytes=size,sha256=sha(path),reason='size bound; original retained locally'))
                    if path.suffix=='.log' and total+65536<LIMIT-(1<<20):
                        with path.open('rb') as f:
                            prefix=f.read(32768); f.seek(max(0,size-32768)); tail=f.read(32768)
                        data=prefix+b'\n[OMITTED MIDDLE; ORIGINAL RETAINED]\n'+tail
                        archive.writestr(name+'.excerpt',data); total+=len(data)
                        files.append(dict(path=name+'.excerpt',bytes=len(data),sha256='sha256:'+hashlib.sha256(data).hexdigest()))
                    continue
                archive.write(path,name); total+=size
                files.append(dict(path=name,bytes=size,sha256=sha(path)))
            manifest=seal(dict(version='progressive-workstation-support/v1',phase=self.phase,
                qualified=False,scientific_status='UNMEASURED',files=files,omissions=omissions,
                note='Support archive; consult original qualification/scientific receipts. Private inputs and tensors stay local.'))
            archive.writestr('manifest.json',json.dumps(manifest,sort_keys=True))
        os.replace(temp,target)
        print('Share diagnostics: '+str(target),flush=True)
        return target


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return 'sha256:'+h.hexdigest()


def telemetry(root):
    import glob
    values=dict(unix=time.time(),boot_id=None,kernel=platform.release(),free_disk_bytes=shutil.disk_usage(root).free,
                free_inodes=os.statvfs(root).f_favail,memory={},thermal={},throughput=None,
                throughput_reason='use original learner progress; host heartbeat is not committed work')
    try: values['boot_id']=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError: pass
    try:
        for line in Path('/proc/meminfo').read_text().splitlines():
            key,rest=line.split(':',1)
            if key in {'MemTotal','MemAvailable','SwapTotal','SwapFree','Dirty'}: values['memory'][key]=rest.strip()
    except OSError: values['memory']={'unavailable':'procfs not exposed'}
    for name in glob.glob('/sys/class/drm/card*/device/hwmon/hwmon*/temp*_input'):
        try: values['thermal'][name]=int(Path(name).read_text())
        except (OSError,ValueError): pass
    if not values['thermal']: values['thermal']={'unavailable':'render-node thermal sensors not exposed'}
    return values


class TelemetryCollector:
    """Two bounded windows plus durable lifetime counters and collector errors."""
    def __init__(self,folder,*,segment_bytes=8<<20):
        import threading
        self.folder=Path(folder);self.limit=segment_bytes;self.lock=threading.RLock()
        path=self.folder/'telemetry-status.json'
        self.status=read(path) if path.exists() else dict(version='progressive-host-telemetry/v1',
            observations=0,omitted_observations=0,write_failures=0,rotations=0,
            current_observations=0,previous_observations=0,last_observation=None,errors=[])

    def flush(self):
        with self.lock:
            try:write(self.folder/'telemetry-status.json',seal(self.status))
            except OSError as exc:
                self.status['write_failures']+=1
                self.status['errors']=(self.status['errors']+[dict(type=type(exc).__name__,message=str(exc)[:512])])[-32:]
                print('Telemetry status unavailable: '+type(exc).__name__,flush=True)

    def failed(self,error):
        """A failed sensor poll is an omission, never the last good sample."""
        with self.lock:
            self.status['observations']+=1
            self.status['omitted_observations']+=1
            self.status['write_failures']+=1
            self.status['errors']=(self.status['errors']+[dict(type=type(error).__name__,message=str(error)[:512])])[-32:]
            print('Telemetry polling failed: '+type(error).__name__,flush=True)
            self.flush()

    def observe(self,value):
        with self.lock:
            self.status['observations']+=1
            raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
            path=self.folder/'telemetry.jsonl';previous=self.folder/'telemetry-previous.jsonl'
            try:
                if len(raw)>self.limit:raise OSError('observation exceeds telemetry segment bound')
                if path.exists() and path.stat().st_size+len(raw)>self.limit:
                    os.replace(path,previous)
                    self.status['omitted_observations']+=self.status['previous_observations']
                    self.status.update(previous_observations=self.status['current_observations'],current_observations=0,
                        rotations=self.status['rotations']+1)
                with path.open('ab') as stream:stream.write(raw);stream.flush()
                self.status['current_observations']+=1
                self.status['last_observation']=value
            except OSError as exc:
                self.status['omitted_observations']+=1;self.status['write_failures']+=1
                self.status['errors']=(self.status['errors']+[dict(type=type(exc).__name__,message=str(exc)[:512])])[-32:]
                print('Telemetry collection failed: '+type(exc).__name__,flush=True)
            self.flush()
