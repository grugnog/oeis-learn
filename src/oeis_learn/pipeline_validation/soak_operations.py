"""Real engine UID/export and interrupted-role recovery, no engine socket in roles."""
import copy
import json
import os
from pathlib import Path
from .identity import write,seal,file_hash


def foreign_user_args(arguments,uid=12345):
    arguments=list(arguments)
    if '--user' not in arguments:raise ValueError('explicit original user mapping required')
    arguments[arguments.index('--user')+1]=f'{uid}:{uid}'
    return arguments


def fixture_code(uid,*,wait=False):
    return f'''import json,os,time,errno,shutil
from pathlib import Path
from oeis_learn.tracking.pipeline_profile import device_view
from oeis_learn.pipeline_validation.identity import write
from oeis_learn.pipeline_validation.bundle import pack,verify
from oeis_learn.pipeline_validation.recovery import minimal_archive
from unittest.mock import patch
out=Path('/work/output')
write(out/'devices.json',device_view())
assert os.getuid()=={uid}
readonly=False
try:(Path('/work/input')/'forbidden').write_text('forbidden')
except OSError as exc:readonly=exc.errno in (errno.EROFS,errno.EACCES,errno.EPERM)
assert readonly
write(out/'ready.json',{{'uid':os.getuid(),'input_readonly':readonly}})
if {wait!r}:time.sleep(120)
else:
 root=out/'export-case';root.mkdir(mode=0o777)
 write(root/'summary.json',{{'selected_scope':'FAIL','stages':{{}},'fixture':True}})
 write(root/'invocation.json',{{'fixture':True}})
 # ENOSPC is injected in this owned export only, never by filling the host disk.
 with patch('oeis_learn.pipeline_validation.bundle.zipfile.ZipFile',side_effect=OSError(errno.ENOSPC,'fixture full disk')):
  try:pack(root,mandatory=['summary.json'],cap=16<<20)
  except OSError as exc:assert exc.errno==errno.ENOSPC
  else:raise RuntimeError('expected export failure missing')
 archive,summary=minimal_archive(root,contract={{}},binding={{'fixture':'UID export'}},receipts={{}},error='expected ENOSPC',cap=16<<20)
 manifest=verify(archive);assert manifest['summary_state']=='FAIL'
 proof=out/'failure-export-evidence';proof.mkdir(mode=0o777);proof.chmod(0o777)
 for name in ('summary.json','export-failure.json','stage-index.json'):
  shutil.copyfile(summary.parent/name,proof/name)
 archive.unlink();shutil.rmtree(root)
 write(out/'uid-export.json',{{'uid':os.getuid(),'readonly_input':readonly,'expected_failed_export_verified':True,'manifest':manifest['identity'],'healthy':True}})
'''


def run(host,image,output,container_stage):
    source=output/'input';source.mkdir();write(source/'fixture.json',dict(kind='engine-operations',no_training=True))
    arbitrary=12345 if os.getuid()!=12345 else 12346
    foreign=copy.copy(host);foreign.engine=copy.copy(host.engine)
    args=foreign_user_args(host.engine.user_args(),arbitrary)
    foreign.engine.user_args=lambda:args
    uid_output=output/'arbitrary-uid';uid_output.mkdir(mode=0o777);uid_output.chmod(0o777)
    # Only this disposable UID fixture allows a second numeric namespace user.
    for relative in ('.runtime','.runtime/tmp','.runtime/home','.runtime/cache'):
        p=uid_output/relative;p.mkdir(mode=0o777,exist_ok=True);p.chmod(0o777)
    container_stage(foreign,image,source,uid_output,mode='uid-export',seconds=60,gpu=False,
        argv=['python','-c',fixture_code(arbitrary)])
    uid=json.loads((uid_output/'uid-export.json').read_text())
    if uid['uid']!=arbitrary or not uid['expected_failed_export_verified']:raise RuntimeError('arbitrary UID export incomplete')
    # Verify true external interruption and acknowledged engine reclamation.
    interrupt=output/'interrupted'
    actual_uid=0 if host.engine.name=='docker' and host.engine.facts()['rootless'] else os.getuid()
    try:
        container_stage(host,image,source,interrupt,mode='expected-timeout',seconds=10,gpu=False,
            argv=['python','-c',fixture_code(actual_uid,wait=True)])
    except TimeoutError:
        ready=json.loads((interrupt/'ready.json').read_text())
        cleanup=json.loads((interrupt/'cleanup.json').read_text())
        inspection=json.loads((interrupt/'container-inspect.json').read_text())
        # Inspect occurs before reclaim; force-remove acknowledgement is the drain proof.
        if not ready['input_readonly'] or not cleanup['within_bound']:
            raise RuntimeError('interrupted role lacks acknowledged quiescent cleanup')
    else:raise RuntimeError('expected interrupted role did not time out')
    healthy=output/'healthy-after-interrupt'
    container_stage(host,image,source,healthy,mode='healthy-after-interrupt',seconds=60,gpu=False,
        argv=['python','-c',fixture_code(actual_uid)])
    result=json.loads((healthy/'uid-export.json').read_text())
    if not result['healthy']:raise RuntimeError('engine did not recover after interruption')
    return seal(dict(version='pipeline-soak-operations/v1',engine=host.engine.name,arbitrary_uid=arbitrary,
        permission_export=True,readonly_inputs=True,expected_enospc_recovery=True,
        interrupted_cleanup=True,healthy_after_interruption=True,training=False))


def score_after_drain(host,image,output,container_stage):
    source=output/'input';source.mkdir();write(source/'fixture.json',{'fixture':'public identity versus constant; no final data'})
    code='''import time
from pathlib import Path
from oeis_learn.tracking.pipeline_profile import device_view
from oeis_learn.pipeline_validation.identity import write,digest
from oeis_learn.synthesis.client import SynthesisClient
from oeis_learn.synthesis.reference import evaluate
from oeis_learn.synthesis.admission import reference_identity
out=Path('/work/output');view=device_view();write(out/'devices.json',view);assert view['devices']=={}
indices=list(map(str,range(100)));expected=indices;results=[]
with SynthesisClient('sha256:'+'a'*64,cache_root=out/'private/native-cache') as client:
 for source,correct in [('(program (return $index))',True),('(program (return (int 0)))',False)]:
  deadline=time.monotonic()+4
  actual=client.execute(source,indices,deadline=deadline)
  independent=evaluate(source,indices,client.limits,deadline=deadline)
  assert actual['outcome']=='ok' and independent.outcome=='ok'
  assert actual['values']==list(independent.values)
  assert (actual['values']==expected)==correct
  results.append(dict(source_sha256=digest(source),complete=True,correct=correct,count=len(expected)))
write(out/'scoring.json',dict(version='pipeline-soak-scorer/v1',cases=results,reference=reference_identity(),cpu_only=True,fixture=True,final_access='sealed'))
'''
    container_stage(host,image,source,output/'role',mode='isolated-scorer-fixture',seconds=90,gpu=False,argv=['python','-c',code])
    value=json.loads((output/'role/scoring.json').read_text())
    if not value['cpu_only'] or len(value['cases'])!=2 or not all(r['complete'] for r in value['cases']):raise RuntimeError('isolated scorer fixture incomplete')
    return value
