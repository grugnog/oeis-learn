"""Installed prospective role entrypoint; only declared channels are available."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
import time

from oeis_learn.generalization.role_contracts import Role,ROLE_INPUTS,ROLE_OUTPUTS
from .identity import seal,write


def preparation(inputs,output,settings,*,seconds):
    from oeis_learn.generalization.contracts import validate
    from oeis_learn.synthesis.client import SynthesisClient
    from .generalization_preparation import prepare_study
    if set(settings)!={'preparation','proposed'}:raise ValueError('closed preparation role settings required')
    authority=validate('preparation',settings['preparation'])
    deadline=time.monotonic()+seconds
    # Assemble only the explicitly mounted preparation inputs. Other role roots,
    # the repository and runtime socket are absent from the container.
    with tempfile.TemporaryDirectory(prefix='generalization-authority-') as temporary:
        root=Path(temporary)
        for channel in inputs.values():
            for path in channel.rglob('*'):
                if path.is_symlink():raise ValueError('symlink in input projection')
                if path.is_file():
                    target=root/path.relative_to(channel)
                    if target.exists():raise ValueError('conflicting projected authority path')
                    target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        with SynthesisClient(authority.identity,cache_root=output/'private/native-cache') as native:
            def execute(source,indices):
                return native.execute(source,indices,deadline=min(deadline,time.monotonic()+4))
            return prepare_study(root,authority,settings['proposed'],output=output/'private/prepared',execute=execute)


def report(inputs,output,settings,*,seconds):
    from oeis_learn.generalization.contracts import validate,verify_artifact
    from oeis_learn.generalization.reconciliation import reconcile_measurements
    if set(settings)!={'study','invocation','attempts'}:raise ValueError('closed report role settings required')
    root=inputs['public_evidence']
    study=json.loads(verify_artifact(root,settings['study']));validate('study',study)
    invocation=json.loads(verify_artifact(root,settings['invocation']))
    attempts=json.loads(verify_artifact(root,settings['attempts']))
    result=reconcile_measurements(invocation,attempts,study)
    for name,payload in result.items():write(output/(name+'.json'),payload)
    return dict(status='PASS',scientific_status='UNMEASURED',study=study['identity'],
                scope='recorded finite evidence',summary=result['summary'])


def learner(inputs,output,settings,*,seconds):
    from oeis_learn.fresh.bigint_worker import generalization_train
    if set(settings)!={'config','resume'}:raise ValueError('closed learner role settings required')
    config=settings['config']
    if config['seconds']>seconds:raise ValueError('learner exceeds enclosing deadline')
    resume=(inputs['verified_training_replay'],settings['resume']) if settings['resume'] is not None else None
    return generalization_train(inputs['training_targets'],output/'private/learner',config,resume=resume)


HANDLERS={(Role.PREPARATION,'prepare'):preparation,(Role.REPORT,'report'):report,
          (Role.LEARNER,'train'):learner}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role',choices=[r.value for r in Role]);parser.add_argument('runtime')
    args=parser.parse_args(argv);role=Role(args.role);runtime=json.loads(args.runtime)
    if set(runtime)!={'settings','operation','seconds','device'}:raise ValueError('closed role invocation required')
    output=Path(ROLE_OUTPUTS[role].destination)
    inputs={m.channel:Path(m.destination) for m in ROLE_INPUTS[role]}
    try:
        handler=HANDLERS[(role,runtime['operation'])]
        result=handler(inputs,output,runtime['settings'],seconds=runtime['seconds'])
    except KeyError:
        result=dict(status='BLOCKED',reason='UNREGISTERED_ROLE_OPERATION')
    except (ValueError,OSError,RuntimeError,TimeoutError) as exc:
        result=dict(status='FAIL',reason=type(exc).__name__,error=str(exc))
    write(output/'result.json',seal(result))
    # A terminal role result is transported even for a negative software outcome.
    # The canonical parent maps that result to the public exit status.
    return 0


if __name__=='__main__':raise SystemExit(main())
