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
    from .generalization_reporting import measurement_report
    return measurement_report(study,invocation,attempts,output=output)


def learner(inputs,output,settings,*,seconds):
    from oeis_learn.fresh.bigint_worker import generalization_train
    if set(settings)!={'config','resume'}:raise ValueError('closed learner role settings required')
    config=settings['config']
    if config['seconds']>seconds:raise ValueError('learner exceeds enclosing deadline')
    resume=(inputs['verified_training_replay'],settings['resume']) if settings['resume'] is not None else None
    return generalization_train(inputs['training_targets'],output/'private/learner',config,resume=resume)


def generator(inputs,output,settings,*,seconds):
    from oeis_learn.generalization.contracts import verify_artifact
    from oeis_learn.fresh.bigint_worker import client
    from .generalization_evaluation import evaluate_retained
    from .sizing_execution import execution_context
    expected={'checkpoint','prompts','recipe','study','source','device','bank_seeds'}
    if set(settings)!=expected:raise ValueError('closed generation role settings required')
    prompts=json.loads(verify_artifact(inputs['visible_prompts'],settings['prompts']))
    with execution_context(),client(dict(recipe=settings['recipe'],contract_sha256=settings['study']),output/'private') as native:
        return evaluate_retained(inputs['frozen_weights'],settings['checkpoint'],prompts,settings['recipe'],output/'generation',
            client=native,device=settings['device'],study=settings['study'],source=settings['source'],seconds=seconds,bank_seeds=settings['bank_seeds'])


def selection(inputs,output,settings,*,seconds):
    from oeis_learn.synthesis.client import SynthesisClient
    from .generalization_scoring import score_selection
    if set(settings)!={'targets','candidates','study'}:
        raise ValueError('closed selection settings required')
    with SynthesisClient(settings['study'],cache_root=output/'private/native-cache') as native:
        return score_selection(inputs['selection_supervision'],settings['targets'],
            inputs['selection_candidates'],settings['candidates'],client=native,
            output=output/'selection',seconds=seconds)


def verify_archive(inputs,output,settings,*,seconds):
    from oeis_learn.generalization.contracts import verify_artifact
    from .generalization_archive import verify_archive as verify
    if set(settings)!={'archive'}:raise ValueError('closed archive verification settings required')
    root=inputs['public_evidence']
    verify_artifact(root,settings['archive'],max_bytes=512<<20)
    result=verify(root/settings['archive']['path'])
    write(output/'verification.json',result)
    return dict(status='PASS',verification=result)


def qualification_consume(inputs,output,settings,*,seconds):
    from oeis_learn.generalization.qualification import consume
    from oeis_learn.generalization.contracts import verify_artifact
    if set(settings)!={'authority','current','required_suites','required_capabilities','research','feature_receipt'}:
        raise ValueError('closed qualification consumption settings required')
    root=inputs['public_evidence'];authority=json.loads(verify_artifact(root,settings['authority']))
    original=consume(authority,settings['current'],root=root,required_suites=settings['required_suites'],
        required_capabilities=settings['required_capabilities'],research=settings['research'],feature_receipt=settings['feature_receipt'])
    return dict(status='PASS',original=original,study_qualified=False)


HANDLERS={(Role.REPORT,'verify-archive'):verify_archive,(Role.SELECTION,'score'):selection,(Role.PREPARATION,'prepare'):preparation,(Role.REPORT,'report'):report,
          (Role.REPORT,'qualification-consume'):qualification_consume,
          (Role.LEARNER,'train'):learner,(Role.GENERATOR,'evaluate'):generator}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role',choices=[r.value for r in Role]);parser.add_argument('runtime')
    args=parser.parse_args(argv);role=Role(args.role);runtime=json.loads(args.runtime)
    if set(runtime)!={'settings','operation','seconds','device'}:raise ValueError('closed role invocation required')
    if role in {Role.LEARNER,Role.GENERATOR}:
        import torch
        torch.set_num_threads(2)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    output=Path(ROLE_OUTPUTS[role].destination)
    inputs={m.channel:Path(m.destination) for m in ROLE_INPUTS[role]}
    handler=HANDLERS.get((role,runtime['operation']))
    if handler is None:
        write(output/'result.json',seal(dict(status='BLOCKED',reason='UNREGISTERED_ROLE_OPERATION')))
        return 0
    try:
        selected=runtime['settings'].get('config',runtime['settings'])
        if 'device' in selected and selected['device']!=runtime['device']:raise ValueError('role device capability mismatch')
        result=handler(inputs,output,runtime['settings'],seconds=runtime['seconds'])
    except (KeyError,ValueError,OSError,RuntimeError,TimeoutError) as exc:
        result=dict(status='FAIL',reason=type(exc).__name__,error=str(exc))
    write(output/'result.json',seal(result))
    # A terminal role result is transported even for a negative software outcome.
    # The canonical parent maps that result to the public exit status.
    return 0


if __name__=='__main__':raise SystemExit(main())
