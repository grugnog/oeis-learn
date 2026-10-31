"""Portable prospective routing stays stdlib-only and performs no HIP work."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest


def host_module():
    path=Path(__file__).resolve().parents[2]/'scripts/foundation_local.py'
    spec=importlib.util.spec_from_file_location('generalization_host',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_verify_bundle_parser():
    args=host_module().parser().parse_args(['--image','sha256:'+'1'*64,'generalization','verify-bundle',
        '--archive','run-state/source.zip','--output','run-state/verified'])
    assert args.operation=='verify-bundle' and args.archive=='run-state/source.zip'


def test_missing_archive_rejected_without_engine(tmp_path):
    from oeis_learn.pipeline_validation.generalization_cli import main
    args=SimpleNamespace(operation='verify-bundle',archive='absent.zip',output='new',image='sha256:'+'1'*64)
    def forbidden(*args):pytest.fail('engine touched before input validation')
    with pytest.raises(ValueError):main(args,tmp_path,forbidden)


def test_report_parser_and_manifest_binding_before_engine(tmp_path):
    import json
    from oeis_learn.pipeline_validation.generalization_cli import main
    args=host_module().parser().parse_args(['generalization','report','--manifest','study.json','--inputs','evidence','--output','reported'])
    (tmp_path/'study.json').write_text('{}');(tmp_path/'evidence').mkdir()
    (tmp_path/'evidence/evidence.json').write_text(json.dumps(dict(study=dict(path='study.json',bytes=2,sha256='sha256:'+'0'*64))))
    def forbidden(*a):pytest.fail('engine touched before study binding')
    with pytest.raises(ValueError,match='study'):main(args,tmp_path,forbidden)


@pytest.mark.parametrize('operation',['preflight','qualify','prepare','evaluate','train','replay'])
def test_registered_runtime_command_parsers(operation):
    argv=['generalization',operation,'--manifest','study.json','--runtime','runtime.json','--output','new']
    if operation not in {'preflight','qualify'}:argv+=['--qualification','qualified']
    if operation=='train':argv+=['--arms','static','refreshed','--seeds','20261016']
    if operation=='evaluate':argv+=['--checkpoints','checkpoint-id']
    args=host_module().parser().parse_args(argv)
    assert args.operation==operation and args.runtime=='runtime.json'


def test_acceptance_parser_has_explicit_frozen_session_inputs():
    args=host_module().parser().parse_args(['generalization','acceptance','--manifest','session.json',
        '--qualification','qualified','--inputs','prepared','--case','refresh','--output','new'])
    assert args.case=='refresh' and args.inputs=='prepared'


def test_command_storage_scope_rejects_checkpoint_and_raw_growth(tmp_path):
    from oeis_learn.pipeline_validation.generalization_lifecycle import scope,guard
    from oeis_learn.pipeline_validation.runner import ResourceBreach
    limits=dict(study_bytes=1024,checkpoint_bytes=10,shard_bytes=100,raw_bytes=10,sharing_bytes=100)
    (tmp_path/'checkpoints').mkdir();(tmp_path/'checkpoints/weights.pt').write_bytes(b'x'*11)
    with scope(tmp_path,limits):
        watcher=guard(tmp_path/'observations')
        with pytest.raises(ResourceBreach,match='checkpoint_bytes'):watcher.check()
    (tmp_path/'checkpoints/weights.pt').unlink();(tmp_path/'logs').mkdir();(tmp_path/'logs/output.log').write_bytes(b'x'*11)
    with scope(tmp_path,limits):
        with pytest.raises(ResourceBreach,match='raw_bytes'):guard(tmp_path/'observations').check()
