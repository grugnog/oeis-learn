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
