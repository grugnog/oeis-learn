"""Installed closed role wiring, not an invented success receipt."""
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
from oeis_learn.generalization.progressive.contracts import atomic_write, reference, seal
from oeis_learn.generalization.progressive.projections import describe, project
from oeis_learn.pipeline_validation.progressive_roles import REGISTRY


def run(root, operation, data, kind, channel):
    atomic_write(root/'input.json',seal(dict(version='discovery-channel/v1',kind=kind,role=REGISTRY[operation].role,data=data)))
    project(root,root/'projected',operation,{channel:[describe(root,'input.json')]},registry=REGISTRY)
    code='from pathlib import Path;from oeis_learn.pipeline_validation.progressive_role import execute;import sys;execute(sys.argv[1],inputs=Path(sys.argv[2]),output=Path(sys.argv[3]))'
    p=subprocess.run([sys.executable,'-c',code,operation,str(root/'projected'),str(root/'out')],env=os.environ.copy(),capture_output=True,text=True,timeout=30)
    assert p.returncode==0,p.stderr
    return json.loads((root/'out/result.json').read_text())


def test_installed_source_pilot_reports_independent_deficits(tmp_path):
    from tests.contract.test_progressive_sources import source
    result=run(tmp_path,'source-pilot',dict(version='progressive-source-pilot-input/v1',sources=[source()],roles=['training']),'source','sources')
    assert result['mechanical_status']=='PARTIAL'
    assert not json.loads((tmp_path/'out/feasibility.json').read_text())['primary_scaling_eligible']


def test_installed_verifier_recomputes_public_bundle(tmp_path):
    from tests.contract.test_progressive_bundle import fixture
    from oeis_learn.generalization.progressive.report import build
    request=fixture(tmp_path)
    build(tmp_path,tmp_path/'report.zip',request)
    result=run(tmp_path,'verify',dict(version='progressive-bundle-input/v1',artifact=reference(tmp_path,'report.zip')),'public_evidence','public')
    assert result['mechanical_status']=='COMPLETE'
    verdict=json.loads((tmp_path/'out/verification.json').read_text())
    assert verdict['status']=='VERIFIED' and verdict['decision']=='INCOMPLETE'


def test_verifier_projection_rejects_changed_original_archive(tmp_path):
    atomic_write(tmp_path/'input.json',seal(dict(version='discovery-channel/v1',kind='public_evidence',role='verifier',data=dict(version='progressive-bundle-input/v1',artifact=dict(path='missing.zip',sha256='sha256:'+'1'*64,bytes=10)))))
    with pytest.raises((ValueError,FileNotFoundError)):
        project(tmp_path,tmp_path/'projected','verify',{'public':[describe(tmp_path,'input.json')]},registry=REGISTRY)
