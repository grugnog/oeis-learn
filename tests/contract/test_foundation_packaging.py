"""Build the distributable and verify its source closure outside the checkout."""
from pathlib import Path
import runpy
import subprocess
import sys
import zipfile
import pytest


def test_wheel_preserves_every_bound_source_resource(tmp_path):
    root=Path(__file__).resolve().parents[2]
    metadata=tmp_path/'metadata';metadata.mkdir()
    result=subprocess.run([sys.executable,'setup.py','egg_info','--egg-base',str(metadata),
        'build','--build-base',str(tmp_path/'build'),
        'bdist_wheel','--bdist-dir',str(tmp_path/'bdist'),'--dist-dir',str(tmp_path/'dist')],
        cwd=root,text=True,capture_output=True,timeout=120)
    assert result.returncode==0,result.stderr[-6000:]
    [wheel]=(tmp_path/'dist').glob('*.whl')
    installed=tmp_path/'installed';installed.mkdir()
    with zipfile.ZipFile(wheel) as archive:archive.extractall(installed)
    module=runpy.run_path(str(installed/'oeis_learn/experiments/source_snapshot.py'))
    snapshot=module['source_snapshot']()
    assert 'src/oeis_learn/generalization/policies/fit-core-amendment.json' in snapshot
    # The sdist must retain the same resources for a subsequent wheel build.
    files=(metadata/'oeis_learn.egg-info/SOURCES.txt').read_text().splitlines()
    assert 'src/oeis_learn/generalization/policies/fit-core-amendment.json' in files
    policy=installed/'oeis_learn/generalization/policies/fit-core-amendment.json'
    policy.write_text('{}')
    with pytest.raises(ValueError,match='installed wheel source identity mismatch'):
        module['source_snapshot']()
