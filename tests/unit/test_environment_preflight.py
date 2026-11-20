import json,sys
from pathlib import Path
import pytest
from oeis_learn.pipeline_validation import environment_preflight as p

def test_command_failure_retains_diagnostics(tmp_path):
    with pytest.raises(RuntimeError) as exc:
        p.run([sys.executable,'-c','import sys;print("first failure",file=sys.stderr);sys.exit(3)'],tmp_path)
    result=json.loads(str(exc.value))
    assert result['returncode']==3 and 'first failure' in result['stderr']

def test_timeout_reclaims_process_and_preserves_tail(tmp_path):
    with pytest.raises(RuntimeError) as exc:
        p.run([sys.executable,'-u','-c','import time;print("started");time.sleep(10)'],tmp_path,.1)
    result=json.loads(str(exc.value))
    assert result['timed_out'] and result['returncode']<0 and 'started' in result['stdout']

def test_probe_never_requests_gpu_and_cleans_up_after_failure(tmp_path,monkeypatch):
    calls=[]
    class Engine:
        executable='podman'
        def command(self,**kw):
            assert kw['gpu'] is False and not kw.get('network',False)
            assert kw['mounts'][0][1:]==('/work/output',True)
            return ['probe']
        def remove(self,name,timeout):calls.append((name,timeout))
    def fake(argv,*a,**kw):
        if argv[0]=='probe':raise RuntimeError('probe failed')
        return dict(stdout=json.dumps([dict(Id='sha256:'+'a'*64)]))
    monkeypatch.setattr(p,'run',fake)
    with pytest.raises(RuntimeError,match='probe failed'):p.image_probe(Engine(),'test',tmp_path,tmp_path)
    assert len(calls)==1 and calls[0][0].startswith('oeis-preflight-')
    assert not list(tmp_path.iterdir())

def test_write_probe_cleans_up(tmp_path):
    assert p.write_probe(tmp_path)['create_write_rename_remove']
    assert not list(tmp_path.iterdir())


ROOT = Path(__file__).resolve().parents[2]


def isolated_checkout(tmp_path, backend):
    import shutil
    root = tmp_path / "checkout"
    (root / "scripts").mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/check_discovery_environment.py", root / "scripts/check_discovery_environment.py")
    (root / "scripts/discovery_hip_handoff.py").write_text("# fixture layout\n")
    package = root / "src/oeis_learn/pipeline_validation"
    package.mkdir(parents=True)
    (package.parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    if backend:
        shutil.copyfile(Path(p.__file__), package / "environment_preflight.py")
    (root / "run-state/prior-session").mkdir(parents=True)
    (root / "run-state/prior-session/journal.json").write_text('{"original":"retain"}')
    return root


@pytest.mark.parametrize("backend", [False, True])
def test_missing_checkout_modules_produce_zip_and_preserve_prior_state(tmp_path, backend):
    import os,subprocess,zipfile
    root = isolated_checkout(tmp_path, backend)
    env = dict(os.environ, PATH="", PYTHONPATH="")
    result = subprocess.run([sys.executable, "-S", str(root / "scripts/check_discovery_environment.py"),
                             "--hash-all", "--probe-images"],
                            cwd=tmp_path, env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 1, result.stderr
    assert "Traceback" not in result.stderr
    assert "Share:" in result.stdout
    [archive] = (root / "run-state").glob("environment-preflight-*/preflight.zip")
    with zipfile.ZipFile(archive) as z:
        report = json.loads(z.read("report.json"))
        assert set(z.namelist()) <= {"report.json", "check_discovery_environment.py", "environment_preflight.py"}
    rows = {r["name"]: r for r in report["checks"]}
    if backend:
        missing = rows["checkout module progressive_research_bridge"]
        assert missing["status"] == "FAIL" and missing["error"] == "ModuleNotFoundError"
        assert rows["diagnostic directory write/rename"]["status"] == "PASS"
        assert rows["container engine and requested limits"]["status"] == "FAIL"
        assert rows["canonical full original-input audit"]["status"] == "FAIL"
    else:
        assert rows["checkout bootstrap"]["error"] == "ModuleNotFoundError"
    assert (root / "run-state/prior-session/journal.json").read_text() == '{"original":"retain"}'
    assert not list((root / "src").rglob("__pycache__"))


def test_committed_entrypoint_help_is_stdlib_only_from_other_directory(tmp_path):
    import subprocess
    result = subprocess.run([sys.executable, "-S", str(ROOT / "scripts/check_discovery_environment.py"), "--help"],
                            cwd=tmp_path, text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert "--probe-images" in result.stdout and "--hash-all" in result.stdout


def test_image_probe_matches_source_and_reclaims_without_gpu(tmp_path, monkeypatch):
    from oeis_learn.experiments import source_snapshot
    from oeis_learn.pipeline_validation.identity import digest
    source = {'fixture': 'sha256:' + 'b' * 64}
    monkeypatch.setattr(source_snapshot, 'collect_snapshot', lambda root: source)
    removed = []
    class Engine:
        executable = 'podman'
        def command(self, **kw):
            assert kw['gpu'] is False and not kw.get('network', False)
            assert 'data/sample_oeis/names' in kw['argv'][2]
            assert 'data/sample_oeis/stripped' in kw['argv'][2]
            return ['probe']
        def remove(self, name, timeout):
            removed.append(name)
    def fake(argv, *a, **kw):
        if argv[0] == 'probe':
            return dict(stdout=json.dumps(dict(source=digest(source))))
        return dict(stdout=json.dumps([dict(Id='sha256:' + 'a' * 64)]))
    monkeypatch.setattr(p, 'run', fake)
    assert '_status' not in p.image_probe(Engine(), 'test', tmp_path, tmp_path)
    assert len(removed) == 1
