"""Real local Git/LFS integration; no external corpus or hardware required."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

import pytest

from oeis_learn.fresh import bulk, corpus
from oeis_learn.fresh.common import read, write


def config():
    return dict(version='oeis-corpus/v1', cache='corpus', stripped='stripped.gz', revision=None,
                max_source_bytes=1 << 30, max_projection_bytes=1 << 26,
                free_disk_floor_bytes=1, seconds=120, attempts=2,
                retry_seconds=1, concurrent_transfers=2)


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    if not shutil.which('git-lfs'):
        pytest.skip('Git LFS required; installed in both project images')
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', '/dev/null')
    monkeypatch.setenv('GIT_LFS_SKIP_SMUDGE', '1')
    source, remote = tmp_path / 'source', tmp_path / 'remote.git'
    subprocess.run(['git', 'init', '-q', str(source)], check=True)
    subprocess.run(['git', 'init', '-q', '--bare', str(remote)], check=True)
    bulk.git(source, 'lfs', 'install', '--local', '--skip-repo')
    (source / '.gitattributes').write_text('files/** filter=lfs diff=lfs merge=lfs -text\n')
    (source / 'time.txt').write_text('fixture snapshot\n')
    payloads = {}
    for n, keyword, count in [(1, 'nonn', 120), (2, 'nonn', 40), (3, 'fini', 120),
                               (4, 'tabl', 120), (5, 'nonn', 0), (6, 'nonn', 120)]:
        aid = f'A{n:06d}'
        seq = source / f'seq/{aid[:4]}/{aid}.seq'
        seq.parent.mkdir(parents=True, exist_ok=True)
        seq.write_text(f'%N {aid} SECRET formula must not become a target\n'
                       f'%S {aid} 0,{n},{2*n}\n%O {aid} -2,1\n%K {aid} {keyword}\n')
        if count:
            raw = ''.join(f'{i - 2} {i*n}\n' for i in range(count)).encode()
            path = source / f'files/{aid[:4]}/b{n:06d}.txt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            payloads[n] = raw
    (source / 'files/A000/attachment.pdf').write_bytes(b'unrequested supporting attachment')
    bulk.git(source, 'add', '.')
    bulk.git(source, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
             'commit', '-qm', 'full corpus fixture')
    bulk.git(source, 'remote', 'add', 'origin', str(remote))
    bulk.git(source, 'push', '-q', 'origin', 'HEAD')
    bulk.git(source, 'lfs', 'push', '--all', 'origin')
    monkeypatch.setattr(bulk, 'REPOSITORY', str(remote))
    with gzip.open(tmp_path / 'stripped.gz', 'wt') as stream:
        stream.write('# retained cache\nA000001 ,0,1,2,\nA000002 ,9,8,7,\n')
    return remote, payloads


def results(tmp_path):
    with sqlite3.connect(tmp_path / 'corpus/inventory.sqlite') as db:
        return {aid: json.loads(result) for aid, result in db.execute('SELECT aid,result FROM processed')}


def test_full_inventory_native_single_fetch_resume_and_classification(tmp_path, upstream, monkeypatch):
    from oeis_learn.pipeline_validation import supervision
    original, calls = supervision.launch, []
    def counted(argv, *args, **kwargs):
        calls.append(argv)
        return original(argv, *args, **kwargs)
    monkeypatch.setattr(supervision, 'launch', counted)
    report = corpus.prepare(tmp_path, config())
    assert report['status'] == 'COMPLETE'
    assert report['counts']['PROCESSED'] == 6
    assert report['counts']['eligible100'] == 2
    rows = results(tmp_path)
    assert rows['A000001']['cache'] == 'VERIFIED_MAIN_TERMS'
    assert rows['A000002']['cache'] == 'DIFFERENT_RELEASE_OR_TERMS'
    assert rows['A000003']['values'] and 'finite' in rows['A000003']['reasons']
    assert rows['A000004']['values'] and 'table' in rows['A000004']['reasons']
    assert rows['A000005']['available_terms'] == 3
    manifest = read(tmp_path / 'corpus/indexed/manifest.json')
    assert len(manifest['records']) == 2
    assert 'SECRET' not in json.dumps(rows)
    lfs_fetches = [cmd for cmd in calls if 'lfs' in cmd and 'fetch' in cmd]
    assert len(lfs_fetches) == 1 and '--all' not in lfs_fetches[0]
    assert '--include=files/**/b*.txt' in lfs_fetches[0]
    attachment_oid = hashlib.sha256(b'unrequested supporting attachment').hexdigest()
    assert not list((tmp_path / 'corpus/oeisdata.git/lfs/objects').rglob(attachment_oid))
    monkeypatch.setattr(corpus, 'inspect_bfile', lambda *a: pytest.fail('reparsed completed b-file'))
    assert corpus.prepare(tmp_path, config())['status'] == 'COMPLETE'
    assert len([cmd for cmd in calls if 'lfs' in cmd and 'fetch' in cmd]) == 1
    # Current consumer accepts the exported indexed contract and exact offsets.
    from oeis_learn.pipeline_validation.progressive_research_source import project
    projected = project(tmp_path, 'corpus/indexed/manifest.json', tmp_path / 'project')
    assert projected['status'] == 'COMPLETE' and projected['eligible'] == 2


def test_transient_native_failure_retries_queue_without_python_batches(tmp_path, upstream, monkeypatch):
    from oeis_learn.pipeline_validation import supervision
    real, attempts = supervision.launch, []
    def flaky(argv, *args, **kwargs):
        if 'lfs' in argv and 'fetch' in argv:
            attempts.append(argv)
            if len(attempts) == 1:
                raise RuntimeError('transient native transfer failure')
        return real(argv, *args, **kwargs)
    monkeypatch.setattr(supervision, 'launch', flaky)
    monkeypatch.setattr(corpus.time, 'sleep', lambda _: None)
    assert corpus.prepare(tmp_path, config())['status'] == 'COMPLETE'
    assert len(attempts) == 2 and attempts[0] == attempts[1]


def test_missing_object_continues_independent_entries_and_later_resumes(tmp_path, upstream, monkeypatch):
    remote, payloads = upstream
    oid = hashlib.sha256(payloads[6]).hexdigest()
    original = remote / 'lfs/objects' / oid[:2] / oid[2:4] / oid
    original.unlink()
    monkeypatch.setattr(corpus.time, 'sleep', lambda _: None)
    report = corpus.prepare(tmp_path, config())
    assert report['status'] == 'PARTIAL'
    assert report['counts']['BLOCKED'] == 1
    assert report['counts']['PROCESSED'] == 5
    assert not (tmp_path / 'corpus/indexed/manifest.json').exists()
    original.write_bytes(payloads[6])
    real, inspected = corpus.inspect_bfile, []
    def inspected_once(path, *args):
        inspected.append(path.name)
        return real(path, *args)
    monkeypatch.setattr(corpus, 'inspect_bfile', inspected_once)
    assert corpus.prepare(tmp_path, config())['status'] == 'COMPLETE'
    assert inspected == [oid]


def test_declared_budget_blocks_fetch_without_dropping_inventory(tmp_path, upstream, monkeypatch):
    from oeis_learn.pipeline_validation import supervision
    real = supervision.launch
    def guard(argv, *args, **kwargs):
        assert not ('lfs' in argv and 'fetch' in argv)
        return real(argv, *args, **kwargs)
    monkeypatch.setattr(supervision, 'launch', guard)
    report = corpus.prepare(tmp_path, dict(config(), max_source_bytes=1))
    assert report['status'] == 'PARTIAL' and report['counts']['BLOCKED'] == 5
    assert report['counts']['PROCESSED'] == 1
    assert 'increase reserved source allowance' in report['fetch_error']


def test_changed_raw_and_projection_fail_closed(tmp_path, upstream):
    corpus.prepare(tmp_path, config())
    row = results(tmp_path)['A000001']
    raw = tmp_path / 'corpus' / row['raw_path']
    raw.write_bytes(b'0 99\n')
    assert corpus.prepare(tmp_path, config())['status'] == 'PARTIAL'
    assert results(tmp_path)['A000001']['status'] == 'BLOCKED'
    # A previously published immutable manifest is historical, never a PASS for
    # the latest failed invocation; selected source files are independently hashed.
    projection = tmp_path / 'corpus' / results(tmp_path)['A000006']['projections']['terms']['path']
    projection.write_text('tampered')
    with pytest.raises(ValueError, match='projection changed'):
        corpus.prepare(tmp_path, config())


def test_zip_has_complete_member_failures_but_no_terms(tmp_path, upstream):
    from oeis_learn.pipeline_validation.corpus_workstation import package
    import zipfile
    corpus.prepare(tmp_path, config())
    package(tmp_path / 'corpus')
    with zipfile.ZipFile(tmp_path / 'corpus/share/corpus.zip') as archive:
        assert 'diagnostics/member-outcomes.jsonl' in archive.namelist()
        assert not any('inventory.sqlite' in n or '/objects/' in n for n in archive.namelist())
        assert '"values"' not in archive.read('diagnostics/member-outcomes.jsonl').decode()
        assert not json.loads(archive.read('manifest.json'))['omissions']


def test_interruption_commits_finished_entries_before_resume(tmp_path, upstream, monkeypatch):
    real, seen = corpus.inspect_bfile, []
    def interrupt(path, *args):
        seen.append(path)
        if len(seen) == 2:
            raise KeyboardInterrupt()
        return real(path, *args)
    monkeypatch.setattr(corpus, 'inspect_bfile', interrupt)
    with pytest.raises(KeyboardInterrupt):
        corpus.prepare(tmp_path, config())
    report = read(tmp_path / 'corpus/diagnostics/result.json')
    assert report['status'] == 'PARTIAL' and report['counts']['PROCESSED'] == 1
    done = seen[0]
    def resume(path, *args):
        assert path != done
        return real(path, *args)
    monkeypatch.setattr(corpus, 'inspect_bfile', resume)
    assert corpus.prepare(tmp_path, config())['status'] == 'COMPLETE'


def test_cache_change_and_projection_allowance_fail_closed(tmp_path, upstream):
    result = corpus.prepare(tmp_path, dict(config(), max_projection_bytes=1))
    assert result['status'] == 'PARTIAL' and result['counts']['eligible100'] == 0
    assert corpus.prepare(tmp_path, config())['status'] == 'COMPLETE'
    with gzip.open(tmp_path / 'stripped.gz', 'wt') as stream:
        stream.write('A000001 ,1,2,3,\n')
    with pytest.raises(ValueError, match='stripped cache changed'):
        corpus.prepare(tmp_path, config())


def test_host_failure_packages_first_error_and_never_requests_gpu(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import corpus_workstation as host
    from oeis_learn import container_runtime
    from oeis_learn.experiments import source_snapshot
    import zipfile
    write(tmp_path / 'config.json', config())
    events = []
    class Engine:
        executable = 'podman'
        def __init__(self, name):
            assert name == 'podman'
        def facts(self):
            return dict(test=True)
        def image(self, tag):
            return 'sha256:' + 'a' * 64
        def command(self, **kw):
            assert kw['gpu'] is False and kw['network'] is True
            assert all(str(m[0]) != str(tmp_path) for m in kw['mounts'])
            assert not any('sock' in str(m) for m in kw['mounts'])
            events.append('command')
            return ['podman', 'run']
        def call(self, argv, **kw):
            events.append('inspect')
            return SimpleNamespace(stdout=b'[{"State":{"OOMKilled":true}}]')
        def remove(self, name):
            events.append('remove')
    def launch(argv, output, **kw):
        if 'run' in argv:
            write(Path(output) / 'process.json', dict(returncode=1, failure='original first failure'))
            raise RuntimeError('original first failure')
    monkeypatch.setattr(container_runtime, 'Engine', Engine)
    monkeypatch.setattr(host, 'launch', launch)
    monkeypatch.setattr(source_snapshot, 'collect_snapshot', lambda _: {})
    args = SimpleNamespace(config='config.json', engine='podman', memory_gib=12, cpus=4)
    assert host.run(tmp_path, args) == 1
    assert events == ['command', 'inspect', 'remove']
    with zipfile.ZipFile(tmp_path / 'corpus/share/corpus.zip') as archive:
        name = next(n for n in archive.namelist() if n.endswith('/outcome.json'))
        assert 'original first failure' in json.loads(archive.read(name))['traceback']
        assert any(n.endswith('/container-inspect.json') for n in archive.namelist())


def test_host_import_and_corpus_config_need_only_system_python():
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(['python3', '-S', '-c',
        "import sys;sys.path.insert(0,'src');"
        "from oeis_learn.pipeline_validation.corpus_workstation import configuration;"
        "import json;configuration(json.load(open('configs/foundation/oeis_corpus.json')));"
        "assert not {'torch','numpy','duckdb','yaml'} & set(sys.modules)"],
        cwd=root, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr


def test_internal_format_line_boundaries_and_signed_values():
    assert corpus.main_terms('%S A000001 1,2\n%T A000001 3,4,\n%U A000001 5\n', 'A000001') == ['1','2','3','4','5']
    assert corpus.main_terms('%S A000001 1,2\n%V A000001 -1,2,\n%W A000001 -3\n', 'A000001') == ['-1','2','-3']


def test_inline_git_bfile_is_retained_and_processed(tmp_path, upstream):
    source = tmp_path / 'source'
    (source / '.gitattributes').write_text('files/** filter=lfs diff=lfs merge=lfs -text\nfiles/A000/b000006.txt -filter -diff -merge\n')
    bulk.git(source, 'add', '--renormalize', '.')
    bulk.git(source, 'add', '.gitattributes')
    bulk.git(source, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
             'commit', '-qm', 'inline b-file')
    bulk.git(source, 'push', '-q', 'origin', 'HEAD')
    assert corpus.prepare(tmp_path, config())['status'] == 'COMPLETE'
    row = results(tmp_path)['A000006']
    assert row['bfile']['storage'] == 'git'
    assert row['raw_path'].startswith('git-bfiles/') and row['eligible100']


def test_auxiliary_bfile_is_fetched_and_independently_hash_verified(tmp_path, upstream):
    source = tmp_path / 'source'
    (source / 'files/A000/b000001-extra.txt').write_bytes(b'0 42\n')
    bulk.git(source, 'add', '.')
    bulk.git(source, '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
             'commit', '-qm', 'auxiliary b-file')
    bulk.git(source, 'push', '-q', 'origin', 'HEAD')
    assert corpus.prepare(tmp_path, config())['status'] == 'COMPLETE'
    outcomes = [json.loads(line) for line in (tmp_path / 'corpus/diagnostics/object-outcomes.jsonl').read_text().splitlines()]
    assert len(outcomes) == 6
    extra = next(r for r in outcomes if 'extra' in r['path'])
    assert extra['status'] == 'VERIFIED'
    raw = corpus.object_path(tmp_path / 'corpus/oeisdata.git/lfs/objects', extra['oid'])
    raw.write_bytes(b'0 41\n')
    report = corpus.prepare(tmp_path, config())
    assert report['status'] == 'PARTIAL' and report['counts']['PROCESSED'] == 6
    assert 'objects unavailable or invalid' in report['fetch_error']
