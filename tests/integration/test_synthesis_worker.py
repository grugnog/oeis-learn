"""Transport/process failures cannot admit results or lose completed prefixes."""

import json
import struct
import sys
import time
from pathlib import Path
import pytest
from oeis_learn.synthesis.client import batches, frame, FrameReader
from oeis_learn.synthesis.worker_pool import WorkerPool


def test_framing_caps_before_body_and_candidate_byte_batching():
    reader = FrameReader()
    with pytest.raises(ValueError, match="frame"):
        reader.feed(struct.pack("<I", 4 * 1024**2 + 1))
    payload = {"request_id": "x"}
    packet = frame(payload)
    assert reader.__class__().feed(packet) == [payload]
    candidates = [{"candidate_id": str(i), "source": "x" * 65536} for i in range(70)]
    grouped = list(batches(candidates))
    assert sum(map(len, grouped)) == 70 and max(map(len, grouped)) <= 32
    assert all(len(frame({"candidates": b})) <= 4 * 1024**2 for b in grouped)


def fake_worker(tmp_path, behavior):
    script = tmp_path / "transport_fixture.py"
    script.write_text(
        """import sys,json,struct,time
n=struct.unpack('<I',sys.stdin.buffer.read(4))[0]
r=json.loads(sys.stdin.buffer.read(n))
def send(value):
 b=json.dumps(value).encode();sys.stdout.buffer.write(struct.pack('<I',len(b))+b);sys.stdout.buffer.flush()
base={k:r[k] for k in ('protocol','request_id','worker_generation','contract_sha256')}
"""
        + behavior
    )
    return [sys.executable, str(script)]


def request():
    return dict(
        protocol="synthesis-worker/v1",
        request_id="r",
        worker_generation=0,
        contract_sha256="sha256:" + "1" * 64,
        candidates=[{"candidate_id": "a"}, {"candidate_id": "b"}],
        remaining_budget=dict(wall_time_ms=1000, attempts=2),
    )


def test_partial_timeout_reaped_and_healthy_restart(tmp_path):
    command = fake_worker(
        tmp_path,
        "send(dict(base,results=[dict(candidate_id='a',outcome='unknown')],unstarted=['b'],complete=False))\ntime.sleep(30)\n",
    )
    with WorkerPool(command=command, contract_sha256=request()["contract_sha256"]) as pool:
        result = pool.call(request(), deadline=time.monotonic() + 0.3)
        assert result["results"][0]["candidate_id"] == "a"
        assert result["unstarted"] == ["b"] and result["outcome"] == "resource_exhausted"
        assert pool.process is None and pool.generation == 1
        pool.command = fake_worker(
            tmp_path,
            "send(dict(base,results=[dict(candidate_id=k['candidate_id'],outcome='unknown') for k in r['candidates']],unstarted=[],complete=True))\n",
        )
        result = pool.call(request(), deadline=time.monotonic() + 1)
        assert result["complete"] and len(result["results"]) == 2


def test_stale_generation_and_stderr_flood_are_rejected(tmp_path):
    for behavior in (
        "send(dict(base,worker_generation=99,results=[],unstarted=[],complete=True))\n",
        "sys.stderr.write('x'*70000);sys.stderr.flush();time.sleep(30)\n",
    ):
        command = fake_worker(tmp_path, behavior)
        with WorkerPool(command=command, contract_sha256=request()["contract_sha256"]) as pool:
            result = pool.call(request(), deadline=time.monotonic() + 1)
            assert not result["complete"] and pool.process is None
            assert result["outcome"] in {"verification_failure", "resource_exhausted"}


def test_cache_reservation_disk_failure_and_oversized_bypass(tmp_path, monkeypatch):
    from oeis_learn.synthesis.worker_pool import RoleCaches
    from oeis_learn.synthesis.service_policy import ServiceLimits
    from oeis_learn.synthesis.contracts import identity

    limits = ServiceLimits(
        result_cache_max_entries=1,
        result_cache_max_bytes=300,
        disk_cache_max_entries=1,
        disk_cache_max_bytes=300,
    )
    cache = RoleCaches(limits, tmp_path / "cache")
    try:
        a, b = identity("a"), identity("b")
        cache.put(a, {"ok": 1})
        assert cache.get(a) == {"ok": 1}
        cache.put(b, {"text": "x" * 1000})
        assert cache.get(a) == {"ok": 1}
        assert cache.metrics()["disk"]["bypasses"] == 1
        from oeis_learn.fresh import common

        monkeypatch.setattr(
            common, "write", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
        )
        cache.put(b, {"ok": 2})
        assert cache.disk.reserved_bytes == 0 and cache.result.reserved_bytes == 0
    finally:
        cache.close()


def test_pending_cancellation_releases_aggregate_reservations(tmp_path):
    import threading

    command = fake_worker(
        tmp_path, "time.sleep(.3)\nsend(dict(base,results=[],unstarted=['a','b'],complete=True))\n"
    )
    pool = WorkerPool(command=command, contract_sha256=request()["contract_sha256"])
    results = {}

    def invoke(key):
        r = dict(request(), request_id=key)
        results[key] = pool.call(r, deadline=time.monotonic() + 2)

    threads = [threading.Thread(target=invoke, args=(key,)) for key in ("first", "second")]
    try:
        threads[0].start()
        until = time.monotonic() + 1
        while not pool.capacity.active and time.monotonic() < until:
            time.sleep(0.005)
        threads[1].start()
        until = time.monotonic() + 1
        while "second" not in pool.capacity.pending and time.monotonic() < until:
            time.sleep(0.005)
        pool.cancel("second")
        for t in threads:
            t.join(timeout=3)
            assert not t.is_alive()
        assert results["second"]["unstarted"] == ["a", "b"]
        assert pool.capacity.pending_bytes == 0 and not pool.capacity.active
    finally:
        pool.close()


def test_real_packaged_native_fit_and_execute():
    import os
    from oeis_learn.synthesis.client import SynthesisClient
    from oeis_learn.synthesis.admission import VisiblePrompt

    binary = Path("/opt/oeis-build-info/synthesis-worker")
    if not binary.exists():
        if os.environ.get("OEIS_REQUIRE_BIGINT_BUILD") == "1":
            pytest.fail("packaged synthesis worker required")
        pytest.skip("requires qualification image")
    with SynthesisClient("sha256:" + "1" * 64) as client:
        assert client.execute("(program (return $index))", ["-7", "19"])["values"] == ["-7", "19"]
        source = "(program (return (add (mul (param.get p0) $index) (param.get p1))))"
        result = client.fit(source, VisiblePrompt(("-7", "19"), ("-19", "59")))
        assert result["outcome"] == "verified_solution" and result["admitted"]


@pytest.mark.parametrize(
    "behavior",
    [
        "send(dict(base,results=[{'candidate_id':'a'},{'candidate_id':'a'}],unstarted=['b'],complete=True))",
        "send(dict(base,results=[{}],unstarted=['a','b'],complete=True))",
        "send(dict(base,results=[],unstarted=['a','a','b'],complete=True))",
        "sys.stdout.buffer.write(struct.pack('<I',4194305));sys.stdout.buffer.flush();time.sleep(30)",
    ],
)
def test_malformed_response_invalidates_generation_and_releases_bytes(tmp_path, behavior):
    with WorkerPool(
        command=fake_worker(tmp_path, behavior), contract_sha256=request()["contract_sha256"]
    ) as pool:
        result = pool.call(request(), deadline=time.monotonic() + 1)
        assert result["outcome"] == "verification_failure" and not result["complete"]
        assert (
            pool.process is None and pool.capacity.pending_bytes == 0 and not pool.capacity.active
        )


def test_partial_input_write_cancellation_and_rss_limit_release(tmp_path):
    for rss in (1, 1 << 30):
        command = fake_worker(tmp_path, "time.sleep(30)")
        with WorkerPool(
            command=command, contract_sha256=request()["contract_sha256"], rss_bytes=rss
        ) as pool:
            # Pipe cannot hold this request. Deadline/kill must release the reservation.
            r = dict(request(), payload="x" * (3 << 20))
            result = pool.call(r, deadline=time.monotonic() + 0.15)
            assert not result["complete"] and result["unstarted"] == ["a", "b"]
            assert (
                pool.process is None
                and pool.capacity.pending_bytes == 0
                and not pool.capacity.active
            )


def test_disk_cache_rejects_bad_names_and_symlinks_without_following(tmp_path):
    from oeis_learn.synthesis.worker_pool import RoleCaches
    from oeis_learn.synthesis.service_policy import ServiceLimits
    import json

    outside = tmp_path / "outside"
    outside.write_text("SECRET")
    root = tmp_path / "cache"
    root.mkdir()
    (root / "bad.json").write_text(json.dumps(dict(key="not-a-hash", value={})))
    (root / ("a" * 64 + ".json")).symlink_to(outside)
    with RoleCaches(ServiceLimits(), root) as cache:
        assert not cache.disk.entries and not list(root.glob("*.json"))
    assert outside.read_text() == "SECRET"


def assert_wave_worker_activity(result,workers):
    """Assert actual factory-specific stats without changing production schemas."""
    assert result['workers']==workers
    metrics=result['metrics']
    if workers==1:
        # WorkerPool deliberately returns its retained serial implementation.
        assert 'workers' not in metrics and 'slots' not in metrics
        slots=[metrics]
    else:
        assert metrics['workers']==workers and len(metrics['slots'])==workers
        slots=metrics['slots']
    assert all(type(slot['batches']) is int and slot['batches']>0 for slot in slots)
    assert sum(slot['batches'] for slot in slots)==metrics['batches']
    assert metrics['batches']>=18  # Distinct prompts must exercise actual slots.


def native_affine_wave_corpus():
    from oeis_learn.pipeline_validation.identity import seal
    source='(program (return (add (mul (param.get p0) $index) (param.get p1))))'
    return seal(dict(version='pipeline-native-corpus/v1',exclusions=[],rows=[
        dict(id=f'affine-{i}',stratum='affine-fixture',source=source,
            indices=[str(20*i+j) for j in range(20)],values=[str(3*(20*i+j)+7) for j in range(20)]) for i in range(18)]))


@pytest.mark.parametrize('workers',[1,2,4])
def test_real_native_slot_waves_preserve_full_corpus_and_clocks(tmp_path,workers):
    import os
    from oeis_learn.pipeline_validation.role import native_trial
    from oeis_learn.pipeline_validation.native_capacity import semantics
    from oeis_learn.pipeline_validation.identity import seal
    binary=Path('/opt/oeis-build-info/synthesis-worker')
    if not binary.exists():
        if os.environ.get('OEIS_REQUIRE_BIGINT_BUILD')=='1':
            pytest.fail('packaged synthesis worker required')
        pytest.skip('requires qualification image')
    # Independent simple affine fixture isolates scheduling from hard solver cases.
    # The operator's full18-stratum screen remains independently required.
    corpus=native_affine_wave_corpus()
    result=native_trial(tmp_path,workers=workers,corpus=corpus,contract='sha256:'+'1'*64,cap=60,admission='slot-waves/v1')
    expected=semantics(result,corpus)
    assert len(expected)==18 and all(r['outcome']=='verified_solution' and r['admitted'] for r in expected)
    assert all(a['assignment']=={'p0':'3','p1':'7'} for r in expected for a in r['admitted'])
    assert_wave_worker_activity(result,workers)
    assert result['primary_seconds']>=result['seconds']
    assert (tmp_path/'measurement.json').is_file()


def test_real_native_coschedule_cycles_keep_unique_jobs_and_parent_clocks(tmp_path):
    import os
    from unittest.mock import patch
    from oeis_learn.pipeline_validation.role import native_trial
    from oeis_learn.pipeline_validation.native_capacity import semantics
    from oeis_learn.pipeline_validation.coschedule import native_run,Cancellation
    binary=Path('/opt/oeis-build-info/synthesis-worker')
    if not binary.exists():
        if os.environ.get('OEIS_REQUIRE_BIGINT_BUILD')=='1':pytest.fail('packaged synthesis worker required')
        pytest.skip('requires qualification image')
    corpus=native_affine_wave_corpus();contract='sha256:'+'1'*64
    baseline=native_trial(tmp_path/'reference',workers=4,corpus=corpus,contract=contract,cap=8,admission='slot-waves/v1')
    reference=dict(identity='fixture-native-reference',semantics=semantics(baseline,corpus))
    control=Cancellation()
    with patch('oeis_learn.pipeline_validation.role.native_corpus',return_value=corpus):
        result=native_run(None,tmp_path/'cycles',dict(source='fixture',contract=contract),reference,time.monotonic()+120,control)
    control.cancel()
    assert result['status']=='COMPLETE' and result['cycles']==8 and result['rows']==144
    jobs=set()
    for i in range(8):
        receipt=json.loads((tmp_path/f'cycles/cycle-{i:02d}/receipt.json').read_text())
        trial=receipt['observations']['result'];assert receipt['state']=='PASS'
        assert_wave_worker_activity(trial,4)
        assert trial['primary_seconds']<=8 and not jobs.intersection(trial['jobs'])
        jobs.update(trial['jobs'])
    assert len(jobs)==144
