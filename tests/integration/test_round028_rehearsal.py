import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
from oeis_learn.pipeline_validation.progressive_round028_rehearsal import (
    measure_stage, sample_stream, training_forecast,
)


def test_rehearsal_samples_only_frozen_positive_and_negative_training_rows():
    from tests.unit.test_round028_training import training_rows
    from tests.unit.test_progressive_training_policy import profile
    rows = training_rows()
    probes = [seal(dict(r, draw_identity=digest(["held", i]))) for i, r in enumerate(rows)]
    base = seal(dict(shards=[rows+probes], layout={"probes": {str(i): {"id": r["draw_identity"]}
                     for i, r in enumerate(probes)}}, profile=profile(), update_policy={"microbatch_size": 7}))
    sample = sample_stream(base, digest("reg"), base["identity"])
    assert sample["support"]["eligible_rows"] == 2
    assert [len(b) for b in sample["batches"]] == [32, 32]
    assert all(s["positive"]["membership"] == s["negative"]["membership"] == "training"
               for b in sample["batches"] for s in b)
    assert all(s["positive"] in rows and s["negative"] in rows for b in sample["batches"] for s in b)
    with pytest.raises(ValueError, match="original diagnostic cut"):
        sample_stream(base, digest("reg"), digest("wrong"))


def test_physical_receipt_is_bound_to_original_outputs_and_reused_without_dispatch(tmp_path):
    image = digest("image")
    host = SimpleNamespace(engine=SimpleNamespace(image=lambda _: image))
    data = tmp_path / "training.json"
    atomic_write(data, seal(dict(version="fixture")))
    req = seal(dict(version="round028-probe-request/v1", data=load(data)["identity"],
                    stage="cpu", engine="podman", source=digest("code"), image=image, parent=None))
    calls = []
    def stage(host, source, image, output, argv, *, gpu, seconds, readonly_inputs, record_lease=False):
        assert not gpu and [(p.name, dst) for p, dst in readonly_inputs] == [
            ("training.json", "/work/inputs/training.json"), ("request.json", "/work/inputs/request.json")]
        calls.append(1)
        atomic_write(output / "producer.json", seal(dict(version="round028-probe-producer/v1", stage="cpu",
                     elapsed_us=25, request=req["identity"], details=dict(gradient_l2=.8, parameter_bytes=8,
                                                                     native_verified_terms=200))))
        atomic_write(output / "container-inspect.json", {})
    first, cost = measure_stage(host, tmp_path, tmp_path, data, "cpu", image, req, invoke=stage)
    later, reused = measure_stage(host, tmp_path, tmp_path, data, "cpu", image, req, invoke=stage)
    assert calls == [1] and first == later and reused["reused"] and not cost["reused"]
    assert reused["elapsed_us"] == cost["elapsed_us"] > 0
    producer = next(tmp_path.glob("stages/*/attempt-*/output/producer.json"))
    producer.write_text(json.dumps(seal(dict(first, elapsed_us=999))))
    with pytest.raises(ValueError):
        measure_stage(host, tmp_path, tmp_path, data, "cpu", image, req, invoke=stage)
    assert calls == [1]


def test_debits_and_reserved_half_shift_recompute_from_measured_paired_updates():
    arms = dict(ordinary=dict(update_us=[2000, 1500], charged_tokens=[100, 50]),
                first_divergence=dict(update_us=[2500, 2000], charged_tokens=[100, 50]))
    forecast = training_forecast(arms)
    assert forecast["pilot_training_us"] == dict(ordinary=120_000_000, first_divergence=160_000_000)
    assert forecast["evaluation_replication_reserve_us"] == 14_400_000_000
    assert forecast["projected_training_excess_us"] == 0
    arms["ordinary"]["update_us"][1] *= 150
    assert training_forecast(arms)["projected_training_excess_us"] > 0


def test_workstation_rehearsal_reuses_an_audit_and_checks_cpu_before_the_device(tmp_path, monkeypatch):
    from oeis_learn.pipeline_validation import progressive_round028_rehearsal as rehearsal
    from oeis_learn.pipeline_validation import progressive_round028, progressive_round028_support, progressive_handoff
    from oeis_learn.generalization.progressive.contracts import reference
    source = Path(__file__).resolve().parents[2]
    folder = tmp_path / "session"
    folder.mkdir()
    inputs = tmp_path / "small-original"
    inputs.write_bytes(b"123")
    cut = dict(inputs=[reference(tmp_path, inputs)], support={"input": digest("base")})
    monkeypatch.setattr(progressive_round028, "reopen", lambda *a, **kw: cut)
    donor = SimpleNamespace(policy={"campaign": digest("history")}, saved=lambda *a, **kw: {"payload":{"references":[]}})
    monkeypatch.setattr(rehearsal, "Donor", lambda *a: donor)
    monkeypatch.setattr(progressive_round028_support, "training_package", lambda *a: {})
    monkeypatch.setattr(rehearsal, "sample_stream", lambda *a: seal(dict(version="fixture", support={"eligible_rows": 3})))
    monkeypatch.setattr(rehearsal, "physical_scope", lambda host, phase: digest(phase))

    image = digest("immutable-image")
    builds = []
    calls = []
    def build(*a, **kw):
        builds.append("build")
        return image
    host = SimpleNamespace(args=SimpleNamespace(gpu_base_image=None, image=None), ensure=build,
                           engine=SimpleNamespace(name="podman", build=build, image=lambda _: image),
                           policy={"free_disk_floor_bytes": 1})
    def stage(host, source, image, output, argv, *, gpu, seconds, readonly_inputs, record_lease=False):
        request = load(readonly_inputs[1][0])
        calls.append(request["stage"])
        if gpu:
            assert calls == ["cpu", "hip"] and request["parent"]
            scope = dict(initial="same", row_schedule="same", charged_tokens=[125, 137], committed_updates=2,
                         update_us=[323, 411], actual_update=True, full_restore=True, save_us=20, restore_us=29,
                         replay_us=33, checkpoint_bytes=22, checkpoint_files=4)
            atomic_write(output / "gpu-lease.json", {"elapsed_us": 1})
            details = dict(peak_device_bytes=100, telemetry=dict(samples=0, readable_samples=0, covered_us=0,
                           busy_us=None, role_wall_us=20), arms={
                "ordinary": dict(scope, final="a", preference_rows=0),
                "first_divergence": dict(scope, final="b", preference_rows=1)})
        else:
            assert request["parent"] is None
            details = dict(parameter_bytes=64, gradient_l2=.14, gradient_loss=.83, gradient_probe_rows=1,
                           native_verified_terms=200)
        atomic_write(output / "producer.json", seal(dict(version="round028-probe-producer/v1", stage=request["stage"],
                     request=request["identity"], elapsed_us=12, details=details)))
        atomic_write(output / "container-inspect.json", {})
    monkeypatch.setattr(progressive_handoff, "stage", stage)
    store = SimpleNamespace(campaign={"identity": digest("reg")}, state={"publication": digest("report"), "audit": {}})
    rev = "1"*40
    args = (None, tmp_path, source, folder, store, rev, {"decision":{"action":"MATCHED_GENERATOR"}})
    first, rc = rehearsal.rehearse(*args, host_factory=lambda *a, **kw: host)
    assert rc == 0 and first["gpu_dispatches"] == 1 and first["scientific_status"] == "PILOT_UNMEASURED"
    second, rc = rehearsal.rehearse(*args, host_factory=lambda *a, **kw: host)
    assert rc == 0 and second["gpu_dispatches"] == 0 and len(builds) == 2 and calls == ["cpu", "hip"]
    assert second["forecast"] == first["forecast"] and second["hip"]["elapsed_us"] == first["hip"]["elapsed_us"]

def test_unavailable_gpu_samples_reduce_coverage_and_preserve_unknown_busy_time(tmp_path):
    from oeis_learn.pipeline_validation.progressive_generator_probe import usage_summary
    rows = [(0, "20"), (1, "80"), (2, None), (3, "0"), (8, "90")]
    path = tmp_path / "samples.jsonl"
    path.write_text("".join(json.dumps({"time": t, "gpu": {"gpu_busy_percent": val}}) + "\n" for t,val in rows))
    result = usage_summary(tmp_path, 9_000_000)
    assert result["samples"] == 5 and result["covered_us"] == 1_000_000 and result["busy_us"] == 500_000
    path.write_text(json.dumps({"time": 0, "gpu": None}) + "\n")
    assert usage_summary(tmp_path, 1_000_000)["busy_us"] is None


def test_collector_preserves_final_json_after_image_build_and_fails_closed(tmp_path):
    import runpy
    root = Path(__file__).resolve().parents[2]
    collector = runpy.run_path(str(root / "specs/015-learned-program-search/validation/collect_audit.py"))
    read_result = collector["read_result"]
    path = tmp_path / "rehearsal.json"
    value = dict(mechanical_status="COMPLETE", scientific_status="PILOT_UNMEASURED",
                 reason="MATCHED_REHEARSAL_MEASURED", cpu={"gradient_l2":1.25},
                 hip={"matched":True}, forecast={"combined_training_us":100})
    path.write_bytes(b"STEP 1/12: FROM base\n"+b"x"*4000000+b"\n"+json.dumps(value).encode()+b"\n")
    assert read_result(path, "rehearsal", 0) == value
    for raw in (b"", b"STEP 1/12\n", b'{"exit":0}\n', b'{"mechanical_status":"COMPLETE"}\n',
                json.dumps(value).encode()+b"\ntrailing garbage\n"):
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            read_result(path, "rehearsal", 0)
    path.write_text('log line\n{"mechanical_status":"BLOCKED","reason":"CPU_REHEARSAL_FAILED"}\n')
    assert read_result(path, "rehearsal", 2)["reason"] == "CPU_REHEARSAL_FAILED"
