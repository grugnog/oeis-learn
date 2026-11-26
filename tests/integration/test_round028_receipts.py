"""Recover original sealed role boundaries without an engine or a learner on the reader."""
import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import pytest

from oeis_learn.experiments.source_snapshot import collect_snapshot
from oeis_learn.fresh.bigint_config import resolve
from oeis_learn.fresh.bigint_state import GeneralizationCheckpointStore
from oeis_learn.generalization.budgets import Ledger
from oeis_learn.generalization.progressive.campaign import diagnostic_registration
from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, reference, seal
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.models import resolve_profile
from oeis_learn.generalization.progressive.program_search_rehearsal_report import (
    ARMS, PHASES, encode, publish, verify_transport,
)
from oeis_learn.generalization.progressive.program_search_report import public_audit, publish_archive
from oeis_learn.generalization.progressive.program_search_routing import project, route
from oeis_learn.generalization.state import STATE_FIELDS
from oeis_learn.pipeline_validation.progressive_round028 import ARCHIVE, SESSION, seal_audit
from oeis_learn.pipeline_validation.progressive_round028_audit import Donor, METHODS, look_name, summarize
from oeis_learn.pipeline_validation.progressive_round028_receipts import ReceiptGap, recover, stage_chain, verify_sample_origin
from oeis_learn.pipeline_validation.progressive_round028_rehearsal import measure_stage, sample_stream
from oeis_learn.pipeline_validation.progressive_training import resolve_update_policy
from tests.integration.test_round028_audit import roster
from tests.unit.test_round028_training import training_rows

SOURCE = Path(__file__).resolve().parents[2]
REVISION = "b" * 40


def historical_inputs(workspace, folder, store, base):
    cases = [dict(row, observations={}, diagnosis=dict(feasible="P", fault=False, unfinished_fit=False,
                  band_witness=True, exact_witness=True, search_closed=True, replay=None)) for row in roster()]
    reg = store.campaign
    names = [look_name(*look) for look in reg["policy"]["historical_looks"]]
    names += [k for m in ("retained_D", "random") for k in (m, *(m + "/" + method for method in METHODS))]
    counts, pairs = summarize(cases, names)
    votes = [dict(r["diagnosis"], task=r["task"], stratum=r["stratum"]) for r in cases
             if r["stratum"] in reg["policy"]["roster"]["primary"]]
    projection = project(votes, training_support=True, replay_budget_us=None)
    checkpoints = [dict(arm=a, seed=i, target=25000000, disposition="unavailable", reason="missing_checkpoint_index")
                   for a in ("refreshed", "static") for i in (1, 2)]
    base_path = workspace / "base.json"
    atomic_write(base_path, base)
    channel = seal(dict(version="discovery-channel/v1", role="learner", kind="training_request",
                        data=dict(membership="training", artifact=reference(workspace, base_path))))
    atomic_write(workspace / "channel.json", channel)
    channel_ref = reference(workspace, workspace / "channel.json")
    data = dict(cases=cases, methods=counts, pairs=pairs, checkpoints=checkpoints, projection=projection,
                decision=route(projection), followup_training={}, support=dict(status="measured", eligible_rows=2,
                eligible_families=2, positions=1, input=base["identity"]), inputs=[channel_ref],
                original_trials={"trial-05-1M": {"costs": dict.fromkeys(PHASES, 3)}})
    cut = seal_audit(folder, reg, data, 0)
    public = public_audit(reg, data, elapsed_us=20, private_identity=load(folder / cut["path"])["identity"])
    receipt = publish_archive(folder / ARCHIVE, public)
    store.append(dict(store.state, registration=reg["identity"], audit=cut, publication=receipt["report"]))
    return {"payload": {"references": [dict(name="private/channels/screening-training", role="learner",
                                           kind="prepared", artifact=channel_ref)]}}


def saved_role_boundaries(area, data, src):
    row_ids = [[s["positive"]["draw_identity"] for s in b] for b in data["batches"]]
    charges = [sum(len(s["positive"]["tokens"]) for s in b) for b in data["batches"]]
    observed = sum(s["label"] is not None for s in data["batches"][0])

    def stage(host, root, image, output, argv, *, gpu, seconds, record_lease, readonly_inputs):
        req = load(readonly_inputs[1][0])
        mounts = [(source, dest, False) for source, dest in readonly_inputs] + [(output, "/work/output", True)]
        atomic_write(output / "container-inspect.json", [dict(Image=image[7:], HostConfig=dict(
            NetworkMode="none", ReadonlyRootfs=True, Privileged=False), Mounts=[dict(
                Type="bind", Source=str(source), Destination=dest, RW=rw) for source, dest, rw in mounts])])
        if gpu:
            arms = {}
            for arm in ARMS:
                policy = resolve_update_policy(data["profile"], objective=arm, effective_batch=32,
                           microbatch_size=data["microbatch"], schedule_horizon_tokens=4_000_000, weight_decay=0)
                checkpoint = GeneralizationCheckpointStore(output / arm, digest([policy["identity"], row_ids]), sum(charges))
                states = {k: k for k in STATE_FIELDS - {"rng", "data"}}
                states.update(data=dict(cursor=32, pending_batch=None), rng={k:k for k in ("python","numpy","torch","device")})
                ledger = Ledger(sum(charges)).reserve("0", charges[0]).commit("0")
                checkpoint.save(states, ledger, encode=lambda s: s.encode())
                files = [p for p in (output / arm).rglob("*") if p.is_file()]
                arms[arm] = dict(initial="shared", final=arm, row_schedule=digest(row_ids), charged_tokens=charges,
                                 committed_updates=2, update_us=[400, 390], actual_update=True, full_restore=True,
                                 save_us=91, restore_us=49, replay_us=53, checkpoint_files=len(files),
                                 checkpoint_bytes=sum(p.stat().st_size for p in files),
                                 preference_rows=observed if arm == "first_divergence" else 0)
            details = dict(arms=arms, hardware={"private": "private-hardware-fixture"}, peak_device_bytes=180,
                           telemetry=dict(samples=3, readable_samples=2, covered_us=1, busy_us=1, role_wall_us=20))
            atomic_write(output / "gpu-lease.json", {"elapsed_us": 1})
        else:
            details = dict(gradient_l2=1.5, gradient_loss=.8, gradient_probe_rows=1, parameter_bytes=320,
                           native_verified_terms=200)
        atomic_write(output / "producer.json", seal(dict(version="round028-probe-producer/v1", stage=req["stage"],
                     request=req["identity"], elapsed_us=16, details=details)))

    parent = None
    host = SimpleNamespace(engine=SimpleNamespace(image=lambda value: value))
    for phase in ("cpu", "hip"):
        image = digest(phase)
        opts = dict(revision=REVISION, phase=phase, engine="podman", base=None, override=None)
        atomic_write(area / ("image-" + digest(opts)[7:] + ".json"), seal(dict(options=opts, image=image, elapsed_us=5)))
        request = seal(dict(version="round028-probe-request/v1", source=src, data=data["identity"],
                            stage=phase, image=image, engine="podman", physical=digest("boot"), parent=parent))
        producer, _ = measure_stage(host, SOURCE, area, area / "training.json", phase, image, request, invoke=stage)
        parent = producer["identity"]


@pytest.fixture
def original(tmp_path, monkeypatch):
    folder = tmp_path / SESSION
    folder.mkdir(parents=True)
    reg = diagnostic_registration("a" * 40)
    store = RunStore(folder, reg, diagnostic=True)
    rows = training_rows()
    probes = [seal(dict(r, draw_identity=digest(["probe", i]))) for i, r in enumerate(rows)]
    base = seal(dict(version="progressive-training-input/v1", scope="NATIVE_VERIFIED", shards=[probes + rows],
                     layout={"probes": {str(i):{"id": p["draw_identity"]} for i,p in enumerate(probes)}},
                     profile=resolve_profile(resolve({"version":"bigint-bootstrap/v1"})),
                     update_policy={"microbatch_size": 8}))
    original_campaign = historical_inputs(tmp_path, folder, store, base)
    monkeypatch.setattr(Donor, "saved", lambda *a, **kw: original_campaign)
    data = sample_stream(base, reg["identity"], base["identity"])
    area = folder / "private/generator-rehearsal" / REVISION
    area.mkdir(parents=True)
    atomic_write(area / "training.json", data)
    src = digest(collect_snapshot(SOURCE))
    saved_role_boundaries(area, data, src)
    return SimpleNamespace(folder=folder, workspace=tmp_path, area=area, data=data, src=src, store=store, base=base)


def test_recover_and_verify_standalone_with_the_original_checkpoint_writer(original):
    o = original
    def file_state():
        return {p:p.stat().st_mtime_ns for p in o.folder.rglob("*") if p.is_file()}
    before = file_state()
    value = recover(o.workspace, o.folder, SOURCE, REVISION, o.store)
    # Exercise the private reader with the same dependency boundary as the workstation host.
    private_reader = """
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from oeis_learn.generalization.progressive.contracts import load
from oeis_learn.pipeline_validation.progressive_round028_receipts import stage_chain, verify_sample_origin
area = Path(sys.argv[2])
data = load(area / 'training.json')
verify_sample_origin(load(sys.argv[3]), data, data['registration'])
stages, closures, space = stage_chain(area, revision=sys.argv[4], source=sys.argv[5], training=data, progress={})
assert 'torch' not in sys.modules
print(json.dumps(closures))
"""
    cmd = [sys.executable, "-I", "-S", "-B", "-c", private_reader, str(SOURCE / "src"), str(o.area),
           str(o.workspace / "base.json"), REVISION, o.src]
    assert json.loads(subprocess.check_output(cmd, cwd=o.workspace)) == value["checkpoint_closures"]
    assert before == file_state()
    assert value["checkpoint_closures"] == dict.fromkeys(ARMS, "VERIFIED")
    assert value["cpu"]["gradient_l2"] == 1.5 and value["forecast"]["combined_training_us"] > 0
    audit = o.folder / ARCHIVE
    archive = o.workspace / "return.zip"
    receipt = publish(archive, value, audit)
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == ["rehearsal.json", "round028.zip"] and z.read("round028.zip") == audit.read_bytes()
        assert b"private-hardware-fixture" not in z.read("rehearsal.json")
        assert b"draw_identity" not in z.read("rehearsal.json")
    cmd = [sys.executable, "-E", "-s", "-S", "-B", str(SOURCE / "scripts/foundation_run.py"),
           "--engine", "podman", "discovery", "round028", "--action", "verify", "--archive", str(archive)]
    assert json.loads(subprocess.check_output(cmd, cwd=o.workspace)) == receipt
    assert receipt["scope"] == "archive_only_rehearsal" and receipt["gpu_dispatches"] == 0

    for field, nested in (("forecast","combined_training_us"), ("historical_trial05_us","training")):
        changed = copy.deepcopy(value)
        changed[field][nested] += 1
        bad = o.workspace / ("bad-" + field + ".zip")
        with zipfile.ZipFile(bad, "w") as z:
            z.writestr("rehearsal.json", encode(seal(changed)))
            z.write(audit, "round028.zip")
        with pytest.raises(ValueError):
            verify_transport(bad)
        failed = o.workspace / ("invalid-" + field + ".zip")
        with pytest.raises(ValueError):
            publish(failed, seal(changed), audit)
        assert not failed.exists()


def test_private_recovery_distinguishes_retired_and_corrupt_checkpoints_and_missing_hip(original):
    o = original
    def stages(progress=None):
        return stage_chain(o.area, revision=REVISION, source=o.src, training=o.data, progress=progress or {})
    verified, _, _ = stages()
    reported = copy.deepcopy(verified)
    reported["hip"]["arms"]["ordinary"]["update_us"][0] += 1
    with pytest.raises(ReceiptGap, match="MISSING_OR_AMBIGUOUS_HIP_BOUNDARY"):
        stages(reported)
    pointer = next(o.area.glob("stages/*/attempt-*/output/ordinary/generalization-latest.json"))
    blob = next((pointer.parent).glob("generalization-*/optimizer.pt"))
    blob.write_bytes(bytes(blob.stat().st_size))
    with pytest.raises(ValueError, match="hash/size mismatch"):
        stages()
    blob.unlink()
    assert stages()[1] == dict(ordinary="UNAVAILABLE", first_divergence="VERIFIED")
    hip_receipt = pointer.parents[3] / "receipt.json"
    hip_receipt.unlink()
    with pytest.raises(ReceiptGap, match="MISSING_OR_AMBIGUOUS_HIP_BOUNDARY") as gap:
        stages()
    assert gap.value.observed["cpu"] == verified["cpu"]


def test_sample_cannot_use_held_probes_or_an_ambiguous_label(original):
    o = original
    for field in ("positive", "negative", "label"):
        tampered = copy.deepcopy(o.data)
        sample = tampered["batches"][0][0]
        if field == "label":
            sample[field]["position"] += 1
        else:
            sample[field] = o.base["shards"][0][0]
        with pytest.raises(ReceiptGap):
            verify_sample_origin(o.base, tampered, o.store.campaign["identity"])


def test_original_console_and_progress_are_retained_even_if_private_recovery_fails(tmp_path, monkeypatch):
    import runpy
    collector = runpy.run_path(str(SOURCE / "specs/015-learned-program-search/validation/collect_audit.py"))
    tools = tmp_path / "run-state/discovery-tools"
    previous = tools / "round028-rehearsal-test"
    previous.mkdir(parents=True)
    atomic_write(previous / "summary.json", dict(candidate=REVISION, steps={"rehearsal":{"exit":0}}))
    arms = {a:dict(charged_tokens=[100, 110], update_us=[25, 28]) for a in ARMS}
    old = dict(candidate=REVISION, mechanical_status="COMPLETE", reason="MATCHED_REHEARSAL_MEASURED",
               cpu={"gradient_l2":.1}, hip={"arms":arms}, forecast={"source":"original"}, private="private-fixture")
    (previous / "rehearsal.json").write_text("STEP 1/4 build\n" + json.dumps(old) + "\n")
    area = tmp_path / SESSION / "private/generator-rehearsal" / REVISION
    area.mkdir(parents=True)
    atomic_write(area / "progress.json", seal(dict(old, phase="complete")))
    called = []
    def missing(cmd, *, stdout, **kw):
        called.append(cmd[cmd.index("--action")+1])
        stdout.write(b'{"mechanical_status":"BLOCKED","reason":"MISSING_OR_AMBIGUOUS_HIP_BOUNDARY"}\n')
        return SimpleNamespace(returncode=2)
    monkeypatch.setattr(subprocess, "run", missing)
    assert collector["collect"](tmp_path, "c"*40, "collect-receipts", REVISION) == 2
    assert called == ["recover-rehearsal"]
    with zipfile.ZipFile(next(tools.glob("round028-receipts-*/round028-receipt-diagnostics.zip"))) as archive:
        assert archive.namelist() == ["summary.json"]
        summary = json.loads(archive.read("summary.json"))
        assert b"private-fixture" not in archive.read("summary.json")
    assert summary["gpu_dispatches"] == 0 and summary["steps"]["recover"]["reason"] == "MISSING_OR_AMBIGUOUS_HIP_BOUNDARY"
    assert len(summary["prior_observations"]) == 2
    assert all(row["reported_updates"] == arms and row["reported_forecast"]["combined_training_us"] > 0
               for row in summary["prior_observations"])


def test_workstation_collector_releases_only_a_standalone_verified_archive(original, monkeypatch):
    import runpy
    o = original
    collect = runpy.run_path(str(SOURCE / "specs/015-learned-program-search/validation/collect_audit.py"))["collect"]
    tools = o.workspace / "run-state/discovery-tools"
    tools.mkdir()
    value = recover(o.workspace, o.folder, SOURCE, REVISION, o.store)
    run = subprocess.run
    calls = []
    def observed(cmd, **kwargs):
        action = cmd[cmd.index("--action")+1]
        calls.append(action)
        if action == "recover-rehearsal":
            assert cmd[cmd.index("--producer-revision")+1] == REVISION
            kwargs["stdout"].write(encode(dict(mechanical_status="COMPLETE", reason="SAVED_REHEARSAL_VERIFIED",
                                               evidence=value)) + b"\n")
            return SimpleNamespace(returncode=0)
        return run(cmd, **kwargs)
    monkeypatch.setattr(subprocess, "run", observed)
    assert collect(o.workspace, "c"*40, "collect-receipts", REVISION) == 0
    assert calls == ["recover-rehearsal", "verify"]
    copy_path = next(tools.glob("round028-receipts-*/round028-rehearsal.zip"))
    assert verify_transport(copy_path)["evidence"] == value["identity"]
