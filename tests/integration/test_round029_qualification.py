"""Controller-only fixtures. Hardware measurements come exclusively from the installed worker."""

from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
from oeis_learn.pipeline_validation import progressive_handoff, progressive_round028_rehearsal
from oeis_learn.pipeline_validation import progressive_velocity_qualification as controller
from oeis_learn.pipeline_validation.progressive_round029 import inspect_status


def test_original_grant_retries_after_capacity_block_and_credits_one_receipt(tmp_path, monkeypatch):
    revision = "1" * 40
    root = tmp_path / "donor"
    root.mkdir()
    folder = root / "run-state/discovery/016-research-round-029"
    folder.mkdir(parents=True)
    grant = load(controller.authorize_learner_check(root, 10, revision))
    assert controller.authorize_learner_check(root, 10, revision).is_file()
    low = {"free_bytes": 1 << 40, "free_inodes": 1}
    monkeypatch.setattr(progressive_round028_rehearsal, "space", lambda root: dict(low))
    monkeypatch.setattr(
        progressive_round028_rehearsal, "physical_scope", lambda *a, **k: digest("fixture physical")
    )
    monkeypatch.setattr(controller, "_original_profile", lambda root: {"fixture": True})
    from oeis_learn.experiments import source_snapshot

    monkeypatch.setattr(source_snapshot, "collect_snapshot", lambda source: {"fixture": True})
    image = digest("fixture image")
    host = SimpleNamespace(
        policy={"free_disk_floor_bytes": 0}, engine=SimpleNamespace(image=lambda ref: image)
    )
    args = SimpleNamespace(image=image, gpu_base_image=None)
    issued = []

    def stage(host, source, image, out, argv, *, gpu, record_lease, seconds, readonly_inputs):
        issued.append(out)
        assert gpu and record_lease and seconds > 0
        path, dest = readonly_inputs[0]
        req = load(path)
        from oeis_learn.pipeline_validation.progressive_velocity_worker import rehearsal_counts

        counts = rehearsal_counts()
        batches = {}
        for size in req["batch_sizes"]:
            batches[str(size)] = dict(
                batch=size,
                full_restore=True,
                useful_updates=2,
                discarded_updates=1,
                **counts[str(size)],
                restored_weights=digest("fixture weights"),
                uninterrupted_weights=digest("fixture weights"),
                next_rows=digest("fixture next"),
                uninterrupted_next_rows=digest("fixture next"),
                restored_optimizer=digest("fixture moments"),
                uninterrupted_optimizer=digest("fixture moments"),
                scheduler={"updates": 2, "tokens": counts[str(size)]["charged_tokens"]},
                interrupted_peak_device_bytes=1,
                checkpoint_bytes=1,
                checkpoint_inodes=1,
                failed_us=1,
                matching_eligible=int(size == 32),
                matching_passes=int(size == 32),
            )
        atomic_write(
            out / "producer.json",
            seal(
                dict(
                    version="round029-device-result/v1",
                    device="hip",
                    request=req["identity"],
                    batches=batches,
                    literal_probes={"fixture": {"status": "fixture"}},
                    peak_device_bytes=1,
                    elapsed_us=2,
                    telemetry={},
                )
            ),
        )
        atomic_write(
            out / "container-inspect.json",
            [
                dict(
                    Image=image,
                    HostConfig=dict(NetworkMode="none", ReadonlyRootfs=True, Privileged=False),
                    Mounts=[
                        dict(Type="bind", Destination=dest, Source=str(path), RW=False),
                        dict(Type="bind", Destination="/work/output", Source=str(out), RW=True),
                    ],
                )
            ],
        )
        atomic_write(
            out / "process.json", dict(outcome="COMPLETE", returncode=0, cleanup_failure=None)
        )
        atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
        atomic_write(out / "gpu-lease.json", {"elapsed_us": 10})

    monkeypatch.setattr(progressive_handoff, "stage", stage)

    def run():
        return controller.qualify(args, root, folder, root, revision, grant, lambda *a, **k: host)

    blocked, code = run()
    assert code == 2 and blocked["first_error"]["deficit_inodes"] > 0
    assert blocked["charged_tokens"] == 0 and not issued
    assert inspect_status(folder)["progress"]["next_action"] == "inspect_capacity"
    low["free_inodes"] = 10000
    accepted, code = run()
    assert code == 0 and len(issued) == 1
    assert accepted["charged_tokens"] == grant["resources"]["tokens"] > 0
    assert accepted["first_error"] == blocked["first_error"]
    receipt = root / accepted["receipt_path"]
    assert receipt.exists() and accepted["receipt"]["lease_us"] == 10
    receipt.unlink()  # clean supervisor termination preceded final host publication
    for _ in range(2):
        repeated, code = run()
        assert code == 0 and repeated == accepted and len(issued) == 1
    assert inspect_status(folder)["last_committed_ref"] == accepted["receipt"]["identity"]
    mounts = load(issued[0] / "container-inspect.json")
    mounts[0]["Mounts"][0]["RW"] = True
    (issued[0] / "container-inspect.json").unlink()
    atomic_write(issued[0] / "container-inspect.json", mounts)
    with pytest.raises(ValueError, match="MOUNTS"):
        run()
