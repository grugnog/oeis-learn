import pytest

from oeis_learn.pipeline_validation.progressive_velocity_worker import rehearse_batch


def profile():
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.models import resolve_profile

    base = resolve({"version": "bigint-bootstrap/v1"})
    return resolve_profile(base, shape="smaller_width", seed=29)


@pytest.mark.parametrize("batch_size", [16, 32])
def test_interrupted_backward_and_two_restores(tmp_path, batch_size):
    result = rehearse_batch(tmp_path, profile(), batch_size=batch_size, device="cpu")
    assert result["restored_weights"] == result["uninterrupted_weights"]
    assert result["restored_optimizer"] == result["uninterrupted_optimizer"]
    assert result["next_rows"] == result["uninterrupted_next_rows"]
    assert result["useful_updates"] == 2 and result["discarded_updates"] == 1
    assert (
        bool(result["matching_eligible"]) == bool(result["matching_passes"]) == (batch_size == 32)
    )
    if batch_size == 32:
        from oeis_learn.pipeline_validation.progressive_velocity_worker import disposable_batch
        from oeis_learn.generalization.progressive.training_rows import matched_batch

        package = disposable_batch(profile(), 32)
        micro = package["policy"]["microbatch_size"]
        batches = [matched_batch(package, start, 32) for start in (0, 32)]
        assert [sum(k in package["donors"] for k in batch) for batch in batches] == [17, 16]
        passes = sum(any(k in package["donors"] for k in batch[i:i+micro])
                     for batch in batches for i in range(0, 32, micro))
        assert result["matching_passes"] == passes and result["matching_eligible"] == 33
        assert result["measured_updates"] == 2 and result["first_measured_update"] == 1
        assert result["matching_loss"] == result["matching_loss_sum"] / 2
        assert result["matching_forward_us"] > 0
    assert result["charged_tokens"] == result["useful_tokens"] + result["discarded_tokens"]
    assert result["discarded_tokens"] > 0 and result["checkpoint_bytes"] > 0


@pytest.mark.gpu
def test_actual_hip_restore(tmp_path):
    import torch

    if not torch.version.hip or not torch.cuda.is_available():
        pytest.skip("requires the workstation HIP device")
    for size in (16, 32):
        assert rehearse_batch(tmp_path / str(size), profile(), batch_size=size, device="cuda")[
            "full_restore"
        ]


def test_registered_operation_recovers_receipt_before_commit_and_protects_holdback(tmp_path):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
    from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
    from oeis_learn.pipeline_validation.progressive_velocity_operations import run_operation

    grant = dict(
        identity=digest("operator"),
        resources=dict(dict.fromkeys(KEYS, 0), elapsed_us=900_000_000, attempts=3),
        holdback_us=300_000_000,
    )
    journal = BudgetJournal(tmp_path / "accounting", grant["identity"], grant["resources"])
    coordinates = dict(
        study=grant["identity"],
        stage="qualification",
        method="fixed",
        seed=0,
        look=0,
        panel="exposed",
        profile="common",
        input_mode="true",
        role="search",
        ordinal=0,
    )
    request = seal(dict(operation="search", candidate=digest("candidate")))
    issued = []

    def execute(out, budget_us):
        assert 0 < budget_us <= 600_000_000
        issued.append(out)
        atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
        if len(issued) == 1:
            raise ValueError("original fit stopped")
        atomic_write(
            out / "producer.json", seal(dict(request=request["identity"], result="checked"))
        )

    def verify(out):
        producer = load(out / "producer.json")
        assert producer["request"] == request["identity"]
        return producer

    args = dict(
        area=tmp_path,
        journal=journal,
        grant=grant,
        coordinates=coordinates,
        request=request,
        cost={"attempts": 1},
        execute=execute,
        verify=verify,
    )
    with pytest.raises(ValueError, match="original fit stopped"):
        run_operation(**args)
    with pytest.raises(ValueError, match="authorized dispatch attempts"):
        run_operation(**args, max_attempts=1)
    assert len(issued) == 1 and journal.snapshot()["consumed"]["attempts"] == 1
    good = run_operation(**args)
    assert good["attempt"] == 1 and len(issued) == 2
    block = issued[-1].parent
    (block / "receipt.json").unlink()
    args["verify"] = lambda out: pytest.fail("committed evidence must reopen without a new verifier")
    for _ in range(2):
        assert run_operation(**args) == good
    assert len(issued) == 2 and journal.snapshot()["consumed"]["attempts"] == 2
    assert load(block / "first-error.json")["reason"] == "original fit stopped"
    with pytest.raises(ValueError):
        run_operation(**dict(args, request=seal(dict(operation="different"))))
    other = dict(coordinates, ordinal=1)
    with pytest.raises(ValueError, match="holdback"):
        run_operation(**dict(args, coordinates=other, grant=dict(grant, holdback_us=900_000_000)))
    assert journal.snapshot()["remaining"]["elapsed_us"] >= grant["holdback_us"]


def test_donated_adamw_state_matches_one_real_next_update(tmp_path, monkeypatch):
    import os
    import shutil
    from oeis_learn.generalization.progressive import training_rows
    from oeis_learn.generalization.progressive.contracts import (
        atomic_write,
        digest,
        reference,
        seal,
    )
    from oeis_learn.pipeline_validation.progressive_pilot_worker import train
    from oeis_learn.pipeline_validation.progressive_velocity_worker import _disposable_rows
    from oeis_learn.pipeline_validation.progressive_velocity_continuation import compare_next_update

    rows, _ = _disposable_rows()
    stream = seal(
        dict(
            order=sorted(rows),
            registration=digest("original"),
            pair=0,
            seeds=dict(initialization=29),
            pairs=[dict(positive_row=sorted(rows)[0], label=None)],
        )
    )
    package = seal(
        dict(
            version="round028-pilot-training/v1",
            profile=profile(),
            microbatch=8,
            stream=stream,
            rows=[dict(row, draw_identity=key) for key, row in rows.items()],
        )
    )
    monkeypatch.setattr(training_rows, "GENERATOR_LOOKS", (0, 12, 4_000_000))
    path = tmp_path / "training.json"
    atomic_write(path, package)
    data = dict(
        training=reference(tmp_path, path),
        registration=stream["registration"],
        package=package["identity"],
        previous=None,
        arm="ordinary",
        target=0,
    )
    first = train(tmp_path, data, device="cpu")
    (tmp_path / "output").rename(tmp_path / "zero")
    shutil.copytree(
        tmp_path / "zero/checkpoint", tmp_path / "inputs/previous", copy_function=os.link
    )
    second = train(tmp_path, dict(data, previous=first["checkpoint"], target=12), device="cpu")
    out = compare_next_update(tmp_path / "output/checkpoint", package, second, device="cpu")
    assert out["parent_updates"] == 1 and out["reference"] == out["candidate"]
    assert out["candidate"]["updates"] == 2
    assert (
        out["charged_tokens"]
        == 2 * (out["candidate"]["tokens"] - second["milestone"]["charged"])
        > 0
    )
    assert out["candidate"]["next_rows"] == digest(training_rows.matched_batch(stream, 64, 32))
    with pytest.raises(ValueError):
        compare_next_update(
            tmp_path / "output/checkpoint",
            package,
            dict(second, milestone=first["milestone"]),
            device="cpu",
        )


def test_supervisor_loss_after_output_keeps_full_debit_and_reopens_without_redispatch(tmp_path):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
    from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
    from oeis_learn.pipeline_validation.progressive_velocity_operations import run_operation

    class SupervisorLoss(BaseException):
        pass

    grant = dict(
        identity=digest("operator"),
        holdback_us=5,
        resources=dict(dict.fromkeys(KEYS, 0), attempts=1, elapsed_us=20_000_005),
    )
    journal = BudgetJournal(tmp_path / "accounting", grant["identity"], grant["resources"])
    coordinates = dict(
        study=grant["identity"],
        stage="qualification",
        method="00",
        seed=42,
        look=0,
        panel="training",
        profile="ordinary",
        input_mode="true",
        role="train",
        ordinal=0,
    )
    produced = seal(dict(request=digest("frozen"), result=digest("producer bytes")))

    def interrupted(out, budget):
        assert budget == 20_000_000
        atomic_write(out / "producer.json", produced)
        atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
        raise SupervisorLoss()

    args = dict(
        area=tmp_path,
        journal=journal,
        grant=grant,
        coordinates=coordinates,
        request=seal(dict(action="train")),
        cost={"attempts": 1},
        execute=interrupted,
        verify=lambda out: load(out / "producer.json"),
    )
    with pytest.raises(SupervisorLoss):
        run_operation(**args)
    assert len(journal.snapshot()["pending"]) == 1
    first = run_operation(**args)
    assert first["evidence"] == produced and first["attempt"] == 0
    assert run_operation(**args) == first
    assert journal.snapshot()["remaining"]["elapsed_us"] == 5 and not journal.snapshot()["pending"]


def test_child_horizon_carries_complete_parent_and_charges_only_new_work(tmp_path):
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
    from oeis_learn.generalization.progressive.contracts import digest, seal
    from oeis_learn.generalization.progressive.velocity_trials import exposure
    from oeis_learn.pipeline_validation.progressive_velocity_child import child_package
    from oeis_learn.pipeline_validation.progressive_velocity_worker import (
        disposable_batch,
        train_to,
    )

    pkg = disposable_batch(profile(), 16)
    totals = [exposure(pkg, pkg["rows"], updates=i, batch=16)["useful_tokens"] for i in (1, 2)]
    path = tmp_path / "original"
    cut = train_to(path, pkg, target=totals[0] - 1, cap=totals[1], device="cpu")
    before = {p: p.stat().st_mtime_ns for p in path.rglob("*")}
    child = child_package(pkg, cut, parent_cap=totals[1], registration=digest("funded child"))
    added = totals[1] - totals[0]
    kwargs = dict(target=totals[1] - 1, cap=added, device="cpu", parent_directory=path)
    original = train_to(
        tmp_path / "reference", pkg, target=totals[1] - 1, cap=totals[1], device="cpu"
    )
    end = train_to(tmp_path / "child", child, **kwargs)
    assert (
        end["weights"] == original["weights"]
        and end["optimizer"] == original["optimizer"]
        and end["rng"] == original["rng"]
    )
    assert (
        end["useful_tokens"] == totals[1]
        and end["last_update_tokens"] == added
        and end["overshoot"] == 1
    )
    assert end["charged_tokens"] == totals[1] and end["incremental_charged_tokens"] == added
    assert end["next_rows"] == original["next_rows"] and end["scheduler"]["updates"] == 2
    assert train_to(tmp_path / "child", child, **dict(kwargs, parent_directory=None)) == end
    assert before == {p: p.stat().st_mtime_ns for p in path.rglob("*")}
    with pytest.raises(ValueError):
        train_to(
            tmp_path / "foreign", seal(dict(child, order=list(reversed(pkg["order"])))), **kwargs
        )
    changed = seal(
        dict(child, parent=seal(dict(child["parent"], charged_tokens=cut["charged_tokens"] + 1)))
    )
    with pytest.raises(ValueError):
        train_to(tmp_path / "unpaid", changed, **kwargs)
    with pytest.raises((ValueError, FileNotFoundError)):
        train_to(
            tmp_path / "wrong-tree", child, **dict(kwargs, parent_directory=tmp_path / "reference")
        )
    caps = dict(dict.fromkeys(KEYS, 0), tokens=added)
    snapshot = BudgetJournal.inspect_existing(
        tmp_path / "child/accounting", digest([child["identity"], added]), caps
    )
    assert snapshot["consumed"]["tokens"] == added and snapshot["remaining"]["tokens"] == 0


@pytest.mark.parametrize("batch_size", [16, 32])
def test_funded_child_rehearsal_preserves_failed_parent_charge(tmp_path, batch_size):
    from oeis_learn.pipeline_validation.progressive_velocity_worker import rehearsal_work

    result = rehearse_batch(tmp_path, profile(), batch_size=batch_size, device="cpu", children=True)
    child = result["child"]
    assert child["restored"] == child["reference"] and child["verified_replays"] == 2
    assert child["charged_tokens"] == result["charged_tokens"] + child["incremental_charged_tokens"]
    assert child["useful_tokens"] + result["discarded_tokens"] == child["charged_tokens"]
    assert (
        result["child_update_tokens"]
        == child["incremental_charged_tokens"]
        == {16: 209, 32: 400}[batch_size]
    )
    assert rehearsal_work(children=True)["tokens"] == 4182


def test_funded_verifier_stops_and_complete_nested_output_survives(tmp_path):
    import time
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
    from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
    from oeis_learn.pipeline_validation.progressive_velocity_operations import InsufficientAuthority, run_operation
    from oeis_learn.pipeline_validation import progressive_velocity_broad_host as broad

    g = dict(identity=digest("grant"), resources=dict(dict.fromkeys(KEYS, 0), elapsed_us=300_000),
             holdback_us=100_000)
    journal = BudgetJournal(tmp_path / "accounting", g["identity"], g["resources"])
    binding = seal(dict(origins={}))
    attempts = []

    def installed(out, reserve):
        attempts.append(out)
        folder = out / "custodian"
        atomic_write(folder / "custody.json", seal(dict(origins={}, disposition="requires_fresh_disjoint_start",
                     training=None, sources=[], counts=dict(candidates=2, excluded=1), ordinary_slots=2,
                     drawn=0, calibration={}, physical=digest("device"))))
        atomic_write(folder / "process.json", dict(outcome="COMPLETE", returncode=0, cleanup_failure=None))
        for p in (out, folder):
            atomic_write(p / "cleanup.json", dict(acknowledged=True, within_bound=True))

    def slow(out):
        time.sleep(1.5)
        (out / "late-write").touch()
        return broad.cpu_evidence(tmp_path, tmp_path, binding, out)

    args = dict(area=tmp_path, journal=journal, grant=g, coordinates=broad.coordinates(g, "prepare"),
                request=binding, cost={}, execute=installed, verify=slow, installed=broad.preparation_installed)
    start = time.monotonic()
    with pytest.raises(InsufficientAuthority, match="verification"):
        run_operation(**args)
    assert time.monotonic() - start < 1
    assert not (attempts[0] / "late-write").exists()
    assert journal.snapshot()["remaining"]["elapsed_us"] == g["holdback_us"]
    for _ in range(2):
        with pytest.raises(InsufficientAuthority, match="verification"):
            run_operation(**args)
    assert not (attempts[0] / "late-write").exists() and len(attempts) == 1
    assert "verification" in load(attempts[0].parent / "first-error.json")["reason"]


@pytest.mark.parametrize("role", ["prepare", "publish"])
def test_measured_host_window_survives_unproven_loss(tmp_path, role):
    import time
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
    from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
    from oeis_learn.pipeline_validation.progressive_velocity_operations import run_operation
    from oeis_learn.pipeline_validation.progressive_velocity_run import window

    total = 1680 * 60_000_000
    grant = dict(identity=digest("host loss"), resources=dict(dict.fromkeys(KEYS, 0), elapsed_us=total),
                 holdback_us=101_545_884)
    journal = BudgetJournal(tmp_path / "accounting", grant["identity"], grant["resources"])
    coords = dict(study=grant["identity"], stage="preparation" if role == "prepare" else "publication",
                  method="wave", seed=1, look=0, panel="development", profile="full", input_mode="true",
                  role=role, ordinal=0)
    timings = dict(preparation_us=3_000_000, publication_us=4_000_000, checkpoint_read_us=500_000)
    allowance = window(grant, {}, coords, {}, None, {}, host_plan=timings)
    assert allowance == (127_000_000 if role == "prepare" else 129_000_000)
    def work(out, limit):
        assert limit == allowance
        if out.name.endswith("0000"):
            raise RuntimeError("lost before termination")
        time.sleep(.04)
        atomic_write(out / "producer.json", seal(dict(verified=True)))
        atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
    args = dict(area=tmp_path, journal=journal, grant=grant, coordinates=coords, cost={},
                request=seal(dict(role=role)), execute=work, verify=lambda p: load(p / "producer.json"),
                maximum_us=allowance, max_attempts=2)
    with pytest.raises(RuntimeError, match="lost before termination"):
        run_operation(**args)
    assert journal.snapshot()["consumed"]["elapsed_us"] == allowance
    result = run_operation(**args)
    assert run_operation(**args) == result and result["attempt"] == 1
    used = journal.snapshot()["consumed"]["elapsed_us"]
    assert allowance + 40_000 <= used < 2 * allowance
    assert journal.snapshot()["remaining"]["elapsed_us"] == total - used > grant["holdback_us"]
