import pytest

from oeis_learn.generalization.progressive.contracts import digest, seal


def test_cpu_training_role_copies_only_verified_parent_and_resumes_full_optimizer(tmp_path):
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.generalization.progressive.models import construct
    from oeis_learn.pipeline_validation import progressive_velocity_checkpoint as cp
    from oeis_learn.pipeline_validation import progressive_velocity_wave as wave
    from oeis_learn.pipeline_validation import progressive_velocity_wave_host as host
    from oeis_learn.pipeline_validation import progressive_velocity_wave_role as role
    from oeis_learn.pipeline_validation.progressive_velocity_worker import disposable_batch
    from tests.integration.test_round029_resume import profile
    import shutil

    recipe = profile()
    batch = disposable_batch(recipe, 16)
    enc, dec, _ = construct(recipe, device="cpu")
    ordinary = dict(
        profile=recipe,
        initial_weights=model_snapshot(enc, dec),
        rows=batch["rows"],
        order=batch["order"],
    )
    learner = cp.package(ordinary, "04", 8, digest("training-origin"))
    targets = wave.targets(learner, [0, 1, 1000])
    cap = 2 * targets[1000]["useful_tokens"]
    base = dict(
        learner=learner["identity"],
        method="04",
        seed=recipe["model"]["seed"],
        registration=digest("scientific registration"),
        cap=cap,
        role="train",
    )
    previous = None
    for n in (0, 1, 1000):
        work = tmp_path / str(n)
        files = []
        if previous:
            old, cut = previous
            files = host.next_files(old, cut["checkpoint"])
            for f in files:
                dst = work / "inputs/parent" / f["path"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(old / f["path"], dst)
        request = dict(base, target=n, parent_files=files)
        producer = role.perform(work, request, dict(package=learner), device="cpu")
        checked = host.check_cut(work / "output", request, learner, producer, targets[n])
        cut = checked["cut"]
        assert cut["model"]["registration"] == base["registration"]
        assert cut["charged_tokens"] == targets[n]["useful_tokens"]
        assert cut["weights"] == producer["cut"]["weights"]
        previous = work / "output/learner", cut
        if n == 1:
            interrupted = tmp_path / "recovery/attempt-0000"
            shutil.copytree(previous[0], interrupted / "learner")
            first = cut
    # The next update was issued, but its tensors never became a saved cut.
    from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
    from oeis_learn.generalization.progressive.contracts import atomic_write

    recovery_target = first["useful_tokens"] + 1
    crossing = wave.targets(learner, [recovery_target])[recovery_target]
    charge = crossing["last_update_tokens"]
    journal = BudgetJournal(
        interrupted / "learner/accounting",
        digest([learner["identity"], cap]),
        dict(dict.fromkeys(KEYS, 0), tokens=cap),
    )
    journal.reserve(
        "learner",
        "update",
        first["updates"],
        dict(dict.fromkeys(KEYS, 0), tokens=charge),
        boot="fixture",
        monotonic_us=1,
    )
    atomic_write(interrupted / "cleanup.json", dict(acknowledged=True, within_bound=True))
    retry = interrupted.parent / "attempt-0001"
    retry.mkdir()
    req = seal(dict(base, target=recovery_target, parent_files=[]))
    mounts = host.restart_mounts(retry, req, [], readonly=False)
    assert mounts == host.restart_mounts(retry, req, [], readonly=True)
    staged = tmp_path / "staged-recovery"
    for src, mount in mounts:
        path = staged / mount.removeprefix("/work/")
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, path)
    produced = role.perform(staged, req, dict(package=learner), device="cpu")
    recovered = host.check_cut(staged / "output", req, learner, produced, crossing)["cut"]
    assert recovered["discarded_tokens"] == charge
    assert recovered["charged_tokens"] == recovered["useful_tokens"] + charge
    ref = previous[0] / "checkpoint" / checked["weight"]["path"]
    with ref.open("ab") as f:
        f.write(b"bad")
    with pytest.raises(ValueError):
        host.check_cut(work / "output", request, learner, producer, targets[1000])


def test_selected_control_role_cannot_read_truth_until_choices_are_sealed(tmp_path):
    import shutil
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.contracts import reference, read_bound_record
    from oeis_learn.generalization.progressive.graph import velocity_look
    from oeis_learn.generalization.progressive.velocity_metrics import measured_look
    from oeis_learn.pipeline_validation.progressive_velocity_wave_role import perform
    from tests.integration.test_round029_isolated_control_prices import SelectedClient, panel

    members, prompts, truth = panel()
    look = velocity_look(
        registration=digest("fixture"),
        method="polynomial",
        seed=29,
        look=4_000_000,
        profile="full",
        development=members,
        retention=[],
    )
    search = tmp_path / "visible"
    (search / "output").mkdir(parents=True)
    req = dict(role="search", model=None, weight=None, look=look, seed=29, seconds=120)
    data = dict(prompts=prompts, recipe=resolve({"version": "bigint-bootstrap/v1"}))
    client = SelectedClient()
    with pytest.raises(ValueError, match="projection"):
        perform(search, req, dict(data, truth=truth), device="cpu", client=client)
    generated = perform(search, req, data, device="cpu", client=client)
    score = tmp_path / "scorer"
    dest = score / "inputs/candidates.json"
    dest.parent.mkdir(parents=True)
    shutil.copyfile(search / "output" / generated["artifact"]["path"], dest)
    score_req = dict(
        role="score", probe=generated["probe"], candidates=reference(score, dest), seconds=120
    )
    scored = perform(score, score_req, dict(truth=truth), device="cpu", client=client)
    result = read_bound_record(score / "output", scored["artifact"], version="discovery-scoring/v1")
    measured = measured_look(generated["probe"], result)
    assert result["candidates"] == generated["artifact"]["sha256"]
    assert len(measured["scores"]["outcomes"]) == len(members)


def test_comparative_capacity_credits_only_registered_checkpoint_files(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from oeis_learn.pipeline_validation import progressive_velocity_wave_host as host
    f = tmp_path / "stale.log"
    f.write_bytes(b"x" * 8192)
    device = tmp_path.stat().st_dev
    row = dict(device=device, required_bytes=12288, required_inodes=3,
               reserved_bytes=4096, reserved_inodes=1)
    accepted = dict(capacity=[row])
    fs = SimpleNamespace(f_frsize=4096, f_bavail=1, f_favail=1)
    monkeypatch.setattr(host.os, "statvfs", lambda p: fs)
    with pytest.raises(host.InsufficientAuthority, match=r"deficit.*12288"):
        host.capacity(tmp_path, accepted, 4096)
    stable = tmp_path / "verified-weights"
    stable.write_bytes(b"w" * 8192)
    limits = {"learner_00": dict(bytes=8192, inodes=1)}
    retained = {"learner_00": {stable: stable.stat()}}
    fs.f_bavail = 2
    with pytest.raises(host.InsufficientAuthority, match="inodes"):
        host.capacity(tmp_path, accepted, 4096, limits, retained)
    fs.f_favail = 3
    host.capacity(tmp_path, accepted, 4096, limits, retained)
    stable.unlink()
    with pytest.raises(host.InsufficientAuthority, match="bytes"):
        host.capacity(tmp_path, accepted, 4096, limits, retained)
