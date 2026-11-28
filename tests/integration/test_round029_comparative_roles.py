from pathlib import Path

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
    cap = targets[1000]["useful_tokens"]
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
    ref = previous[0] / "checkpoint" / checked["weight"]["path"]
    with ref.open("ab") as f:
        f.write(b"bad")
    with pytest.raises(ValueError):
        host.check_cut(work / "output", request, learner, producer, targets[1000])
