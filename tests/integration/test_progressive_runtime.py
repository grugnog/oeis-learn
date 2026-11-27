"""Real watchdog containment; image execution is a distinct required gate."""

import json
import sys
import time
from types import SimpleNamespace
import pytest
from tests.helpers.progressive_fixture import task, caps
from oeis_learn.pipeline_validation.progressive_roles import deadline_for
from oeis_learn.pipeline_validation.generalization_roles import supervise_owned


def test_only_discovery_authority_can_extend_six_hour_ceiling():
    t = task(caps=caps(elapsed_us=86_400_000_000))
    assert (
        deadline_for(t, remaining_us=86_400_000_000, seconds=30, operation="train")
        > time.monotonic()
    )
    with pytest.raises(ValueError):
        deadline_for(t, remaining_us=86_400_000_000, seconds=22000, operation="train")
    with pytest.raises(ValueError):
        deadline_for(t, remaining_us=100, seconds=30, operation="train")


def test_shared_watchdog_terminates_real_hung_subprocess(tmp_path):
    removed = []
    host = SimpleNamespace(
        engine=SimpleNamespace(remove=lambda name, timeout: removed.append((name, timeout)))
    )
    with pytest.raises(TimeoutError):
        supervise_owned(
            host,
            "owned",
            [sys.executable, "-c", "while True: pass"],
            tmp_path,
            deadline=time.monotonic() + 1.2,
            cleanup_seconds=0.5,
        )
    receipt = json.loads((tmp_path / "supervision/process.json").read_text())
    assert receipt["returncode"] is not None
    assert receipt["outcome"] == "FAIL"
    assert removed and removed[0][0] == "owned"


def test_generic_role_guard_stays_six_hours(tmp_path):
    from oeis_learn.pipeline_validation.generalization_roles import invoke

    with pytest.raises(ValueError, match="bounded role duration"):
        invoke(
            None,
            "image",
            tmp_path,
            "learner",
            {},
            output="out",
            operation="train",
            settings={},
            seconds=22000,
        )
