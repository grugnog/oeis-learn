import pytest

from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.pipeline_validation.identity import seal
from tests.unit.test_progressive_freeze import campaign


def test_finite_slots_stop_when_predecessor_is_missing(tmp_path):
    c = campaign()
    store = RunStore(tmp_path, c)
    assert store.next_task() is None  # no task allocated to the prerequisite slot
    before = list(tmp_path.iterdir())
    assert store.read()["status"] == "FROZEN" and list(tmp_path.iterdir()) == before


def test_pause_is_owned_pending_request_and_resume_invocation_is_distinct(tmp_path):
    c = campaign()
    store = RunStore(tmp_path, c)
    store.append({"tasks": {}, "status": "RUNNING", "invocation": "first"})
    assert store.pause() == "PENDING_EXTERNAL_DEADLINE"
    assert store.pause_path().is_file()
    store.append({"tasks": {}, "status": "PAUSED", "invocation": "first"})
    store.append({"tasks": {}, "status": "RUNNING", "invocation": "second"})
    assert not store.pause_path().exists()
    changed = seal(dict(c, purpose="research"))
    with pytest.raises(ValueError):
        RunStore(tmp_path, changed)
