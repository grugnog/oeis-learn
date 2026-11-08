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


def test_actual_installed_role_interrupt_resume_and_successor(tmp_path):
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture, invoke_cpu

    c, tasks, channel, charge = fixture(tmp_path)
    first, _one = invoke_cpu(tmp_path, c, tasks[0], channel, ordinal=0)
    assert first["mechanical_status"] == "PARTIAL", first
    assert first["metrics"]["task_status"] == "PAUSED"
    resumed, two = invoke_cpu(tmp_path, c, tasks[0], channel, ordinal=1)
    assert resumed["mechanical_status"] == "COMPLETE", resumed
    successor, three = invoke_cpu(tmp_path, c, tasks[1], channel, ordinal=2)
    assert successor["mechanical_status"] == "COMPLETE", successor
    assert successor["metrics"]["charged_tokens"] == 5 * charge
    assert successor["metrics"]["discarded_tokens"] == charge
    assert successor["metrics"]["useful_tokens"] == 4 * charge
    from oeis_learn.generalization.progressive.contracts import load

    assert load(two / "role/checkpoint.json")["payload"]["model_update"] == 2
    assert load(three / "role/checkpoint.json")["payload"]["model_update"] == 4
    assert list((three / "role/progress").glob("*.json"))


def test_installed_role_partial_and_failure_publish_original_diagnostics(tmp_path):
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture, invoke_cpu

    partial = tmp_path / "partial"
    c, tasks, channel, charge = fixture(partial, interrupt=False, first_cap_updates=1)
    result, out = invoke_cpu(partial, c, tasks[0], channel, ordinal=0)
    assert result["metrics"]["task_status"] == "PARTIAL", result
    assert result["metrics"]["charged_tokens"] == charge
    failed = tmp_path / "failed"
    c, tasks, channel, _ = fixture(failed, interrupt=False, fail=True)
    result, out = invoke_cpu(failed, c, tasks[0], channel, ordinal=0)
    assert result["mechanical_status"] == "FAILED"
    assert result["producer_error"]["type"] == "ValueError"
    assert result["producer_error"]["local_traceback"]
    assert (out / "role/private/producer-traceback.txt").is_file()


def test_weights_only_child_is_new_initialization_with_fresh_clocks(tmp_path):
    from oeis_learn.generalization.progressive.contracts import atomic_write, digest, load, seal
    from oeis_learn.pipeline_validation.progressive_acceptance import fixture, invoke_cpu

    c, tasks, channel, _ = fixture(tmp_path, interrupt=False, weights_only=True)
    control = tmp_path / "control"
    atomic_write(
        control / ("pause-" + digest("invocation-0").split(":")[1] + ".json"),
        seal({"campaign": c["identity"], "requested": True}),
    )
    result, out = invoke_cpu(tmp_path, c, tasks[0], channel, ordinal=0)
    assert result["metrics"]["initialization"] == "new_initialization", result
    assert result["metrics"]["charged_tokens"] == 0
    assert result["metrics"]["ancestor_cost_available"] is False
    checkpoint = load(out / "role/checkpoint.json")
    assert checkpoint["payload"]["model_update"] == 0
