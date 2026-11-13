"""Host orchestration counterexamples; runtime attestations below are unit doubles."""

from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive.contracts import atomic_write, load, reference, seal
from oeis_learn.pipeline_validation import progressive_commands as commands
from oeis_learn.pipeline_validation import progressive_roles as roles
from oeis_learn.pipeline_validation.progressive_acceptance import fixture


@pytest.mark.parametrize("producer_failure", [False, True])
def test_run_pause_resume_status_keep_original_caps_and_new_outputs(
    tmp_path, monkeypatch, producer_failure
):
    c, _tasks, _channel, charge = fixture(tmp_path, interrupt=False)
    atomic_write(
        tmp_path / "producer/qualification.json",
        {"covered_capabilities": ["progressive-lifecycle"]},
    )
    atomic_write(
        tmp_path / "runtime.json",
        {"payload": {"qualification": [reference(tmp_path, "producer/qualification.json")]}},
    )
    atomic_write(
        tmp_path / "accepted/acceptance.json",
        seal(
            {
                "version": "discovery-acceptance/v1",
                "scope": "progressive-lifecycle",
                "qualified": True,
                "campaign": c["identity"],
                "runtime": c["payload"]["runtime"],
                "status": "COMPLETE",
            }
        ),
    )
    monkeypatch.setattr(
        roles, "current_runtime", lambda *a, **kw: SimpleNamespace(execution_ready=True)
    )
    called = []

    def invoke(*a, task, journal, **kw):
        left = journal.task_remaining(task)
        called.append((task["payload"]["id"], left["tokens"], kw["seconds"]))
        if producer_failure and len(called) == 2:
            raise RuntimeError("original owned producer failure")
        debit = dict.fromkeys(left, 0)
        debit.update(tokens=charge, elapsed_us=1)
        r = journal.reserve(
            task["payload"]["id"],
            "unit-orchestration",
            len(called),
            debit,
            boot="unit",
            monotonic_us=0,
        )
        journal.commit(r["key"])
        status = "PAUSED" if len(called) == 1 else "COMPLETE"
        result=seal(dict(version='discovery-result/v1',operation='train',mechanical_status='PARTIAL' if status=='PAUSED' else 'COMPLETE',metrics={'task_status':status}))
        atomic_write(tmp_path/kw['output']/'role/result.json',result)
        return result

    monkeypatch.setattr(roles, "invoke", invoke)

    def run(operation, invocation, **kw):
        token = commands._INVOCATION.set(invocation)
        try:
            return commands.lifecycle_command(
                SimpleNamespace(operation=operation, **kw), tmp_path, lambda _: object()
            )
        finally:
            commands._INVOCATION.reset(token)

    first, code = run("run", "first", campaign="campaign.json", acceptance="accepted", output="one")
    assert code == 2 and first["reason"] == "PAUSE_ACKNOWLEDGED"
    assert commands.status(SimpleNamespace(run="one"), tmp_path)[0]["mechanical_status"] == "PAUSED"
    assert run("pause", "pause", run="one")[0]["mechanical_status"] == "PAUSED"
    if producer_failure:
        with pytest.raises(RuntimeError, match="original owned producer"):
            run("resume", "second", run="one", output="two")
        assert (
            commands.status(SimpleNamespace(run="one"), tmp_path)[0]["mechanical_status"]
            == "FAILED"
        )
        return
    resumed, code = run("resume", "second", run="one", output="two")
    assert code == 0 and resumed["mechanical_status"] == "COMPLETE"
    assert called[0][0] == called[1][0] == "first" and called[1][1] == called[0][1] - charge
    assert called[2][0] == "second"
    assert load(tmp_path / "two/run.json")["previous"] == "one"
    assert (
        commands.status(SimpleNamespace(run="one"), tmp_path)[0]["mechanical_status"] == "COMPLETE"
    )
    with pytest.raises(ValueError):
        run("resume", "third", run="one", output="one")


def test_limited_cpu_receipt_cannot_authorize_research_or_runtime(tmp_path):
    c, *_ = fixture(tmp_path, interrupt=False)
    atomic_write(
        tmp_path / "accepted/acceptance.json",
        seal(
            {
                "version": "progressive-lifecycle-acceptance/v1",
                "qualified": False,
                "campaign": c["identity"],
                "runtime": c["payload"]["runtime"],
                "status": "COMPLETE",
            }
        ),
    )
    token = commands._INVOCATION.set("unqualified")
    try:
        with pytest.raises(ValueError, match="acceptance"):
            commands.lifecycle_command(
                SimpleNamespace(
                    operation="run",
                    campaign="campaign.json",
                    acceptance="accepted",
                    output="blocked",
                ),
                tmp_path,
                lambda _: pytest.fail("must not contact engine"),
            )
    finally:
        commands._INVOCATION.reset(token)
