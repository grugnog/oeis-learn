"""Host-adapter regression tests. Run with pytest from the repository root."""
import copy
import fcntl
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest

from oeis_learn.generalization.progressive.budgets import BudgetJournal, KEYS
from oeis_learn.generalization.progressive.contracts import atomic_write, load, record, reference, seal
from oeis_learn.generalization.progressive.decisions import publish, verify_gate_producers
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.generalization.progressive.projections import describe, project
from oeis_learn.pipeline_validation import progressive_role
from tests.unit.test_progressive_dag_decisions import configured

spec = importlib.util.spec_from_file_location("publication_recovery", Path(__file__).with_name("continue-round027-publication.py"))
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


@pytest.fixture
def round026_space(tmp_path):
    launcher = Path(__file__).resolve().with_name("launch-round027.sh")
    (tmp_path / "src").symlink_to(launcher.parents[3] / "src", target_is_directory=True)
    folder = tmp_path / "run-state/discovery/013-research-round-026"
    stage = folder / "prepare-research/base-qualification/pipeline/stages/cpu.semantic/bigint-cpu-regressions"
    scratch = stage / ".runtime"
    scratch.mkdir(parents=True)
    (folder / "session.lock").touch()
    (scratch / "fixture.pt").write_bytes(b"disposable only with the original proof")
    atomic_write(stage / "command.json", dict(argv=[f"type=bind,src={stage},dst=/work/output",
                 "TMPDIR=/work/output/.runtime/tmp", "--basetemp=/work/output/.runtime/pytest"]))

    def run(*args):
        return subprocess.run(["bash", "-s", "--", *args], cwd=tmp_path,
                              text=True, input=launcher.read_text(), capture_output=True)

    return folder, scratch, run


def test_round026_storage_mode_cannot_launch_or_clean_without_the_original_proof(round026_space):
    folder, scratch, run = round026_space
    before = {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()}
    assert run("--round026-space", "--help").returncode == 0
    rejected = run("some-commit", "--round026-space", "--apply")
    assert rejected.returncode == 2 and "Usage:" in rejected.stderr
    for args in [(), ("--apply",)]:
        result = run("--round026-space", *args)
        assert result.returncode == 2 and "Round026 storage check stopped" in result.stderr
        assert str(scratch.relative_to(folder)) in result.stdout
        assert {p: p.read_bytes() for p in folder.rglob("*") if p.is_file()} == before
    assert run("--round026-space", "--app").returncode == 2


def test_round026_storage_preview_apply_and_repeat(tmp_path, round026_space):
    folder, scratch, run = round026_space
    # Substitute only the identity of the historical failure. Exercise the production
    # shell entrypoint, both handoff comparisons, file accounting, locking and removal.
    src = tmp_path / "src"
    real_src = src.resolve()
    src.unlink()
    package = Path("oeis_learn/pipeline_validation")
    for path in (package.parent, package):
        (src / path).mkdir(parents=True)
        (src / path / "__init__.py").write_text(f"__path__.append({str(real_src / path)!r})\n")
    (src / package / "progressive_round025_successor.py").write_text(
        "SUCCESSOR = 'run-state/discovery/013-research-round-026'\n"
        "RETRY = 'run-state/discovery/013-research-round-027'\n"
        "def registration_failure(root, receipt, session, journal, ancestor):\n"
        "    return ancestor\n")
    proof = seal(dict(closed=True))
    for name in ("session.json", "journal.json", "preparation-owner-proof.json"):
        atomic_write(folder / name, proof)
    atomic_write(folder / "preparation-owner.json", seal(dict(receipt=proof)))
    next_proof = folder.with_name("013-research-round-027") / "preparation-owner-proof.json"
    atomic_write(next_proof, seal(dict(closed=False)))
    retained = folder / "prepare-research" / "retained-input"
    retained.write_bytes(b"reusable")
    os.link(retained, scratch / "linked-input")
    (scratch / "symlink-to-preparation").symlink_to(retained.parent, target_is_directory=True)
    other = scratch.parents[1] / "unrecorded-worker" / ".runtime"
    other.mkdir(parents=True)
    (other.parent / "command.json").write_text("{")
    (other / "keep").write_bytes(b"unclassified")

    def check(*args):
        return run("--round026-space", *args)

    mismatch = check("--apply")
    assert mismatch.returncode == 2 and "differs from the handoff" in mismatch.stderr
    next_proof.unlink()
    atomic_write(next_proof, proof)
    with (folder / "session.lock").open("rb") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert check("--apply").returncode == 2
    assert check().returncode == 0 and (scratch / "fixture.pt").is_file()
    removed = check("--apply")
    assert removed.returncode == 0 and "Verified the handoff after cleanup" in removed.stdout
    assert not scratch.exists() and retained.read_bytes() == b"reusable"
    assert (other / "keep").read_bytes() == b"unclassified" and load(next_proof) == proof
    repeated = check("--apply")
    assert repeated.returncode == 0 and "No eligible round026 qualification scratch remains" in repeated.stdout


def test_panel_comparison_checks_saved_references_and_counts_distinct_cases(tmp_path, monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("reporter", Path(__file__).with_name("report-round027.py"))
    reporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reporter)
    folder = tmp_path / reporter.FOLLOWUP
    roster = [dict(task=f"{s}-{i}", stratum=s, group=str(i)) for s in reporter.STRATA for i in range(32)]
    refs = []
    for i in range(7):
        trial, coordinate, name = (i, 1000000, f"trial-{i:02d}-1M") if i < 6 else (0, 2000000, "trial-00-2000000")
        rows = {r["task"]: {"companion": "success" if r["group"] == str(i) else "failure"} for r in roster}
        look = seal(dict(trial=trial, coordinate=coordinate, roster=roster, panels=dict(development=rows)))
        atomic_write(folder / f"private/{name}.json", look)
        refs.append(look["identity"])
    replay = seal(dict(measurements=refs[:6]))
    final = seal(dict(previous_report=replay["identity"], measurement=refs[6]))
    atomic_write(folder / "share/1M-replay.json", replay)
    atomic_write(folder / "share/trial-00-2M.json", final)
    monkeypatch.setattr(reporter, "TWO_M_REPORT", final["identity"])
    outputs = {}
    for model, outcome in (("random", "unresolved"), ("retained_D", "success")):
        for part in range(8):
            name = f"capability-{model}-{part:02d}-score"
            role = tmp_path / name
            cases = {r["task"]: {"equal_budget": {"outcome": outcome}} for r in roster[part::8]}
            atomic_write(role / "scores.json", dict(version="discovery-scoring/v1", outcomes=cases))
            atomic_write(role / "result.json", dict(operation="score", mechanical_status="COMPLETE",
                         metrics=dict(scores=reference(role, role / "scores.json"))))
            outputs[name] = dict(result=reference(tmp_path, role / "result.json"))
    atomic_write(tmp_path / "cut.json", dict(state=dict(outputs=outputs)))
    monkeypatch.setattr(reporter, "CAPABILITY_CUT", reference(tmp_path, tmp_path / "cut.json"))
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    reporter.panel_report(tmp_path)
    output = capsys.readouterr().out
    assert "0 / 32" in next(line for line in output.splitlines() if line.startswith("random"))
    assert "32 / 0" in next(line for line in output.splitlines() if line.startswith("retained_D"))
    assert output.count("6/32") == 4 and "trial 00 2M" in output
    assert {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    path = tmp_path / "capability-random-00-score/scores.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError):
        reporter.panel_report(tmp_path)
    path.unlink()
    reporter.panel_report(tmp_path)
    partial = capsys.readouterr()
    assert "Capability reference unavailable: random" in partial.err
    assert "retained_D" in partial.out and "trial 05 1M" in partial.out


def test_elapsed_only_amendment(tmp_path):
    journal = BudgetJournal(tmp_path, "campaign", dict.fromkeys(KEYS, 10**9))
    costs = dict.fromkeys(KEYS, 0)
    costs.update(native_work=10**6, elapsed_us=20_000_000)
    original = journal.reserve(recovery.NAME, "publish", 95, costs, boot="test", monotonic_us=0)
    journal.commit(original["key"], boot="test", monotonic_us=12_818_869, termination_proven=True)
    replay = dict(ledger=reference(tmp_path, journal.path), budget=journal.snapshot())
    tail, costs = recovery.charge_state(journal, replay)
    assert not tail and costs["native_work"] == 0
    journal.reserve(recovery.NAME, "publish", recovery.ORDINAL, costs, boot="test", monotonic_us=30_000_000)
    assert len(recovery.charge_state(journal, replay)[0]) == 1
    journal.commit(recovery.KEY, boot="test", monotonic_us=35_000_000, termination_proven=True)
    assert len(recovery.charge_state(journal, replay)[0]) == 2
    spent = journal.snapshot()["consumed"]
    assert (spent["native_work"], spent["elapsed_us"]) == (1_000_000, 17_818_869)
    wrong = copy.deepcopy(replay)
    wrong["ledger"]["sha256"] = "wrong"
    with pytest.raises(ValueError, match="prefix changed"):
        recovery.charge_state(journal, wrong)
    journal.reserve("other", "publish", 0, costs, boot="test", monotonic_us=0)
    with pytest.raises(ValueError, match="already run"):
        recovery.charge_state(journal, replay)


def test_interrupted_formal_publication(tmp_path, monkeypatch):
    fx, old_bundle, policy, remaining = configured(tmp_path)
    campaign = copy.deepcopy(fx.campaign)
    gate = load(tmp_path/"gate.json")
    rows = old_bundle["rows"]
    for producer in gate["producers"]:
        if producer["arm"] != "candidate" or producer["seed"] != 1:
            continue
        scores = load(tmp_path / producer["scores"]["path"])
        for row in rows:
            if row["arm"] == "candidate" and row["seed"] == 1 and row["outcome"] == "success":
                row["complete"] = False
                scores["outcomes"][row["task"]]["equal_budget"]["complete"] = False
        atomic_write(tmp_path/"partial-scores.json", seal(scores))
        producer["scores"] = reference(tmp_path, "partial-scores.json")
    atomic_write(tmp_path/"partial-gate.json", seal(gate))
    next(r for r in campaign["payload"]["references"] if r["name"] == "gate")["artifact"] = reference(tmp_path, "partial-gate.json")
    plans = campaign["payload"]["tasks"]
    plan = dict(plans[0], id=recovery.NAME, output="compare", kind="evaluation", token_target=0)
    plans.insert(-1, plan)
    plans[-1]["id"] = "trial-00-propose"
    graph = campaign["payload"]["graph"]
    graph["nodes"][0]["caps"] = {k: 3*v for k,v in plan["caps"].items()}
    graph["edges"][0]["condition"] = "COMPLETE"
    campaign = seal(campaign)
    store = RunStore(tmp_path/"control", campaign)
    task = record("task", dict(plan, campaign=campaign["identity"]), created_at=campaign["created_at"],
                  purpose=campaign["purpose"], track=campaign["track"])
    original = dict(store.state, tasks={"parent-1":"COMPLETE", "parent-2":"COMPLETE", recovery.NAME:"FAILED"}, status="FAILED")
    store.append(original)
    assert store.next_task() is None
    journal = BudgetJournal(tmp_path/"account", campaign["identity"], campaign["payload"]["caps"])
    costs = dict.fromkeys(KEYS, 0)
    costs["elapsed_us"] = 180
    journal.reserve(recovery.NAME, "publish", 95, costs, boot="test", monotonic_us=0)
    journal.commit(recovery.NAME + "/publish/95")
    before = reference(tmp_path, journal.path)
    journal.reserve(recovery.NAME, "publish", recovery.ORDINAL, costs, boot="test", monotonic_us=0)
    journal.commit(recovery.KEY)
    data = {k:old_bundle[k] for k in ("source_slot","target_slot","rows","roster","seeds","design")}
    kwargs = dict(target_id="gain", action="RETAIN_BASELINE")
    decision = publish(rows, data["roster"], data["seeds"], data["design"], policy=policy, remaining=remaining,
                       created_at=policy["created_at"], purpose=policy["purpose"], track=policy["track"], **kwargs)
    bundle = seal(dict(data, version="progressive-decision-bundle/v1", decision=decision, kwargs=kwargs))
    replay = dict(ledger=before, decision=decision, decision_bundle=bundle["identity"],
                  gate_registration=verify_gate_producers(tmp_path, campaign, bundle, store=store))
    table = seal(dict(version="discovery-channel/v1", role="verifier", kind="public_evidence",
                      data=dict(data, version="progressive-decision-input/v1", kwargs=kwargs, remaining=remaining, policy=policy)))
    assert recovery.check_table(table, replay) == bundle
    atomic_write(tmp_path/"table.json", table)
    a = describe(tmp_path, "table.json")
    target = tmp_path/"publisher"
    project(tmp_path, target/"inputs", "publish", {"public":[a]}, registry=recovery.progressive_roles.REGISTRY, authorized=[a.ref])
    result = progressive_role.execute("publish", inputs=target/"inputs", output=target/"role")
    atomic_write(target/"supervision/process.json", dict(returncode=0, cleanup_failure=None))
    atomic_write(target/"supervision/cleanup.json", dict(acknowledged=True, within_bound=True))
    assert recovery.terminal(target, bundle) == result
    atomic_write(tmp_path/"authorization.json", seal(dict(test="synthetic")))
    def interrupt(*args):
        raise InterruptedError
    for obj, name in ((recovery.progressive_outputs, "commit"), (store, "boundary")):
        with monkeypatch.context() as patch:
            patch.setattr(obj, name, interrupt)
            with pytest.raises(InterruptedError):
                recovery.finish(tmp_path, store, task, journal, target, bundle, replay, original)
        assert store.next_task() is None
    for _ in range(2):
        recovery.finish(tmp_path, store, task, journal, target, bundle, replay, original)
    assert len(store.state["decisions"]) == 1
    assert store.next_task()["payload"]["id"] == "trial-00-propose"
    assert load(target/"role/decision-audit.json")["original_budget_prefix"] == before
    assert load(target/"role/gain-cost.json")["costs"][recovery.NAME] == 360
