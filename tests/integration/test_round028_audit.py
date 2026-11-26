import json
from pathlib import Path

import pytest

from oeis_learn.generalization.progressive.campaign import diagnostic_registration
from oeis_learn.generalization.progressive.contracts import atomic_write, digest, reference, seal
from oeis_learn.generalization.progressive.lifecycle import RunStore
from oeis_learn.pipeline_validation.progressive_round028_audit import (
    Donor, join_measurement, roster_map, control_rows, checkpoint_inventory, paired_counts,
)


def roster():
    return [dict(task=f"case-{s}-{i}", stratum=s, group=f"{s}-{i // 2}")
            for s in ("parameter", "composition", "unseen_structure", "oeis") for i in range(32)]


def test_private_reader_requires_committed_registration(tmp_path):
    reg = diagnostic_registration("a" * 40)
    with pytest.raises((OSError, ValueError)):
        Donor(tmp_path, tmp_path / "uncommitted", reg)
    session = tmp_path / "run-state/discovery/015-research-round-028"
    session.mkdir(parents=True)
    with RunStore.owner(session):
        store = RunStore(session, reg, diagnostic=True)
        with pytest.raises(ValueError, match="committed"):
            Donor(tmp_path, session, reg)
        store.append(dict(store.state, registration=reg["identity"]))
        assert Donor(tmp_path, session, reg).policy == reg["policy"]


def test_join_preserves_scoring_rules_and_reports_missing_measurements():
    members = roster()
    ids = roster_map(members)
    look = dict(trial=5, coordinate=1000000, roster=members, panels={"development":{
        k:dict(frozen="unresolved", companion="failure", reasons={"incomplete_fitting": 1}, cap_attempts=1)
        for k in ids}})
    result = join_measurement(look, (5, 1000000), ids)
    assert result[members[0]["task"]]["frozen"] == "unresolved"
    assert result[members[0]["task"]]["companion"] == "failure"
    with pytest.raises(ValueError, match="checkpoint"):
        join_measurement(look, (0, 2000000), ids)
    with pytest.raises(ValueError):
        roster_map(members[:-1] + [members[0]])


def test_control_on_larger_roster_and_paired_unresolved_overlay():
    keys = roster_map(roster())
    scores = {k: {"equal_budget": dict(outcome="failure", success=False, complete=True)}
              for k in [*keys, "outside-screening"]}
    one = next(iter(keys))
    scores[one]["equal_budget"] = dict(outcome="success", success=True, complete=False)
    rows = control_rows(scores, keys, "equal_budget")
    assert set(rows) == set(keys)
    assert rows[one]["frozen"] == "success" and rows[one]["search_complete"] is False
    before = {one:"success", "two":"failure", "three":"unresolved", "four":"unavailable"}
    after = {one:"success", "two":"success", "three":"failure", "four":"failure"}
    assert paired_counts(before, after) == dict(both=1, first_only=0, second_only=1, neither=2,
                                                unresolved=1, unavailable=1, denominator=4)


def test_protected_checkpoint_inventory_never_opens_confirmation_scores():
    plans = [dict(id=f"diversity-{arm}-{seed}", arm=arm, seed=seed, token_target=25000000,
                  kind="training", stage="diversity")
             for seed in (11, 12) for arm in ("static", "refreshed")]
    items = checkpoint_inventory(plans, None, None)
    assert len(items) == 4 and {i["disposition"] for i in items} == {"unavailable"}


def install_donor_fixture(root, monkeypatch):
    from oeis_learn.generalization.progressive import campaign
    from oeis_learn.pipeline_validation.pilot_clocks import DRAIN, clock_record
    from oeis_learn.pipeline_validation.progressive_round028_audit import FOLLOWUP, ORIGINAL, METHODS
    from oeis_learn.synthesis.admission import source_identity

    policy = campaign.learned_search_policy()
    def put(path, value):
        record = seal(value)
        atomic_write(root / path, record, compact=True)
        return reference(root, path), record["identity"]
    members = roster()
    keys = [r["task"] for r in members]
    by_name = []
    for k in keys:
        for prefix, role, kind, size in (("visible", "generator", "visible_prompt", 20),
                                        ("truth", "scorer", "development_truth", 100)):
            logical = f"private/channels/{prefix}-{digest(k)[7:]}"
            indices = list(map(str, range(7, 7 + size)))
            data = dict(id=k, indices=indices, values=indices)
            ref, _ = put(f"original/{logical}.json", dict(version="discovery-channel/v1", role=role, kind=kind, data=data))
            by_name.append(dict(name=logical, role=role, kind="prepared", artifact=ref))
    evaluation = {"trial-05-1000000-development-score": dict(roster=members)}
    study, _ = put("original/study.json", dict(evaluations=evaluation))
    plans = [dict(id=f"diversity-{arm}-{i}", arm=arm, seed=i, token_target=25000000, kind="training", stage="diversity")
             for arm in ("static", "refreshed") for i in (1, 2)]
    by_name.append(dict(name="study-policy", role="proposal", kind="policy", artifact=study))
    _, policy["campaign"] = put(ORIGINAL + "/prepare-research/frozen/campaign.json",
                                    dict(payload=dict(references=by_name, tasks=plans)))
    outputs = {}
    score = dict(outcome="failure", success=False, complete=True, candidate_sha256=None,
                 scoring=dict(outcome="no_selected_candidate", complete=True), upstream={"complete": True})
    for model in ("retained_D", "random"):
        for i in range(8):
            part = keys[i * 16:i * 16 + 16] + [f"outside-{i}"]
            name = f"capability-{model}-{i:02d}"
            methods = ("equal_budget", *METHODS)
            records = {k: {"selections": dict.fromkeys(methods)} for k in part}
            g, _ = put(f"original/{name}/candidates.json", dict(version="discovery-candidates/v1", model_control=model,
                          methods=["correct", *METHODS], records=records))
            s, _ = put(f"original/{name}/scores.json", dict(version="discovery-scoring/v1", candidates=g["sha256"],
                          denominator=len(part), outcomes={k: dict.fromkeys(methods, score) for k in part}))
            for op, metric, ref in (("generate", "candidates", g), ("score", "scores", s)):
                nested = dict(ref, path=Path(ref["path"]).name)
                receipt, _ = put(f"original/{name}/{op}.json", dict(operation=op, mechanical_status="COMPLETE",
                                                                      metrics={metric:nested}))
                outputs[name+"-"+op] = dict(operation=op, task=digest(name + op), result=receipt)
    transition = f"run-state/discovery-control/{policy['campaign'][7:]}/transition-0096.json"
    ref, _ = put(transition, dict(state=dict(outputs=outputs)))
    policy["capability_cut"] = dict(name=Path(transition).name, sha256=ref["sha256"], bytes=ref["bytes"])
    mids = []
    q, _ = put("original/qualification.json", dict(status="PASS", device="actual_HIP"))
    for t, n in policy["historical_looks"]:
        name = f"trial-{t:02d}-1M" if n == 1000000 else "trial-00-2000000"
        weights, _ = put("original/"+name+"/weights.json", dict(test_weights=name))
        clocks = [clock_record(prompt=k, mode="correct", snapshot=name, policy=DRAIN, start=0, scope_deadline=500,
                               finish=2, expected=8, started=8, terminal=8) for k in keys]
        prompts = [dict(id=k, indices=list(map(str, range(7,27))), values=list(map(str, range(7,27)))) for k in sorted(keys)]
        records, classifications, scored = {}, {}, {}
        for member in members:
            k = member["task"]
            unfinished = t == 5 and member["stratum"] == "composition" and int(k.split("-")[-1]) < 6
            source = "(program (return (int 0)))"
            details = [dict(position=i, source=source, source_sha256=source_identity(source),
                            generation={"complete": True}, fitting=dict(complete=not (unfinished and i == 1),
                            outcome="unsat_in_scope" if not (unfinished and i == 1) else "resource_exhausted"))
                       for i in range(8)]
            portfolio = dict(details=details, complete=not unfinished, archive=dict(retained=[], omitted=[]), equal_budget=None)
            upstream = dict(complete=not unfinished)
            records[k] = dict(costs=dict(correct=portfolio), selections=dict(equal_budget=None), arm_evidence=dict(equal_budget=upstream))
            scored[k] = {"equal_budget": dict(score, upstream=upstream, complete=not unfinished,
                                                outcome="unresolved" if unfinished else "failure")}
            classifications[k] = dict(frozen="unresolved" if unfinished else "failure",
                                      companion="unresolved" if unfinished else "failure", reasons={"incomplete_fitting":1} if unfinished else {})
        g, _ = put(f"original/{name}/candidates.json", dict(version="discovery-candidates/v1", model=name,
                       checkpoint=weights["sha256"], methods=["correct"], prompts=digest(prompts), records=records,
                       clock=seal(dict(policy=DRAIN, clocks=clocks))))
        s, _ = put(f"original/{name}/scores.json", dict(version="discovery-scoring/v1", outcomes=scored,
                       candidates=g["sha256"], denominator=128))
        _, mid = put(f"{FOLLOWUP}/private/{name}.json", dict(trial=t, coordinate=n, evidence=digest(name), roster=members,
                    panels=dict(development=classifications), producers=dict(checkpoint=weights, qualification=q,
                    generation=g, scoring=s)))
        mids.append(mid)
    _, first = put(FOLLOWUP + "/share/1M-replay.json", dict(registration=digest(1), measurements=mids[:6]))
    _, policy["two_m_report"] = put(FOLLOWUP + "/share/trial-00-2M.json", dict(campaign=policy["followup"],
              next_action="STOP_AT_2M", registration=digest(1), measurement=mids[-1], previous_report=first, training={}))
    monkeypatch.setattr(campaign, "learned_search_policy", lambda: policy)
    return diagnostic_registration("a" * 40)


def test_registered_audit_consumes_all_original_looks_then_reuses_one_boundary(tmp_path, monkeypatch):
    from argparse import Namespace
    from oeis_learn.pipeline_validation import progressive_round028 as commands
    from oeis_learn.pipeline_validation.progressive_round028 import audit_command, ARCHIVE, reopen
    from oeis_learn.pipeline_validation.progressive_round028_audit import SESSION
    from oeis_learn.generalization.progressive.program_search_report import verify_archive
    from oeis_learn.pipeline_validation.progressive_outputs import round028_batch
    import zipfile

    reg = install_donor_fixture(tmp_path, monkeypatch)
    folder = tmp_path / SESSION
    folder.mkdir(parents=True)
    with RunStore.owner(folder):
        store = RunStore(folder, reg, diagnostic=True)
        original = audit_command(tmp_path, folder, store, reg)
        state = store.tip
        assert original["denominators"] == {"historical":128, "routing":64}
        assert original["unfinished_trial05_composition"] == 6
        assert original["decision"]["action"] == "RETAIN_REFERENCE"
        assert len([name for name in original["methods"] if name.startswith("trial-")]) == 7
        assert audit_command(tmp_path, folder, store, reg) == original and store.tip == state
        cut = reopen(tmp_path, folder, reg, store.state["audit"])
        with pytest.raises(ValueError, match="order"):
            round028_batch(folder, cut["batches"][0], registration=reg["identity"], ordinal=1)
        assert len(cut["cases"]) == 128
    archive = folder / ARCHIVE
    assert verify_archive(archive)["decision"] == original["decision"]
    with zipfile.ZipFile(archive) as zipped:
        raw = zipped.read("audit.json")
        assert b"case-composition" not in raw and b"(program" not in raw and b"original/" not in raw
    # The launcher checks the actual Git identity separately. Here the test donor
    # keeps its fixture hashes for both the workspace and archive verification.
    monkeypatch.setattr(commands, "learned_search_policy", lambda: reg["policy"])
    monkeypatch.setattr(commands, "metadata", lambda root, policy: dict(qualified_checkout=True,
                        files={commands.ORIGINAL + "/prepare-research/frozen/campaign.json": True}))
    monkeypatch.setattr(commands, "candidate_revision", lambda source, root: reg["candidate"])
    args = Namespace(workspace=str(tmp_path), archive=None, action="verify")
    receipt, code = commands.execute(args, tmp_path)
    assert code == 0 and receipt["private_replay"] == "VERIFIED"
    assert receipt["public_recomputation"]["report"] == original["receipt"]["report"]
    args.action = "status"
    assert commands.execute(args, tmp_path)[0]["scientific_status"] == "HISTORICAL_AUDIT"
    args.action = "run"
    assert commands.execute(args, tmp_path) == (original, 0)
    monkeypatch.setattr(commands, "candidate_revision", lambda source, root: "b" * 40)
    # A new implementation verifies the published parent under its original
    # registration. The same physical work must never acquire a second debit.
    assert commands.execute(args, tmp_path) == (original, 0)
    assert RunStore(folder, reg, diagnostic=True).tip == state
    assert json.loads((folder / "campaign.json").read_text()) == reg
    (tmp_path / "original/trial-05-1M/weights.json").write_text("modified donor")
    with pytest.raises(ValueError, match="identity"):
        reopen(tmp_path, folder, reg, store.state["audit"])


def test_replay_separates_passed_over_source_absent_bytes_and_unfinished_selection():
    from oeis_learn.fresh.synthesis_policy import select_visible
    from oeis_learn.generalization.progressive.program_search_bank import inspect_bank
    from oeis_learn.synthesis.admission import VisiblePrompt, source_identity
    from oeis_learn.fresh.candidate_archive import retain_alternatives

    indices = list(map(str, range(100)))
    truth = dict(id="private", indices=indices, values=["0"] * 20 + ["1"] * 80)
    prompt = dict(id="private", indices=indices[:20], values=truth["values"][:20])
    miss = "(program (return (int 0)))"
    good = "(program (return (select (lt $index (int 20)) (int 0) (int 1))))"
    originals = [dict(id=source_identity(s), source=s, structure=digest(i), cost=1, verified=True,
                      verification=digest(i), track="generic_only", profile="bigint-body/v1", assignment={})
                 for i, s in enumerate((miss, good))]
    archive = retain_alternatives(originals)
    chosen = select_visible([dict(source=r["source"], source_sha256=r["id"], verified=True, visible_count=20,
                                  cost=r["cost"]) for r in originals])
    env = dict(selected=chosen, prompt_sha256=VisiblePrompt(tuple(prompt["indices"]), tuple(prompt["values"])).sha256)
    clock = dict(started=8, termination="attempt_roster_terminal", snapshot=digest("model"))
    details = [dict(position=i, source=miss, source_sha256=source_identity(miss), generation=dict(complete=True),
                    fitting=dict(complete=True, outcome="unsat_in_scope", admission=[])) for i in range(8)]
    for i, r in enumerate(originals):
        details[i].update(source=r["source"], source_sha256=r["id"])
        details[i]["fitting"].update(outcome="verified_solution", admission=[dict(outcome="verified_solution", source_sha256=r["id"])])
    portfolio = dict(archive=archive, details=details, complete=True, equal_budget=env)
    rec = dict(costs=dict(correct=portfolio), selections=dict(equal_budget=env), arm_evidence=dict(equal_budget=dict(complete=True)))
    score = dict(candidate_sha256=source_identity(miss), success=False, upstream=rec["arm_evidence"]["equal_budget"],
                 scoring=dict(outcome="continuation_mismatch", complete=True))
    witness = dict(task="private", indices=indices, values=truth["values"], source=good)
    def run():
        return inspect_bank(rec, score, "success" if score["success"] else "failure", clock, prompt, truth, witness)
    assert run()["feasible"] == "S"
    archive["retained"] = [originals[0]]
    archive["omitted"] = [dict(id=source_identity(good))]
    assert run()["bank_gap"] == "missing_discarded_source" and run()["feasible"] == "NSP"
    details[1]["seed"] = 128
    details[1]["fitting"]["clock"] = dict(admitted=2.0, deadline=5.75)
    recovered = run()
    assert recovered["replay"] == dict(ordinals=[1], cost_us=3750000, resolves="NSP")
    assert recovered["replay_inputs"][0]["source_sha256"] == source_identity(good)
    archive["omitted"] = []
    details[1]["fitting"].update(outcome="unsat_in_scope", admission=[])
    assert run()["feasible"] == "P"
    details[3]["fitting"].update(complete=False, outcome="resource_exhausted", reason="timeout")
    portfolio["complete"] = False
    details[3]["seed"] = 129
    details[3]["fitting"]["clock"] = dict(admitted=2., deadline=6.)
    assert run()["replay"] == dict(ordinals=[3], cost_us=4000000, resolves="NP")
    score.update(success=True, scoring=dict(outcome="success", complete=True))
    truth["values"][20:] = ["0"] * 80
    evidence = run()
    assert evidence["feasible"] == "N" and evidence["unfinished_ordinals"] == [3] and not evidence["search_closed"]
    score["candidate_sha256"] = digest("wrong")
    with pytest.raises(ValueError, match="hash"):
        run()


def test_verified_full_states_on_protected_panels_have_no_development_vote():
    plans = [dict(kind="training", stage="diversity", arm=a, seed=i, token_target=25000000, id=f"{a}-{i}")
             for a in ("static", "refreshed") for i in range(2)]
    index = dict(training_checkpoints={p["id"]: dict(source={}, useful_tokens=25000001, files=[p["id"]]) for p in plans})
    checked = []
    rows = checkpoint_inventory(plans, index, checked.append)
    assert set(checked) == set(index["training_checkpoints"])
    assert all(row["disposition"] == "incompatible" and row["reason"] == "protected_panels_only" for row in rows)


def test_pinned_source_cache_rejects_tampering_extra_imports_and_directory_aliases(tmp_path):
    import runpy
    import subprocess
    helper = Path(__file__).resolve().parents[2] / "specs/015-learned-program-search/validation/candidate_source.py"
    materialize = runpy.run_path(str(helper))["materialize"]
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "worker.py").write_text("qualified = True\n")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()
    git("add", "worker.py")
    git("-c", "user.name=fixture", "-c", "user.email=fixture@example.org", "commit", "-qm", "source")
    pin = git("rev-parse", "HEAD")
    source = materialize(tmp_path, pin)
    assert source == materialize(tmp_path, pin)
    (source / "extra.py").write_text("pass")
    with pytest.raises(ValueError, match="unexpected candidate"):
        materialize(tmp_path, pin)
    (source / "extra.py").unlink()
    (source / "worker.py").write_text("changed")
    with pytest.raises(ValueError, match="cached candidate differs"):
        materialize(tmp_path, pin)
    (source / "worker.py").unlink()
    (source / "worker.py").symlink_to(tmp_path / "worker.py")
    with pytest.raises(ValueError, match="alias"):
        materialize(tmp_path, pin)


def test_copied_public_share_recomputes_route_without_workspace(tmp_path):
    import copy
    import subprocess
    import sys
    import zipfile
    from oeis_learn.generalization.progressive.program_search_report import public_audit, publish_archive, verify_archive
    from oeis_learn.generalization.progressive.program_search_routing import project, route
    from oeis_learn.pipeline_validation.progressive_round028_audit import summarize, look_name, METHODS
    reg = diagnostic_registration("a" * 40)
    cases = [dict(r, observations={}, diagnosis=dict(feasible="NSP", fault=False, unfinished_fit=False, band_witness=False, replay=None))
             for r in roster()]
    methods = [look_name(*look) for look in reg["policy"]["historical_looks"]] + [
        k for m in ("retained_D", "random") for k in (m, *(m + "/" + method for method in METHODS))]
    counts, pairs = summarize(cases, methods)
    votes = [dict(r["diagnosis"], task=r["task"], stratum=r["stratum"]) for r in cases if r["stratum"] in ("composition", "unseen_structure")]
    p = project(votes, training_support=False)
    checkpoints = [dict(arm=a, seed=i, target=25000000, disposition="unavailable", reason="missing_checkpoint_index")
                   for a in ("refreshed", "static") for i in (1, 2)]
    public = public_audit(reg, dict(methods=counts, pairs=pairs, checkpoints=checkpoints, original_trials={}, followup_training={},
                          support=dict(status="unavailable", eligible_rows=0, eligible_families=0, positions=0),
                          projection=p, decision=route(p), cases=cases), elapsed_us=14, private_identity=digest("private"))
    archive = tmp_path / "round028.zip"
    publish_archive(archive, public)
    cli = Path(__file__).resolve().parents[2] / "scripts/foundation_run.py"
    out = subprocess.check_output([sys.executable, str(cli), "--engine", "podman", "discovery", "round028",
                                   "--action", "verify", "--archive", str(archive)], cwd=tmp_path, text=True)
    assert json.loads(out)["scope"] == "archive_only" and not (tmp_path / "run-state").exists()
    changed = copy.deepcopy(public)
    changed["pairs"]["trial-05-1M vs retained_D"]["composition"]["frozen"]["both"] = 1
    with zipfile.ZipFile(tmp_path / "edited.zip", "w") as z:
        z.writestr("audit.json", json.dumps(seal(changed), sort_keys=True))
    with pytest.raises(ValueError):
        verify_archive(tmp_path / "edited.zip")
    with zipfile.ZipFile(tmp_path / "embed.zip", "w") as z:
        z.writestr("audit.json", json.dumps(public, sort_keys=True).replace('{"', '{"scope":"private encoded trace","', 1))
    with pytest.raises(ValueError, match="canonical"):
        verify_archive(tmp_path / "embed.zip")
