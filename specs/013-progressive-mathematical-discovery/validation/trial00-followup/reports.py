"""Durable private measurements and a small independently reproducible public report."""
from collections import Counter

from oeis_learn.generalization.progressive.contracts import load, seal
from oeis_learn.pipeline_validation.identity import confined, file_hash
from oeis_learn.pipeline_validation.preparation_recovery import retain
from oeis_learn.pipeline_validation.progressive_study import original, weight_ref, costs as phase_costs
from evidence import audit, measure, need, next_action, summarize, verify_history
from registration import ACCOUNT, CHECKPOINT, LOOKS, RULE, STUDY, content, names, one_ref
from execution import state_dir, task_for


def check_refs(root, refs):
    for ref in refs.values():
        path = confined(root, ref["path"])
        need(
            path.is_file() and path.stat().st_size == ref["bytes"]
            and file_hash(path) == ref["sha256"], "retained producer bytes changed: " + ref["path"])


def saved_measurement(root, path, trial, look, e, outputs, prompts, *, replay=False):
    if path.exists() and not replay and look == 1000000:
        m = audit["saved"](path)
        need((m["trial"], m["coordinate"], m["evidence"], m["producers"],
              m["roster"]) == (trial, look, e["identity"], e["producer_refs"], e["roster"]),
             "cached measurement binding differs")
        check_refs(root, m["producers"])
    else:
        m = measure(root, trial, look, e, outputs, prompts)
        retain(path, m)
    return m


def format_rule(row):
    result = dict(row)
    for name in ("frozen", "companion"):
        metrics = dict(result[name])
        metrics["passes"] = metrics.pop("passes_1m_checks")
        result[name] = metrics
    return result


def changes(before, after):
    need(before["roster"] == after["roster"], "development roster changed between looks")
    result = {}
    for arm, rows in after["panels"].items():
        old = before["panels"][arm]
        need(rows.keys() == old.keys(), "paired cases differ")
        result[arm] = {
            rule: dict(Counter(old[k][rule] + " to " + v[rule] for k, v in rows.items()))
            for rule in ("frozen", "companion")
        }
    return result


def write_report(path, value):
    # Reopening after atomic publication is also the standalone verifier's path.
    checked = seal(value)
    retain(path, checked)
    need(audit["saved"](path) == checked, "published report differs from verified measurements")
    return checked


def history(root, home, source, states, prompt, *, replay=False):
    measurements = []
    for i in range(6):
        print(f"Checking saved 1M evidence for trial {i:02d}.", flush=True)
        result = states[i]["result"]
        need((result["status"], result["role"]) == ("INFEASIBLE", "development"),
             "original trial status differs")
        m = saved_measurement(root,
                              home / "private" / f"trial-{i:02d}-1M.json",
                              i,
                              1000000,
                              audit["checked"](result["evidence"]),
                              source.state["outputs"],
                              prompt,
                              replay=replay)
        measurements.append(m)
    rows = [summarize(m) for m in measurements]
    verify_history(rows)
    report = write_report(
        home / "share/1M-replay.json",
        dict(version="trial00-replay/v1",
             registration=audit["saved"](home / "registration.json")["identity"],
             original_status="INFEASIBLE",
             frozen_passes=[],
             companion_passes=[0],
             trials=[format_rule(r) for r in rows],
             measurements=[m["identity"] for m in measurements]))
    return measurements[0], report


def look_evidence(root, source, store, journal, coordinate, one_m):
    base = f"trial-00-{coordinate}"
    policy = audit["checked"](content(root, one_ref(store.campaign, "study-policy")))
    need(policy["identity"] == STUDY, "study policy differs")
    spec = policy["evaluations"][base + "-development-score"]
    expected = dict(generation_task=base + "-development-generate",
                    checkpoint_task=base + "-train",
                    retention_baseline_generation="trial-00-retention-baseline-generate",
                    retention_baseline_scoring="trial-00-retention-baseline-score",
                    retention_candidate_generation=base + "-retention-generate",
                    retention_candidate_scoring=base + "-retention-score")
    need(
        all(spec[k] == name for k, name in expected.items()) and spec["roster"] == one_m["roster"],
        "frozen evaluation design or retention reference changed")
    q = task_for(store, base + "-train")["payload"]["iteration_budget"]["producer"]
    need(spec["qualification"] in (None, q), "model qualification differs")
    refs = dict(checkpoint=weight_ref(root, store, base + "-train"), qualification=q)
    arms = [("", "generation_task", base + "-development-score", store),
            ("retention_baseline_", "retention_baseline_generation",
             spec["retention_baseline_scoring"], source),
            ("retention_candidate_", "retention_candidate_generation",
             spec["retention_candidate_scoring"], store)]
    scores = []
    for prefix, key, scoring, owner in arms:
        refs[prefix + "generation"] = original(root, owner, spec[key], "generate", "candidates")
        refs[prefix + "scoring"] = original(root, owner, scoring, "score", "scores")
        scores.append(audit["artifact"](root, refs[prefix + "scoring"])["outcomes"])
    need(
        all(refs[k] == one_m["producers"][k]
            for k in ("retention_baseline_generation", "retention_baseline_scoring")),
        "original retention baseline changed")
    roster = spec["roster"]
    retention = [
        dict(task=k,
             baseline=scores[1][k]["equal_budget"]["outcome"],
             candidate=scores[2][k]["equal_budget"]["outcome"])
        for k in sorted(one_m["panels"]["baseline"])
    ]
    gen = audit["artifact"](root, refs["generation"])
    scored = audit["artifact"](root, refs["scoring"])
    included = {n for look in LOOKS if look <= coordinate for n in names(look)}
    plans = [t for t in store.campaign["payload"]["tasks"] if t["id"] in included]
    timing, _ = phase_costs(journal, {"payload": {"tasks": plans}}, plans[0]["trial"])
    return seal(
        dict(version="progressive-trial-evidence/v1",
             role="development",
             scope="NATIVE_VERIFIED",
             costs=timing,
             cost_scope="supervised task intervals; startup and cleanup enclosed in phase costs",
             generator=gen["identity"],
             scorer=scored["identity"],
             checkpoint=refs["checkpoint"]["sha256"],
             qualification=audit["artifact"](root, q)["identity"],
             roster=roster,
             outcomes=[
                 dict(r, outcome=scores[0][r["task"]]["equal_budget"]["outcome"]) for r in roster
             ],
             retention=retention,
             producer_refs=refs))


def checkpoint_costs(root, store, journal, coord):
    task = task_for(store, names(coord)[0])
    role = confined(root, store.state["outputs"][names(coord)[0]]["result"]["path"]).parent
    cp, final = (audit["saved"](role / f) for f in ("checkpoint.json", "accounting-final.json"))
    p = cp["payload"]
    full = content(state_dir(root, task) / "checkpoints", p["full_state"])
    milestone = audit["checked"](final["milestone"])
    need(
        final["task"] == p["task"] == task["identity"] and final["checkpoint"] == cp["identity"]
        and p["campaign"] == store.campaign["identity"] and not final["pending"]
        and full["ledger"]["pending"] is None and p["model_update"] == full["ledger"]["updates"]
        and coord <= full["ledger"]["consumed"] <= milestone["tokens"]
        and final["remaining"] == journal.task_remaining(task),
        "completed training/accounting cut differs")
    # Validate the learner's complete stored tensor closure by streaming every artifact.
    from oeis_learn.generalization.contracts import artifact_refs
    for ref in artifact_refs(full["states"]):
        check_refs(state_dir(root, task) / "checkpoints", {"state": ref})
    charges = {}
    for name in names(coord):
        plan = task_for(store, name)["payload"]
        left = journal.plan_remaining(plan)
        charges[name] = {k: v - left[k] for k, v in plan["caps"].items()}
    need(
        milestone["new_tokens"] == charges[names(coord)[0]]["tokens"]
        and milestone["new_cost_us"] == charges[names(coord)[0]]["elapsed_us"],
        "learner settlement differs")
    return dict(checkpoint=cp["identity"],
                parent=final["ancestor_checkpoint"],
                updates=p["model_update"],
                model_tokens=full["ledger"]["consumed"],
                accounting=final["identity"],
                milestone=milestone,
                charges=charges)


def publish_look(root,
                 home,
                 source,
                 store,
                 journal,
                 coord,
                 one_m,
                 prior,
                 prior_report,
                 prompt,
                 *,
                 replay=False):
    evidence = look_evidence(root, source, store, journal, coord, one_m)
    path = home / "private" / f"trial-00-{coord}.json"
    outputs = dict(source.state["outputs"], **store.state["outputs"])
    m = saved_measurement(root, path, 0, coord, evidence, outputs, prompt, replay=replay)
    costs = checkpoint_costs(root, store, journal, coord)
    donor = audit["saved"](state_dir(root, task_for(store, "trial-00-1000000-train")) /
                           "import.json")
    expected_parent = donor["checkpoint"] if coord == LOOKS[0] else prior_report["training"][
        "checkpoint"]
    expected_tokens = 1000564 if coord == LOOKS[0] else prior_report["training"]["model_tokens"]
    need(
        costs["parent"] == expected_parent
        and costs["milestone"]["ancestor_tokens"] == expected_tokens,
        "child did not continue the verified preceding full-state checkpoint")
    reg = audit["saved"](home / "registration.json")
    report = write_report(
        home / "share" / f"trial-00-{coord//1000000}M.json",
        dict(version="trial00-look/v1",
             campaign=store.campaign["identity"],
             registration=reg["identity"],
             acceptance=audit["saved"](home / "acceptance/acceptance.json")["identity"],
             selection=RULE["selection"],
             source_checkpoint=CHECKPOINT,
             source_accounting=ACCOUNT,
             original_1M_tokens=1000564,
             original_1M_training_us=578661207,
             setup_worker_seconds={
                 stage: load(home / path)["elapsed_seconds"]
                 for stage, path in (("checkpoint_import",
                                      "import/worker/supervision/process.json"),
                                     ("software_acceptance", "acceptance/software/process.json"),
                                     ("device_acceptance", "acceptance/device/process.json"))
             },
             coordinate=coord,
             previous_report=prior_report["identity"],
             measurement=m["identity"],
             metrics=format_rule(summarize(m)),
             paired_changes_since_previous=changes(prior, m),
             training=costs,
             next_action=next_action(m)))
    return m, report
