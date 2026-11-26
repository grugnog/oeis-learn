"""Read-side import of the closed round027 development study. No research dispatch."""

from collections import Counter
import json
from pathlib import Path
import zipfile

from oeis_learn.generalization.progressive.campaign import validate_diagnostic_registration
from oeis_learn.generalization.progressive.cohorts import STRATA, validate_prompt
from oeis_learn.generalization.progressive.contracts import (
    CAMPAIGN_DOCUMENT_BYTES, digest, load, read_bound_record, reference, seal,
)
from oeis_learn.generalization.progressive.program_search_bank import inspect_bank, replay_source, selected_source
from oeis_learn.generalization.progressive.program_search_routing import project, route
from .identity import confined
from .pilot_clocks import DRAIN, clock_record
from .progressive_research_inputs import checked_file

SESSION = "run-state/discovery/015-research-round-028"
ORIGINAL = "run-state/discovery/013-research-round-027"
FOLLOWUP = "run-state/discovery/013-round027-trial00-followup"
PRIMARY = ("composition", "unseen_structure")
METHODS = ("template", "polynomial", "linear_recurrence")
OUTCOMES = {"success", "failure", "unresolved"}
RULES = ("frozen", "companion")


def need(condition, message):
    if not condition:
        raise ValueError(message)


def roster_map(rows):
    members = {r["task"]: (r["stratum"], r["group"]) for r in rows}
    need(len(rows) == len(members) == 128 and Counter(s for s, _ in members.values()) == dict.fromkeys(STRATA, 32),
         "sealed 128-case roster required")
    return members


def join_measurement(value, look, members):
    need((value["trial"], value["coordinate"]) == look, "trial checkpoint differs")
    need(roster_map(value["roster"]) == members, "screening family or stratum differs")
    rows = value["panels"]["development"]
    need(rows.keys() == members.keys(), "private measurement roster differs")
    for row in rows.values():
        need(row["frozen"] in OUTCOMES and (row["frozen"] == row["companion"]
             or (row["frozen"], row["companion"]) == ("unresolved", "failure")), "historical scoring rule differs")
    return rows


def paired_counts(first, second):
    need(first.keys() == second.keys(), "paired case roster differs")
    counts = dict(both=0, first_only=0, second_only=0, neither=0, unresolved=0, unavailable=0,
                  denominator=len(first))
    labels = {(True, True): "both", (True, False): "first_only", (False, True): "second_only",
              (False, False): "neither"}
    for key, a in first.items():
        b = second[key]
        counts[labels[a == "success", b == "success"]] += 1
        counts["unresolved"] += "unresolved" in (a, b)
        counts["unavailable"] += "unavailable" in (a, b)
    return counts


def control_rows(outcomes, members, method):
    rows = {}
    for key in members.keys() & outcomes.keys():
        score = outcomes[key][method]
        outcome = score["outcome"]
        need(outcome == ("success" if score["success"] else "failure" if score["complete"] else "unresolved"),
             "control selection disagrees with verification")
        rows[key] = dict(frozen=outcome, companion=None,
                         search_complete=score.get("upstream", {}).get("complete", score["complete"]))
    return rows


def checkpoint_inventory(plans, replay, verify):
    """Only the training checkpoint index is read. Its evaluation roles are protected."""
    result = []
    for plan in plans:
        if plan["kind"] != "training" or plan["stage"] != "diversity":
            continue
        item = dict(arm=plan["arm"], seed=plan["seed"], target=plan["token_target"],
                    panels=["confirmation-diversity", "retention-diversity"], profile="bigint-20-100/v1",
                    disposition="unavailable", reason="missing_checkpoint_index")
        info = replay and replay["training_checkpoints"].get(plan["id"])
        if info and "error" not in info:
            item.update(source=info["source"], useful_tokens=info["useful_tokens"], files=info["files"])
            try:
                for ref in info["files"]:
                    verify(ref)
            except FileNotFoundError:
                item.update(reason="missing_checkpoint_closure")
            else:
                item.update(disposition="incompatible", reason="protected_panels_only")
        result.append(item)
    need(Counter(r["arm"] for r in result) == {"static": 2, "refreshed": 2}
         and all(r["target"] == 25000000 for r in result), "four original diversity checkpoints required")
    return result


class Donor:
    def __init__(self, root, session, registration):
        self.root = Path(root)
        self.session = Path(session)
        verified = validate_diagnostic_registration(registration)
        need(self.session == confined(root, SESSION), "wrong diagnostic owner")
        need(load(confined(session, "campaign.json")) == verified, "diagnostic policy not committed")
        from oeis_learn.generalization.progressive.lifecycle import RunStore
        owner = RunStore(session, verified, diagnostic=True)
        need(owner.state.get("registration") == verified["identity"], "diagnostic access not committed")
        self.policy = verified["policy"]
        self.refs = {}
        self.missing = set()

    def bound(self, ref, *, version=None):
        value = read_bound_record(self.root, ref, version=version)
        self.remember(ref)
        return value

    def remember(self, ref):
        prior = self.refs.setdefault(ref["path"], ref)
        need(prior == ref, "conflicting producer bytes for same path")

    def check(self, ref):
        # Tensor closure is streamed; it never enters the case matrix or public transport.
        if self.refs.get(ref["path"]) != ref:
            checked_file(self.root, ref)
            self.remember(ref)

    def saved(self, name, expected, *, maximum=32 << 20):
        path = confined(self.root, name)
        value = load(path, maximum=maximum)
        need(seal(value) == value and value["identity"] == expected, "committed historical identity differs: " + name)
        self.remember(reference(self.root, path))
        return value

    def optional(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except FileNotFoundError as exc:
            self.missing.add(str(Path(exc.filename).relative_to(self.root)))
            return None

    def replay_index(self):
        folder = confined(self.root, "run-state/discovery-tools", exists=False)
        for path in sorted(folder.glob("round027-comparison-*.zip")):
            with zipfile.ZipFile(confined(self.root, path)) as archive:
                if archive.namelist() != ["replay.json"]:
                    continue
                if archive.getinfo("replay.json").file_size > CAMPAIGN_DOCUMENT_BYTES:
                    raise ValueError("historical index exceeds the campaign document allowance")
                value = json.loads(archive.read("replay.json"))
            if value.get("identity") == self.policy["diversity_index"]:
                need(seal(value) == value and value["campaign"] == self.policy["campaign"], "diversity index differs")
                self.remember(reference(self.root, path))
                return value
        return None


def metric_ref(donor, cut, operation, metric):
    receipt = donor.bound(cut["result"])
    need(receipt["operation"] == cut["operation"] == operation and receipt["mechanical_status"] == "COMPLETE"
         and receipt.get("task") in (None, cut["task"])
         and receipt.get("campaign") in (None, donor.policy["campaign"]), "control producer does not match its cut")
    ref = receipt["metrics"][metric]
    parent = confined(donor.root, cut["result"]["path"]).parent
    child = confined(parent, ref["path"], exists=False)
    return dict(ref, path=str(child.relative_to(donor.root)))


def channels(donor, references, members):
    output = {}
    for key in members:
        data = {}
        for name, role, kind in (("visible", "generator", "visible_prompt"), ("truth", "scorer", "development_truth"),
                                 ("witness", "custodian", "coverage_witness")):
            ref = references.get(f"private/channels/{name}-{digest(key)[7:]}")
            data[name] = None
            if ref is None:
                continue
            need((ref["role"], ref["kind"]) == (role, "prepared"), "unregistered channel authority")
            value = donor.optional(donor.bound, ref["artifact"], version="discovery-channel/v1")
            if value is not None:
                need(value["role"] == role and value["kind"] == kind
                     and value["data"].get("id", value["data"].get("task")) == key, "unexpected historical channel")
                data[name] = value["data"]
                if name == "visible":
                    validate_prompt(data[name])
        output[key] = data
    return output


def import_controls(donor, members, case_rows, inputs):
    cut = donor.policy["capability_cut"]
    name = f"run-state/discovery-control/{donor.policy['campaign'][7:]}/{cut['name']}"
    control = donor.optional(donor.bound, dict(path=name, bytes=cut["bytes"], sha256=cut["sha256"]))
    producers = {}
    if control is None:
        return producers
    outputs = control["state"]["outputs"]
    for model in ("retained_D", "random"):
        seen = set()
        for i in range(8):
            prefix = f"capability-{model}-{i:02d}"
            try:
                gref = metric_ref(donor, outputs[prefix + "-generate"], "generate", "candidates")
                sref = metric_ref(donor, outputs[prefix + "-score"], "score", "scores")
            except FileNotFoundError as exc:
                donor.missing.add(str(Path(exc.filename).relative_to(donor.root)))
                continue
            gen = donor.optional(donor.bound, gref, version="discovery-candidates/v1")
            scores = donor.optional(donor.bound, sref, version="discovery-scoring/v1")
            producers[prefix] = dict(generation=gref, scoring=sref)
            if scores is None:
                continue
            need(scores["candidates"] == gref["sha256"] and scores["denominator"] == len(scores["outcomes"]),
                 "control scorer parent differs")
            if gen is not None:
                need(gen["model_control"] == model and gen["records"].keys() == scores["outcomes"].keys(),
                     "control generator identity or original roster differs")
            covered = members.keys() & scores["outcomes"].keys()
            need(not seen.intersection(covered), "duplicate control case across partitions")
            seen.update(covered)
            for method in ("equal_budget", *METHODS):
                label = model if method == "equal_budget" else f"{model}/{method}"
                for key, row in control_rows(scores["outcomes"], members, method).items():
                    score = scores["outcomes"][key][method]
                    if gen is not None:
                        required = "correct" if method == "equal_budget" else method
                        if required not in gen["methods"]:
                            row = dict(frozen="unavailable", companion=None, reason="method_not_executed")
                        else:
                            chosen = selected_source(gen["records"][key]["selections"][method], score, inputs[key]["visible"])
                            row["selected"] = chosen and chosen["source_sha256"]
                            local = inputs[key]
                            if chosen and score["success"] and local["truth"] is not None and not local.get("control_witness"):
                                if replay_source(chosen["source"], local["visible"], local["truth"]) == "solve":
                                    local["control_witness"] = dict(task=key, source=chosen["source"],
                                        indices=local["truth"]["indices"], values=local["truth"]["values"], origin=label)
                    row.update(producer=prefix, candidate=score["candidate_sha256"])
                    case_rows[key]["observations"][label] = row
    return producers


def clock_map(gen, members):
    summary = gen["clock"]
    need(seal(summary) == summary and summary["policy"] == DRAIN, "unverified historical search clock")
    clocks = {}
    for c in summary["clocks"]:
        if c["mode"] != "correct":
            continue
        fields = {k: c[k] for k in ("policy", "prompt", "mode", "snapshot", "start", "scope_deadline", "finish",
                                     "expected", "started", "terminal")}
        need(clock_record(**fields) == c and c["policy"] == DRAIN and c["snapshot"] == gen["model"]
             and c["prompt"] not in clocks, "changed or duplicate trial clock")
        clocks[c["prompt"]] = c
    need(clocks.keys() == members.keys(), "trial clock roster differs")
    return clocks


def look_name(trial, tokens):
    return f"trial-{trial:02d}-{tokens // 1000000}M"


def import_looks(donor, members, case_rows, inputs):
    looks = donor.policy["historical_looks"]
    final = donor.optional(donor.saved, FOLLOWUP + "/share/trial-00-2M.json", donor.policy["two_m_report"])
    if final is None:
        return {}, None
    need(final["campaign"] == donor.policy["followup"] and final["next_action"] == "STOP_AT_2M", "2M follow-up differs")
    first = donor.optional(donor.saved, FOLLOWUP + "/share/1M-replay.json", final["previous_report"])
    if first is not None:
        need(len(first["measurements"]) == 6 and first["registration"] == final["registration"],
             "six registered 1M measurements required")
    ids = [*(first["measurements"] if first else [None] * 6), final["measurement"]]
    producers = {}
    for (trial, tokens), mid in zip(looks, ids, strict=True):
        if mid is None:
            continue
        name = look_name(trial, tokens)
        filename = "trial-00-2000000" if tokens == 2000000 else name
        m = donor.optional(donor.saved, f"{FOLLOWUP}/private/{filename}.json", mid)
        if m is None:
            continue
        historical = join_measurement(m, (trial, tokens), members)
        refs = m["producers"]
        producers[name] = dict(measurement=mid, evidence=m["evidence"], **refs)
        # Only development case bytes enter diagnosis. Retention is already summarized
        # by the signed-off follow-up and never supplies a new case or witness.
        qualification = donor.optional(donor.bound, refs["qualification"])
        if qualification is not None:
            need(qualification.get("status") in {"PASS", "COMPLETE"} and qualification.get("device") == "actual_HIP",
                 "historical qualification scope differs")
        donor.optional(donor.check, refs["checkpoint"])
        producer_closure = qualification is not None and refs["checkpoint"]["path"] in donor.refs
        gref = refs["generation"]
        gen = donor.optional(donor.bound, gref, version="discovery-candidates/v1")
        scores = donor.optional(donor.bound, refs["scoring"], version="discovery-scoring/v1")
        if scores is not None:
            need(scores["candidates"] == gref["sha256"] and scores["denominator"] == 128
                 and scores["outcomes"].keys() == members.keys(), "trial scorer parent or roster differs")
        if gen is not None:
            need(gen["checkpoint"] == refs["checkpoint"]["sha256"] and gen["methods"] == ["correct"]
                 and gen["records"].keys() == members.keys(), "trial checkpoint, method or roster differs")
            if all(inputs[k]["visible"] is not None for k in members):
                need(gen["prompts"] == digest([inputs[k]["visible"] for k in sorted(members)]),
                     "trial visible input digest differs")
            clocks = clock_map(gen, members)
        for key, old in historical.items():
            obs = dict(old, measurement=mid, producer=name)
            case_rows[key]["observations"][name] = obs
            if scores is None:
                continue
            score = scores["outcomes"][key]["equal_budget"]
            need(score["outcome"] == old["frozen"] == ("success" if score["success"] else "failure"
                 if score["complete"] else "unresolved"), "trial scorer differs from original result")
            obs.update(candidate=score["candidate_sha256"], search_complete=score["upstream"]["complete"],
                       scoring=score["scoring"])
            if gen is None:
                continue
            rec = gen["records"][key]
            prompt = inputs[key]["visible"]
            selected_source(rec["selections"]["equal_budget"], score, prompt)
            if trial == donor.policy["diagnosis"]["trial"] and producer_closure:
                local = inputs[key]
                witness = local["witness"]
                if (witness is None or witness["source"] is None) and local.get("control_witness"):
                    witness = local["control_witness"]
                diagnosis = inspect_bank(rec, score, old["companion"], clocks[key], prompt, local["truth"],
                                         witness if local["truth"] is not None else None)
                if diagnosis["exact_witness"]:
                    diagnosis["witness_origin"] = witness.get("origin", "custodian")
                case_rows[key]["diagnosis"] = diagnosis
    return producers, final["training"]


def summarize(cases, methods):
    result = {}
    for name in methods:
        result[name] = {}
        for s in STRATA:
            observations = [c["observations"].get(name, {}) for c in cases if c["stratum"] == s]
            result[name][s] = {rule: dict(Counter(r.get(rule) or "unavailable" for r in observations))
                               for rule in RULES if rule == "frozen" or name.startswith("trial-")}
    pairs = {}
    for a, b in [("trial-05-1M", "retained_D"), ("trial-00-2M", "trial-00-1M"),
                 *((n, "retained_D") for n in methods if n not in {"retained_D", "trial-05-1M"})]:
        strata = {}
        for s in STRATA:
            overlap = {}
            for rule in RULES:
                def values(name):
                    col = rule if name.startswith("trial-") else "frozen"
                    return {r["task"]: r["observations"].get(name, {}).get(col) or "unavailable"
                            for r in cases if r["stratum"] == s}
                overlap[rule] = paired_counts(values(a), values(b))
            strata[s] = overlap
        pairs[f"{a} vs {b}"] = strata
    return result, pairs


def audit(donor):
    policy = donor.policy
    campaign = donor.saved(ORIGINAL + "/prepare-research/frozen/campaign.json", policy["campaign"],
                           maximum=CAMPAIGN_DOCUMENT_BYTES)
    refs = {r["name"]: r for r in campaign["payload"]["references"]}
    study = donor.bound(refs["study-policy"]["artifact"])
    members = roster_map(study["evaluations"]["trial-05-1000000-development-score"]["roster"])
    cases = {k: dict(task=k, stratum=s, group=g, observations={}) for k, (s, g) in members.items()}
    inputs = channels(donor, refs, members)
    for key in cases:
        cases[key]["channels"] = {k: refs.get(f"private/channels/{k}-{digest(key)[7:]}", {}).get("artifact")
                                  for k in ("visible", "truth", "witness")}
    producers = import_controls(donor, members, cases, inputs)
    trials, original_costs = import_looks(donor, members, cases, inputs)
    from oeis_learn.generalization.progressive.trials import historical_trials
    study_id = study["identity"]
    journal_dir = f"run-state/discovery-studies/{policy['campaign'][7:]}/{study_id[7:]}"
    history = donor.optional(historical_trials, donor.root, journal_dir, study_id)
    trial_costs = {}
    if history is not None:
        states, previous = history
        for i in range(6):
            result = states[str(i)]["result"]
            need(result["status"] == "INFEASIBLE" and result["role"] == "development", "formal trial result changed")
            name = look_name(i, 1000000)
            evidence = result["evidence"]
            if name in trials:
                need(evidence["identity"] == trials[name]["evidence"] and seal(evidence) == evidence
                     and evidence["producer_refs"] == {k: trials[name][k] for k in evidence["producer_refs"]},
                     "cached measurement differs from the original trial journal")
            trial_costs[name] = dict(status=result["status"], costs=evidence["costs"],
                                     scope=evidence["cost_scope"], evidence=evidence["identity"], journal_tip=previous)
        directory = confined(donor.root, journal_dir)
        for path in [directory / "binding.json", *sorted(directory.glob("event-*.json"))]:
            donor.remember(reference(donor.root, path))
    producers.update(trials)
    replay = donor.replay_index()
    checkpoints = checkpoint_inventory(campaign["payload"]["tasks"], replay, donor.check)
    methods = [look_name(*look) for look in policy["historical_looks"]]
    methods.extend([m for model in ("retained_D", "random") for m in (model, *(f"{model}/{k}" for k in METHODS))])
    rows = list(cases.values())
    aggregates, pairs = summarize(rows, methods)
    votes = []
    for case in rows:
        if case["stratum"] not in PRIMARY:
            continue
        old = case["observations"].get("trial-05-1M", {})
        evidence = case.get("diagnosis") or dict(feasible="N" if old.get("frozen") == "success" else "NSP",
                unfinished_fit=bool(old.get("reasons", {}).get("incomplete_fitting")), band_witness=False,
                fault=False, replay=None, bank_gap="missing_producers")
        case["diagnosis"] = evidence
        votes.append(dict(evidence, task=case["task"], stratum=case["stratum"]))
    from .progressive_round028_support import training_support
    support = training_support(donor, refs)
    projection = project(votes, training_support=support["eligible_rows"] > 0)
    inventory = {}
    for name, entry in refs.items():
        if (name.startswith(("iteration_calibration", "panel:")) or name in {
                "registered-research-data", "images", "preparation_data", "preparation_plan"}):
            artifact = entry["artifact"]
            present = donor.optional(checked_file, donor.root, artifact) is not None
            if present:
                donor.remember(artifact)
            inventory[name] = dict(artifact=artifact, role=entry["role"], kind=entry["kind"],
                                   disposition="verified_historical_scope" if present else "unavailable")
    return dict(cases=rows, original_trials=trial_costs, donor_inventory=inventory, methods=aggregates, pairs=pairs, producers=producers, checkpoints=checkpoints,
                support=support, projection=projection, decision=route(projection),
                followup_training=original_costs, original_budget=replay and replay["budget"],
                inputs=list(donor.refs.values()), missing_inputs=sorted(donor.missing))
