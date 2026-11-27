"""Apply exactly the same two interpretations to recorded and prospective looks."""
from collections import Counter
from functools import lru_cache
from pathlib import Path
import importlib.util
import re
import runpy
import time

from oeis_learn.generalization.progressive.contracts import digest, seal
from oeis_learn.generalization.progressive.cohorts import validate_prompt
from oeis_learn.pipeline_validation.identity import file_hash

# The pinned worker tree predates the host reader correction for large candidate
# records. Load the one reviewed verifier from the same extracted Git tree.
READER = Path(
    __file__).resolve().parents[4] / "src/oeis_learn/generalization/progressive/trial_evidence.py"
READER_HASH = "sha256:ad4970551770035570fa85e86f42164482f9c9211f67c39580bb7ec22926aa10"
if file_hash(READER) != READER_HASH:
    raise ValueError("extract the registered trial evidence reader alongside the follow-up")
reader_spec = importlib.util.spec_from_file_location(
    "oeis_learn.generalization.progressive.followup_evidence_reader", READER)
reader = importlib.util.module_from_spec(reader_spec)
reader_spec.loader.exec_module(reader)
objective = reader.objective

audit = runpy.run_path(str(Path(__file__).resolve().parents[1] / "audit-round027-screening.py"))
need = audit["require"]
HISTORICAL = "sha256:61a31b3a68ca5554e4bd6cde5b605492a9cb7e5b6a12e8b16e30559d906ce238"
PANEL = (("development", ""), ("baseline", "retention_baseline_"), ("candidate",
                                                                    "retention_candidate_"))


def reference_check(draw, indices):
    from oeis_learn.generalization.targets import _control_taint, ground
    from oeis_learn.synthesis.admission import source_identity
    from oeis_learn.synthesis.reference import evaluate
    from oeis_learn.synthesis.reference_parser import parse

    native = draw.get("native") or {}
    if not (draw["generation"]["complete"] is True and draw["complete"] is True
            and draw["fitting"]["complete"] is False and all(
                native.get(k) == v and draw["fitting"].get(k) == v
                for k, v in (("phase", "symbolic"), ("outcome", "resource_exhausted"),
                             ("reason", "control_range"))) and native.get("assignments") == []
            and native.get("searches") == [] and native.get("completed_bounds") == []
            and draw["fitting"].get("admission") == []):
        return "other_incomplete_fit"
    source = draw["source"]
    need(source_identity(source) == draw["source_sha256"], "source hash differs")
    # The native literal-abstraction policy protects controls. For explicit parameters
    # the independent taint check establishes that a zero assignment cannot
    # change the control path. This diagnostic never seeks an assignment that fits.
    try:
        program = parse(source, allow_parameters=True)
        if _control_taint(program):
            return "control_depends_on_fit_parameters"
        params = re.findall(r"\(\s*param\.get\s+(p[0-7])\s*\)", source)
        executable = ground(source, {p: "0" for p in params}) if params else source
    except ValueError:
        return "reference_source_unavailable"
    verdict = evaluate(executable, indices, deadline=time.monotonic() + 2)
    if verdict.outcome == "failure" and verdict.reason == "control_range":
        return "reproduced_control_range"
    return "reference_" + (verdict.reason or verdict.outcome)


def prompts(root, campaign):
    refs = {
        r["name"]: r["artifact"]
        for r in campaign["payload"]["references"] if r["role"] == "generator"
        and r["kind"] == "prepared" and r["name"].startswith("private/channels/visible-")
    }

    @lru_cache(maxsize=512)
    def get(key):
        doc = audit["artifact"](root, refs["private/channels/visible-" + digest(key)[7:]])
        need((doc["version"], doc["kind"], doc["role"],
              doc["data"]["id"]) == ("discovery-channel/v1", "visible_prompt", "generator", key),
             "visible input differs")
        return validate_prompt(doc["data"])

    return get


def classify_case(row, score, clock, indices):
    capped, caps, gaps = audit["classify"](row, score, clock)
    reasons = Counter(gaps)
    witnessed = capped
    if capped == "unresolved" and gaps.get("incomplete_fitting"):
        verdicts = [
            reference_check(draw, indices) for draw in row["costs"]["correct"]["details"]
            if draw["generation"]["complete"] is True and draw["fitting"]["complete"] is not True
        ]
        reasons.update(verdicts)
        if (set(gaps) == {"incomplete_fitting"} and len(verdicts) == gaps["incomplete_fitting"]
                and verdicts and all(v == "reproduced_control_range" for v in verdicts)):
            witnessed = "failure"
    if witnessed == "unresolved":
        reasons.update(
            "fitting:" + str(draw["fitting"].get("reason") or draw["fitting"].get("outcome"))
            for draw in row["costs"]["correct"]["details"]
            if draw["generation"]["complete"] is True and draw["fitting"]["complete"] is not True)
        if score["scoring"].get("outcome"):
            reasons["scoring:" + score["scoring"]["outcome"]] += 1
    return dict(frozen=score["outcome"],
                companion=witnessed,
                cap_attempts=caps,
                reasons=dict(reasons))


def panel(root, refs, outputs, name, prefix, expected, prompt):
    docs = []
    for operation, key, metric in (("generate", "generation", "candidates"), ("score", "scoring",
                                                                              "scores")):
        ref = refs[prefix + key]
        need(ref == audit["bound_ref"](root, outputs, name + "-" + operation, operation, metric),
             "producer does not belong to this look: " + name)
        docs.append(audit["artifact"](root, ref))
    gen, scores = docs
    need(
        gen["methods"] == ["correct"]
        and set(expected) == set(gen["records"]) == set(scores["outcomes"])
        and len(expected) == scores["denominator"] == 128
        and digest([prompt(k) for k in sorted(expected)]) == gen["prompts"],
        "panel or visible prompts differ")
    summary = audit["checked"](gen["clock"])
    clocks = {}
    for c in summary["clocks"]:
        fields = {
            k: c[k]
            for k in ("policy", "prompt", "mode", "snapshot", "start", "scope_deadline", "finish",
                      "expected", "started", "terminal")
        }
        need(
            c == audit["clock_record"](**fields)
            and c["policy"] == summary["policy"] == audit["DRAIN"] and c["mode"] == "correct"
            and c["snapshot"] == gen["model"] and c["prompt"] not in clocks,
            "portfolio clock differs")
        clocks[c["prompt"]] = c
    need(set(clocks) == set(expected), "clock roster differs")
    result = {}
    for key, frozen in expected.items():
        score = scores["outcomes"][key]["equal_budget"]
        need(score["outcome"] == frozen, "scorer and retained evidence disagree")
        result[key] = classify_case(gen["records"][key], score, clocks[key], prompt(key)["indices"])
    return result


def measure(root, trial, coordinate, evidence, outputs, prompt):
    e = audit["checked"](evidence)
    need(e["scope"] == "NATIVE_VERIFIED", "physical research producers required")
    reason = None
    try:
        value = objective(e, elapsed_us=sum(e["costs"].values()), root=root)
    except ValueError as exc:
        if str(exc) not in {"development completion below 95 percent", "retention regression"}:
            raise
        value, reason = None, str(exc)
    roster = e["roster"]
    need(
        len(roster) == len({r["task"]
                            for r in roster}) == len(e["outcomes"]) == len(e["retention"]) == 128
        and {(r["task"], r["group"], r["stratum"])
             for r in roster} == {(r["task"], r["group"], r["stratum"])
                                  for r in e["outcomes"]}, "original development roster differs")
    frozen = [{r["task"]: r["outcome"] for r in e["outcomes"]}]
    frozen.extend({r["task"]: r[arm] for r in e["retention"]} for arm in ("baseline", "candidate"))
    names = (f"{coordinate}-development", "retention-baseline", f"{coordinate}-retention")
    panels = {
        arm:
        panel(root, e["producer_refs"], outputs, f"trial-{trial:02d}-{name}", prefix, expected,
              prompt)
        for (arm, prefix), name, expected in zip(PANEL, names, frozen, strict=True)
    }
    return seal(
        dict(trial=trial,
             coordinate=coordinate,
             roster=roster,
             panels=panels,
             producers=e["producer_refs"],
             evidence=e["identity"],
             native_value=value,
             native_reason=reason))


def summarize(m):
    audit["checked"](m)
    summary = dict(trial=m["trial"], coordinate=m["coordinate"])
    panels = m["panels"]
    for rule in ("frozen", "companion"):
        arms = []
        for arm, _ in PANEL:
            rows = panels[arm]
            for row in rows.values():
                need(
                    row["frozen"] in {"success", "failure", "unresolved"}
                    and (row["companion"] == row["frozen"] or
                         (row["frozen"], row["companion"]) == ("unresolved", "failure")),
                    "invalid case resolution")
            arms.append({k: r[rule] for k, r in rows.items()})
        values = audit["evaluate"](m["roster"], *arms)
        summary[rule] = values
        summary[rule + "_outcomes"] = {
            arm: dict(Counter(a.values()))
            for (arm, _), a in zip(PANEL, arms)
        }
    need(
        summary["frozen"]["passes_1m_checks"] == (m["native_reason"] is None)
        and (m["native_value"] is None or m["native_value"] == summary["frozen"]["verified_yield"]),
        "frozen metrics disagree with the installed screening gate")
    summary["frozen_reason"] = m["native_reason"]
    reasons = {}
    for arm, _ in PANEL:
        counts = Counter()
        for row in panels[arm].values():
            if row["companion"] == "unresolved":
                counts.update(row["reasons"].keys() or ["unresolved_without_reclassification"])
        reasons[arm] = dict(counts)
    summary["unresolved_reasons_by_case"] = reasons
    return summary


def verify_history(rows):
    need([r["trial"] for r in rows] == list(range(6))
         and all(r["coordinate"] == 1000000 for r in rows),
         "all six original looks required in order")
    need(
        digest([[r["trial"], r["frozen"], r["companion"]] for r in rows]) == HISTORICAL,
        "historical replay differs from the six saved measurements")
    need([r["trial"] for r in rows if r["frozen"]["passes_1m_checks"]] == []
         and [r["trial"] for r in rows if r["companion"]["passes_1m_checks"]] == [0],
         "1M eligibility differs")


def next_action(m):
    row = summarize(m)
    need(row["trial"] == 0 and row["coordinate"] in (2000000, 4000000), "unregistered look")
    return ("CONTINUE_TO_4M" if row["coordinate"] == 2000000
            and row["companion"]["passes_1m_checks"] else "STOP_AT_" +
            ("2M" if row["coordinate"] == 2000000 else "4M"))
