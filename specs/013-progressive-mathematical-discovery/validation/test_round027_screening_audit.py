"""Offline audit regressions, runnable with system Python at the pinned checkout."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("audit", Path(__file__).with_name("audit-round027-screening.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
from oeis_learn.generalization.progressive.contracts import reference, seal


def case(capped=False):
    draw = dict(complete=True, generation=dict(complete=True, reason=None),
                fitting=dict(complete=True, outcome="unsat_in_scope"))
    draws = [dict(copy.deepcopy(draw), position=i) for i in range(8)]
    if capped:
        draws[0].update(complete=False, source="", tokens=[3] * 1024, source_sha256=None,
                        raw=dict(outcome="not_attempted"), generation=dict(complete=False, reason="token_cap_without_eos"),
                        fitting=dict(complete=False, outcome="not_attempted"))
    arm = dict(complete=not capped, reason="token_cap_without_eos" if capped else "unsat_in_scope")
    rec = dict(costs={"correct": {"details": draws}}, selections={"equal_budget": None},
               arm_evidence={"equal_budget": arm})
    row = dict(upstream=arm, scoring=dict(complete=not capped, outcome="no_selected_candidate"),
               complete=not capped, success=False, candidate_sha256=None,
               outcome="unresolved" if capped else "failure")
    return rec, row


class AuditTests(unittest.TestCase):
    def test_only_fully_observed_cap_counts(self):
        clock = dict(started=8, terminal=8, termination="attempt_roster_terminal")
        original, score = case(True)
        self.assertEqual(audit.classify(original, score, clock)[:2], ("failure", 1))
        for change in ("timeout", "short", "eos", "missing", "clock", "scoring", "native"):
            with self.subTest(change=change):
                rec, s, c = copy.deepcopy((original, score, clock))
                draws = rec["costs"]["correct"]["details"]
                if change == "timeout":
                    draws[0]["generation"]["reason"] = "deadline"
                if change == "short":
                    draws[0]["tokens"].pop()
                if change == "eos":
                    draws[0]["tokens"][-1] = audit.EOS_ID
                if change == "missing":
                    draws.pop()
                if change == "clock":
                    c["termination"] = "parent_cut"
                if change == "scoring":
                    rec["selections"]["equal_budget"] = {"source": "example"}
                    s["scoring"] = dict(complete=False, outcome="not_attempted")
                if change == "native":
                    draws[-1]["fitting"] = dict(complete=False, outcome="resource_exhausted")
                self.assertEqual(audit.classify(rec, s, c)[0], "unresolved")
        original["selections"]["equal_budget"] = {"source": "example"}
        score["scoring"] = dict(complete=True, outcome="continuation_mismatch")
        self.assertEqual(audit.classify(original, score, clock)[0], "failure")
        score.update(success=True, outcome="success")
        score["scoring"]["outcome"] = "success"
        self.assertEqual(audit.classify(original, score, clock)[0], "success")

    def test_files_to_report(self):
        # Same nesting, path bases, producer seals and journal chain as the workstation.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def put(path, data):
                path = root / path
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(data))

            study = "sha256:" + "1" * 64
            campaign = seal(dict(payload=dict(tasks=[dict(id="trial-00-propose", trial=dict(study=study))])))
            cid = campaign["identity"]
            put("run-state/discovery/013-research-round-027/prepare-research/frozen/campaign.json", campaign)
            journal = Path("run-state/discovery-studies") / cid[7:] / study[7:]
            binding = seal(dict(study=study))
            put(journal / "binding.json", binding)
            roster = [dict(task=str(i), stratum=("composition", "unseen_structure", "parameter", "oeis")[i//32],
                           group=str(i//2)) for i in range(128)]
            outputs, refs, observed = {}, {}, {}
            for name, prefix in (("1000000-development", ""), ("retention-baseline", "retention_baseline_"),
                                 ("1000000-retention", "retention_candidate_")):
                generated, scored, clocks = {}, {}, []
                for i in range(128):
                    key = str(i)
                    rec, row = case(i < 5 and name != "1000000-retention")
                    generated[key], scored[key] = rec, dict(equal_budget=row)
                    clocks.append(audit.clock_record(prompt=key, mode="correct", snapshot="model", policy=audit.DRAIN,
                                  start=0, scope_deadline=500, finish=2, expected=8, started=8, terminal=8))
                generator = seal(dict(version="discovery-candidates/v1", methods=["correct"], model="model",
                                      records=generated, clock=seal(dict(policy=audit.DRAIN, clocks=clocks))))
                for op, producer, metric in (("generate", "generation", "candidates"), ("score", "scoring", "scores")):
                    task = f"trial-00-{name}-{op}"
                    role = Path(task) / "role"
                    doc = generator if op == "generate" else seal(dict(version="discovery-scoring/v1",
                               denominator=128, candidates=refs[prefix+"generation"]["sha256"], outcomes=scored))
                    path = role / metric / "data.json"
                    put(path, doc)
                    put(role / "result.json", seal(dict(mechanical_status="COMPLETE", operation=op,
                        metrics={metric:reference(root / role, root / path)})))
                    outputs[task] = dict(operation=op, result=reference(root, root / role / "result.json"))
                    refs[prefix + producer] = reference(root, root / path)
                observed[prefix] = [scored[str(i)]["equal_budget"]["outcome"] for i in range(128)]
            evidence = seal(dict(scope="NATIVE_VERIFIED", roster=roster, producer_refs=refs,
                           outcomes=[dict(r, outcome=v) for r,v in zip(roster, observed[""])],
                           retention=[dict(task=str(i), baseline=b, candidate=c) for i,(b,c) in
                                      enumerate(zip(observed["retention_baseline_"], observed["retention_candidate_"]))]))
            state = dict(proposal=dict(ordinal=0, study=study, parameters={}),
                         result=dict(status="INFEASIBLE", role="development", evidence=evidence))
            put(journal / "event-0000.json", seal(dict(previous=binding["identity"], ordinal="0", state=state)))
            put(Path("run-state/discovery-control") / cid[7:] / "transition-0000.json",
                seal(dict(state=dict(outputs=outputs))))
            with patch.object(audit, "CAMPAIGN", cid), patch.object(audit, "TRIALS", range(1)):
                result = audit.audit(root)
                self.assertEqual(result["passing_trials_by_1m_yield"], [0])
                row = result["trials"][0]
                self.assertFalse(row["frozen"]["passes_1m_checks"])
                self.assertEqual(row["exploratory"]["retention_loss_pp"], 0)
                self.assertEqual(row["panels"]["development"]["recoded_cases"], 5)
                path = root / refs["scoring"]["path"]
                path.write_bytes(path.read_bytes().replace(b"failure", b"unknown"))
                with self.assertRaisesRegex(ValueError, "hash/size mismatch"):
                    audit.audit(root)


if __name__ == "__main__":
    unittest.main()
