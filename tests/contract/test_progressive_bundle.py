"""Public bundle authority is independently reduced, never a supplied PASS."""

import json
import zipfile
import pytest
from oeis_learn.generalization.progressive.contracts import atomic_write, digest, seal, reference
from oeis_learn.generalization.progressive.report import build
from oeis_learn.generalization.progressive.verify import verify


def test_report_campaign_reference_and_embedded_record_follow_their_contracts():
    from oeis_learn.generalization.progressive.contracts import schema

    properties = schema("report")["properties"]
    assert properties["campaign"] == {"$ref": "progressive-common.schema.json#/$defs/hash"}
    assert properties["campaign_record"]["oneOf"] == [
        {"type": "null"}, schema("campaign")
    ]


def fixture(root):
    roster = [dict(task=digest("task"), group=digest("group"), stratum="composition")]
    rows = [
        dict(roster[0], seed=s, arm=a, outcome="failure")
        for s in [1, 2]
        for a in ["baseline", "candidate"]
    ]
    docs = dict(roster=roster, outcomes=rows, accounting=[], ancestry=[], qualifications=[])
    tables = {}
    for k, v in docs.items():
        atomic_write(root / (k + ".json"), v)
        tables[k] = [reference(root, k + ".json")]
    request = seal(
        dict(
            version="progressive-public-report-input/v1",
            campaign_record=None,
            scope_record=None,
            terminal=None,
            campaign=digest("campaign"),
            source=digest("source"),
            seeds=[1, 2],
            design=dict(claim="capability", bootstrap_seed=13, comparisons=1, links=[], costs=None),
            scope=dict(required=list(docs), qualification_scopes=[], tasks=[]),
            tables=tables,
            decisions=[],
            producer_error=None,
            collector_errors=[],
        )
    )
    return bind_scope(root, request, roster)


def test_bounded_closed_export_and_independent_incomplete_verdict(tmp_path):
    request = fixture(tmp_path)
    report = build(tmp_path, tmp_path / "report.zip", request)
    assert report["status"] == "COMPLETE"
    result = verify(tmp_path / "report.zip")
    assert result["status"] == "VERIFIED"
    assert result["decision"] == "INCOMPLETE"
    with zipfile.ZipFile(tmp_path / "report.zip") as z:
        assert all("private" not in p and not p.endswith(".pt") for p in z.namelist())


def rewrite(path, edit):
    with zipfile.ZipFile(path) as z:
        docs = {n: z.read(n) for n in z.namelist()}
    edit(docs)
    with zipfile.ZipFile(path, "w") as z:
        for n, b in docs.items():
            z.writestr(n, b)


@pytest.mark.parametrize("change", ["counts", "missing", "gate", "extra", "zip_path"])
def test_forged_or_missing_evidence_never_verifies(tmp_path, change):
    build(tmp_path, tmp_path / "report.zip", fixture(tmp_path))

    def edit(docs):
        manifest = json.loads(docs["manifest.json"])
        if change == "counts":
            manifest["summary"]["outcomes"] = 99
        if change == "gate":
            manifest["decisions"] = [dict(verdict="CONFIRMED_GAIN", action="ADVANCE_REGISTERED")]
        if change == "missing":
            del docs["tables/roster-0000.json"]
        if change == "extra":
            docs["private.json"] = b"{}"
        if change == "zip_path":
            docs["../escape.json"] = b"{}"
        docs["manifest.json"] = json.dumps(seal(manifest)).encode()

    rewrite(tmp_path / "report.zip", edit)
    with pytest.raises(ValueError):
        verify(tmp_path / "report.zip")


def test_export_bounds_and_missing_qualification_are_explicit(tmp_path):
    request = fixture(tmp_path)
    request = seal(dict(request, scope=dict(request["scope"], qualification_scopes=["hip-short"])))
    request = bind_scope(tmp_path, request, json.loads((tmp_path / "roster.json").read_text()))
    build(tmp_path, tmp_path / "report.zip", request)
    assert verify(tmp_path / "report.zip")["status"] == "INCOMPLETE"
    with pytest.raises(ValueError):
        build(tmp_path, tmp_path / "tiny.zip", request, max_bytes=20)


def test_independent_reducer_matches_gold_paired_negative_and_positive():
    from tests.unit.test_progressive_decisions import fixture as panel
    from oeis_learn.generalization.progressive.verify import reduce_outcomes

    data, roster, design, _ = panel()
    metrics, verdict = reduce_outcomes(data, roster, [1, 2], design)
    assert verdict == "CONFIRMED_GAIN" and metrics["mean_gain"] > 0
    for r in data:
        if r["arm"] == "candidate" and r["stratum"] != "retention":
            r["outcome"] = "failure"
    metrics, verdict = reduce_outcomes(data, roster, [1, 2], design)
    assert verdict == "NO_MATERIAL_GAIN" and metrics["upper"] == 0
    for r in data:
        if r["arm"] == "baseline" and r["stratum"] != "retention":
            r["outcome"] = "success"
    metrics, verdict = reduce_outcomes(data, roster, [1, 2], design)
    assert verdict == "REGRESSION" and metrics["upper"] == -1


def test_original_qualification_producer_is_required_and_replayed(tmp_path):
    request = fixture(tmp_path)
    producer = seal(
        dict(
            version="progressive-hip-case/v1",
            source=request["source"],
            scope="short",
            device="actual_HIP",
            status="PASS",
            requested=[
                "finite_update",
                "parameter_change",
                "full_restore",
                "deterministic_continuation",
                "generation",
                "native_reference_score",
                "cleanup",
            ],
            executed=[
                "finite_update",
                "parameter_change",
                "full_restore",
                "deterministic_continuation",
                "generation",
                "native_reference_score",
                "cleanup",
            ],
            measurements=dict(
                finite_update=True,
                parameter_change=True,
                full_restore=True,
                deterministic_continuation=True,
                native_reference_score=True,
                generation=True,
                cleanup=True,
            ),
        )
    )
    atomic_write(tmp_path / "hip.json", producer)
    row = {
        k: producer[k]
        for k in ("identity", "source", "scope", "device", "status", "requested", "executed")
    }
    row["receipt"] = reference(tmp_path, "hip.json")
    (tmp_path / "qualifications.json").unlink()
    atomic_write(tmp_path / "qualifications.json", [row])
    request = seal(
        dict(
            request,
            tables=dict(
                request["tables"], qualifications=[reference(tmp_path, "qualifications.json")]
            ),
            scope=dict(request["scope"], qualification_scopes=["short"]),
        )
    )
    request = bind_scope(tmp_path, request, json.loads((tmp_path / "roster.json").read_text()))
    build(tmp_path, tmp_path / "report.zip", request)
    assert verify(tmp_path / "report.zip")["status"] == "VERIFIED"

    def edit(docs):
        name = next(n for n in docs if n.startswith("producers/"))
        del docs[name]

    rewrite(tmp_path / "report.zip", edit)
    with pytest.raises(ValueError):
        verify(tmp_path / "report.zip")


def test_failed_terminal_export_keeps_first_and_collector_errors(tmp_path):
    from oeis_learn.generalization.progressive.report import terminal_request

    atomic_write(
        tmp_path / "result.json",
        seal(
            dict(
                version="discovery-result/v1",
                campaign=None,
                producer_error=dict(
                    type="RuntimeError", code="FIRST", producer="learner", local_traceback=None
                ),
                collector_errors=[
                    dict(type="OSError", code="SECOND", producer="collector", local_traceback=None)
                ],
            )
        ),
    )
    request = terminal_request(tmp_path, tmp_path)
    build(tmp_path, tmp_path / "failed.zip", request)
    result = verify(tmp_path / "failed.zip")
    assert result["status"] == "INCOMPLETE"
    assert result["producer_error"]["code"] == "FIRST"
    assert result["collector_errors"][0]["code"] == "SECOND"


def bind_scope(root, request, roster):
    from tests.unit.test_progressive_freeze import campaign

    qrefs = request["tables"].get("qualifications", [])
    qualifiers = []
    for ref in qrefs:
        qualifiers.extend(json.loads((root / ref["path"]).read_text()))
    scope = seal(
        dict(
            version="progressive-public-scope/v1",
            source=request["source"],
            scope=request["scope"],
            seeds=request["seeds"],
            design=request["design"],
            roster=digest(roster),
            qualification_producers={q["scope"]: q["identity"] for q in qualifiers},
        )
    )
    name = "scope-" + scope["identity"].split(":")[1] + ".json"
    if not (root / name).exists():
        atomic_write(root / name, scope)
    ref = reference(root, name)
    c = campaign()
    c["payload"]["references"].append(
        dict(name="public-scope", role="verifier", kind="policy", artifact=ref)
    )
    c = seal(c)
    return seal(dict(request, campaign=c["identity"], campaign_record=c, scope_record=ref))


def test_resealing_cannot_weaken_prospectively_frozen_scope(tmp_path):
    request = fixture(tmp_path)
    build(tmp_path, tmp_path / "report.zip", request)

    def edit(docs):
        m = json.loads(docs["manifest.json"])
        m["scope"]["required"] = []
        m["seeds"] = [1]
        docs["manifest.json"] = json.dumps(seal(m)).encode()

    rewrite(tmp_path / "report.zip", edit)
    with pytest.raises(ValueError, match="frozen authority"):
        verify(tmp_path / "report.zip")


def test_efficiency_uses_measured_costs_instead_of_supplied_summary():
    from tests.unit.test_progressive_decisions import fixture as panel
    from oeis_learn.generalization.progressive.verify import reduce_outcomes

    data, roster, design, _ = panel()
    for row in data:
        row["outcome"] = "success"
    design = dict(
        design,
        claim="efficiency",
        costs=dict(
            metric="total_active_us", scope="preparation_search_training", amortization="none"
        ),
    )
    assert reduce_outcomes(data, roster, [1, 2], design)[1] == "INCOMPLETE"
    assert (
        reduce_outcomes(data, roster, [1, 2], design, costs=dict(baseline=100, candidate=80))[1]
        == "CONFIRMED_EFFICIENCY"
    )
    assert (
        reduce_outcomes(data, roster, [1, 2], design, costs=dict(baseline=100, candidate=90))[1]
        == "NO_MATERIAL_GAIN"
    )
