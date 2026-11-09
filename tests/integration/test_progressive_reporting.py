"""Canonical reporting collects original owned evidence on every terminal path."""

from types import SimpleNamespace

import pytest

from oeis_learn.generalization.progressive.contracts import (
    atomic_write,
    digest,
    load,
    reference,
    seal,
)


@pytest.mark.parametrize("status", ["COMPLETE", "PARTIAL", "FAILED"])
def test_run_report_verify_collects_owned_scores_and_resource_journal(
    tmp_path, monkeypatch, capsys, status
):
    from oeis_learn.pipeline_validation import progressive_roles as roles
    from oeis_learn.pipeline_validation.progressive_commands import main
    from tests.contract.test_progressive_bundle import fixture as bundle_fixture
    from tests.unit.test_progressive_cohorts import fixture as cohort_fixture

    request = bundle_fixture(tmp_path)
    c = request["campaign_record"]
    import copy

    scope = load(tmp_path / request["scope_record"]["path"])
    # A failed CPU case proves nested original-producer collection without
    # fabricating actual HIP evidence or granting device authority.
    case = seal(
        {
            "version": "progressive-hip-case/v1",
            "source": scope["source"],
            "scope": "diagnostic-cpu",
            "device": "cpu",
            "status": "FAIL",
            "requested": [],
            "executed": [],
            "measurements": {},
        }
    )
    atomic_write(tmp_path / "qualification/case.json", case)
    authority = seal(
        {
            "version": "generalization-qualification/v2",
            "producer": {"artifacts": [reference(tmp_path / "qualification", "case.json")]},
        }
    )
    atomic_write(tmp_path / "qualification/original.json", authority)
    c["payload"]["references"].append(
        {
            "name": "original-qualification",
            "role": "custodian",
            "kind": "qualification",
            "artifact": reference(tmp_path, "qualification/original.json"),
        }
    )
    scope["qualification_producers"] = {"diagnostic-cpu": case["identity"]}
    p = c["payload"]["tasks"][0]
    p.update(stage="capability", slot="capability", kind="evaluation", arm="static")
    roster = load(tmp_path / "roster.json")
    if status == "COMPLETE":
        from tests.unit.test_progressive_decisions import fixture as comparison_fixture

        _, original, _, _ = comparison_fixture()
        roster = [dict(r, task=digest(r["task"]), group=digest(r["group"])) for r in original]
        plans = []
        for seed in (1, 2):
            for arm in ("static", "refreshed"):
                plans.append(
                    dict(
                        copy.deepcopy(p),
                        id=arm + str(seed),
                        seed=seed,
                        arm=arm,
                        output=arm + str(seed),
                    )
                )
        c["payload"]["tasks"] = plans
        c["payload"]["caps"] = {k: sum(t["caps"][k] for t in plans) for k in p["caps"]}
        c["payload"]["graph"]["nodes"][0]["caps"] = dict(c["payload"]["caps"])
        c["payload"]["caps"] = {
            k: sum(n["caps"][k] for n in c["payload"]["graph"]["nodes"]) for k in p["caps"]
        }
    scope["scope"]["tasks"] = [digest(t["id"]) for t in c["payload"]["tasks"]]
    scope["roster"] = digest(roster)
    atomic_write(tmp_path / "accounted-scope.json", seal(scope))
    next(r for r in c["payload"]["references"] if r["name"] == "public-scope")["artifact"] = (
        reference(tmp_path, "accounted-scope.json")
    )
    panel, _ = cohort_fixture(tmp_path / "cohort")
    template = panel["payload"]["rows"][0]
    panel["payload"]["rows"] = [
        dict(template, id=r["task"], groups=[r["group"]], stratum=r["stratum"]) for r in roster
    ]
    atomic_write(tmp_path / "panel.json", seal(panel))
    c["payload"]["references"].append(
        {
            "name": "public-panel",
            "role": "custodian",
            "kind": "groups",
            "artifact": reference(tmp_path, "panel.json"),
        }
    )
    req = {
        "operation": "score",
        "artifacts": {"candidates": [], "score_truth": []},
        "image": digest("image"),
        "seconds": 1,
        "reservation": {k: 0 for k in p["caps"]},
    }
    atomic_write(tmp_path / "score-request.json", req)
    for plan in c["payload"]["tasks"]:
        plan["policy"] = reference(tmp_path, "score-request.json")["sha256"]
    # Replace the bundle fixture's opaque, unused policy with the actual
    # closed score request below.
    c["payload"]["references"] = [r for r in c["payload"]["references"] if r["name"] != "policy"]
    c["payload"]["references"].append(
        {
            "name": "score-request",
            "role": "custodian",
            "kind": "policy",
            "artifact": reference(tmp_path, "score-request.json"),
        }
    )
    # Raise only the predeclared metadata partition; no execution caps change.
    c["payload"]["storage"]["package_bytes"] = 1 << 20
    c["payload"]["storage"]["required_bytes"] = (
        c["payload"]["storage"]["checkpoint_bytes"]
        * (2 + c["payload"]["storage"]["milestone_pins"])
        + c["payload"]["storage"]["shard_bytes"]
        + (1 << 20)
    )
    c["payload"]["storage"]["available_bytes"] = c["payload"]["storage"]["required_bytes"]
    c = seal(c)
    atomic_write(tmp_path / "campaign.json", c)
    atomic_write(tmp_path / "producer.json", {"covered_capabilities": ["progressive-lifecycle"]})
    atomic_write(
        tmp_path / "runtime.json",
        {"payload": {"qualification": [reference(tmp_path, "producer.json")]}},
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

    def invoke(*a, task, journal, output, **kw):
        base = tmp_path / output / "role"
        from oeis_learn.generalization.progressive.storage import register_invocation

        register_invocation(tmp_path, c, task, journal.root, base.parent)
        score = seal(
            {"outcomes": {r["task"]: {"equal_budget": {"outcome": "failure"}} for r in roster}}
        )
        atomic_write(base / "scores.json", score)
        debit = {k: 0 for k in p["caps"]}
        debit["elapsed_us"] = 1
        e = journal.reserve(
            task["payload"]["id"], "observed", 0, debit, boot="test", monotonic_us=0
        )
        journal.commit(e["key"])
        from oeis_learn.pipeline_validation.progressive_evidence import publish

        if status == "FAILED":
            from oeis_learn.pipeline_validation.progressive_evidence import failure

            return failure(
                base,
                "score",
                RuntimeError("original diagnostic failure"),
                producer="installed-score",
            )

        return publish(
            base,
            operation="score",
            mechanical_status=status,
            reason="DIAGNOSTIC_NEGATIVE",
            campaign=c["identity"],
            metrics={"scores": reference(base, base / "scores.json")},
        )

    monkeypatch.setattr(roles, "invoke", invoke)
    assert main(
        SimpleNamespace(
            operation="run", campaign="campaign.json", acceptance="accepted", output="run"
        ),
        tmp_path,
        lambda a: object(),
    ) in {0, 1, 2}
    capsys.readouterr()
    assert (
        main(SimpleNamespace(operation="report", run="run", output="report.zip"), tmp_path, None)
        == 0
    )
    capsys.readouterr()
    manifest = load(tmp_path / "report.zip.metadata/public-report.json")
    assert manifest["campaign_record"] == c
    assert manifest["tables"]["accounting"]
    qualifications = load(tmp_path / manifest["tables"]["qualifications"][0]["path"])
    assert qualifications[0]["identity"] == case["identity"]
    assert qualifications[0]["receipt"] == reference(tmp_path, "qualification/case.json")
    accounting = load(tmp_path / manifest["tables"]["accounting"][0]["path"])
    assert accounting[0]["resources"]["spent"]["elapsed_us"] == 1
    assert accounting[0]["resources"]["caps"] == p["caps"]
    assert (
        main(
            SimpleNamespace(operation="verify-bundle", archive="report.zip", output="verified"),
            tmp_path,
            None,
        )
        == 0
    )
    result = load(tmp_path / "verified/verification.json")
    assert result["status"] == "VERIFIED" and result["decision"] == (
        "NO_MATERIAL_GAIN" if status == "COMPLETE" else "INCOMPLETE"
    )
    if status == "FAILED":
        import zipfile

        assert result["producer_error"]["producer"] == "installed-score"
        trace = result["producer_error"]["local_traceback"]
        with zipfile.ZipFile(tmp_path / "report.zip") as archive:
            assert b"original diagnostic failure" in archive.read(
                "errors/" + trace["sha256"][7:] + ".txt"
            )
    if status == "COMPLETE":
        from oeis_learn.generalization.progressive.report import build
        from oeis_learn.generalization.progressive.verify import verify

        accounting[0]["resources"]["caps"]["tokens"] += 1
        atomic_write(tmp_path / "altered-accounting.json", accounting)
        manifest["tables"]["accounting"] = [reference(tmp_path, "altered-accounting.json")]
        build(tmp_path, tmp_path / "altered.zip", seal(manifest))
        with pytest.raises(ValueError, match="resource accounting"):
            verify(tmp_path / "altered.zip")
