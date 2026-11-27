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
    tmp_path, monkeypatch, capsys, status, paired=False, with_difficulty=False
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
    panel, exposure = cohort_fixture(tmp_path / "cohort")
    template = panel["payload"]["rows"][0]
    if with_difficulty:
        from oeis_learn.generalization.progressive.complexity import measure
        template['difficulty'] = measure(None)
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
    if paired:
        first=copy.deepcopy(panel);second=copy.deepcopy(panel);unrelated=copy.deepcopy(panel)
        # Two prospective source panels and an unrelated development panel.
        # The primary scope is their ordered union, not every published cohort.
        first['payload']['rows']=panel['payload']['rows'][:1]
        second['payload']['rows']=[dict(template,id=digest('retention'),groups=[digest('retention-group')],stratum='retention')]
        unrelated['payload']['rows']=[dict(template,id=digest('unrelated'),groups=[digest('unrelated-group')])]
        first,second,unrelated=map(seal,(first,second,unrelated))
        roster=[roster[0],dict(task=digest('retention'),group=digest('retention-group'),stratum='retention')]
        scope['roster']=digest(roster)
        atomic_write(tmp_path/'paired-scope.json',seal(scope))
        next(r for r in c['payload']['references'] if r['name']=='public-scope')['artifact']=reference(tmp_path,'paired-scope.json')
        c['payload']['references']=[r for r in c['payload']['references'] if r['name']!='public-panel']
        for name,value in [('primary',first),('retention',second),('unrelated',unrelated)]:
            atomic_write(tmp_path/(name+'.json'),value)
            c['payload']['references'].append(dict(name=name,role='custodian',kind='groups',artifact=reference(tmp_path,name+'.json')))
        gate=seal(dict(version='progressive-gate-registration/v1',panels=[first['identity'],second['identity']],
            source_slot='capability',target_slot='diversity',seeds=[1,2],design=digest(scope['design']),producers=[]))
        atomic_write(tmp_path/'paired-gate.json',gate)
        c['payload']['references'].append(dict(name='paired-gate',role='verifier',kind='policy',artifact=reference(tmp_path,'paired-gate.json')))
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

        metrics={"scores": reference(base, base / "scores.json")}
        if with_difficulty:
            from oeis_learn.generalization.progressive.cohorts import census
            atomic_write(base/'census.json',census(seal(panel),exposure=exposure))
            metrics['census']=reference(base,base/'census.json')
        return publish(
            base,
            operation="score",
            mechanical_status=status,
            reason="DIAGNOSTIC_NEGATIVE",
            campaign=c["identity"],
            metrics=metrics,
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
    if paired:
        # Explicit unresolved finite-choice boundary: accounting must retain the
        # original immutable task cap and debit, without inventing a selection.
        monkeypatch.setattr('oeis_learn.generalization.progressive.lifecycle.RunStore.resolved_plan',lambda *a:None)
    assert (
        main(SimpleNamespace(operation="report", run="run", output="report.zip"), tmp_path, None)
        == 0
    )
    capsys.readouterr()
    manifest = load(tmp_path / "report.zip.metadata/public-report.json")
    if paired:
        assert load(tmp_path/manifest['tables']['roster'][0]['path'])==roster
    assert manifest["campaign_record"] == c
    assert manifest["tables"]["accounting"]
    if with_difficulty:
        observations=[r for ref in manifest['tables']['observations'] for r in load(tmp_path/ref['path'])]
        measured=[r['data']['difficulty'] for r in observations if 'difficulty' in r['data']]
        assert measured and sum(v['work']['unavailable'] for v in measured[0].values())==len(roster)
        assert all(v['work']['maximum'] is None for v in measured[0].values())
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


def test_combined_primary_retention_scope_and_unselected_accounting(tmp_path,monkeypatch,capsys):
    test_run_report_verify_collects_owned_scores_and_resource_journal(tmp_path,monkeypatch,capsys,'PARTIAL',paired=True)


def test_report_preserves_unknown_complexity_without_zero_measurements(tmp_path,monkeypatch,capsys):
    test_run_report_verify_collects_owned_scores_and_resource_journal(
        tmp_path,monkeypatch,capsys,'PARTIAL',with_difficulty=True)


def test_two_seed_selector_reopens_private_scores_and_canonical_nested_publication(tmp_path):
    import base64
    import gzip
    import io
    import json
    import zipfile
    from oeis_learn.generalization.progressive.program_search_outcome import archive_bytes
    from oeis_learn.generalization.progressive.program_search_rehearsal_report import (
        encode,
        verify_transport,
    )
    from oeis_learn.generalization.progressive.program_search_selector import (
        SELECTOR_ARMS,
        selector_counts,
        selector_spending,
    )
    from oeis_learn.generalization.progressive.selector_replication import freeze_panel
    from oeis_learn.pipeline_validation.progressive_round028_selector import (
        rank_training_stream,
        selector_score,
    )
    from oeis_learn.pipeline_validation.progressive_selector_replication import run_repeat, reopen

    secret = "private-case-and-trace-a715"
    sequence = "81726354019283746501"
    body = "(program (return (int 81726354019283746501)))"
    reg = seal(
        dict(repeat_seconds=620, proposer="template_only", reference={"method": "template_only"})
    )
    roster = [
        dict(task=secret + s, stratum=s, group="private-family-" + s)
        for s in ("composition", "unseen_structure", "retention")
    ]
    witnesses = {
        r["task"]: dict(id=r["task"], indices=list(map(str, range(100))), values=[sequence] * 100)
        for r in roster
    }
    design = freeze_panel(reg["identity"], roster, reg["repeat_seconds"])

    def training(i):
        return dict(
            rows=[
                dict(
                    draw_identity="positive", values=[sequence if i == 0 else "1"] * 20, source=body
                ),
                dict(draw_identity="negative", source="(program (return $index))"),
            ],
            stream=seal(
                dict(
                    seeds={"training_families": design["seeds"][i]},
                    pairs=[{"positive_row": "positive", "negative_row": "negative"}],
                )
            ),
        )

    model = rank_training_stream(training(0), design["seeds"][0])
    cells = [
        dict(
            stratum=r["stratum"],
            group=i,
            counts={"SSSSSS" if r["stratum"] == "retention" else "FSFFSF": 1},
        )
        for i, r in enumerate(roster)
    ]
    registered = seal(
        dict(
            version="round028-selector-design/v1",
            registration=reg["identity"],
            roster=digest("development"),
            shuffle=digest("shuffle"),
            model=model["identity"],
            reference="retained_D",
            arms=list(SELECTOR_ARMS),
            seed=3,
            profile=dict(attempts=8, portfolio_us=60_000_000, fit_us=4_000_000, tokens=1024),
            sizes=[dict(stratum=c["stratum"], group=c["group"], cases=1) for c in cells],
            replication=dict(identity=design["identity"], seconds=reg["repeat_seconds"]),
        )
    )
    totals, pairs = selector_counts(cells, SELECTOR_ARMS)
    costs = dict(bank=1, live_incumbent=2, live_learned=2, reference=2, score=1)
    result = seal(
        dict(
            version="round028-selector-report/v1",
            design=registered,
            registration=reg["identity"],
            selections=digest("choices"),
            cells=cells,
            cases=3,
            arms=totals,
            pairs=pairs,
            spending=selector_spending(cells, comparable_cost=True, repeat_funded=True),
            costs=costs,
            elapsed_us=sum(costs.values()),
            search_us=4,
            gpu_dispatches=0,
        )
    )
    development = archive_bytes({"selector.json": encode(result)})

    class Role:
        last_wall_us = 12

        def __call__(self, action, data, **_):
            if action == "selector_score":
                return selector_score(data, None)
            mode = data["mode"]
            return seal(
                dict(
                    choices={mode: None},
                    comparable={mode: True},
                    complete=True,
                    seconds=60,
                    seed=data["seed"],
                    eligible=[],
                    bank=digest([]),
                    request=data["identity"],
                    proposer=data["proposer"],
                )
            )

    area = tmp_path / "private"
    area.mkdir()
    for _ in range(2):
        report, code = run_repeat(
            area,
            tmp_path,
            reg,
            design,
            roster,
            witnesses,
            model,
            lambda work, **kw: Role(),
            lambda *args: training(1),
            {"template_only": {}},
            {"template_only": (None, [])},
            development,
        )
        assert code == 0 and report["scientific_status"] == "INCOMPLETE"
        saved = load(area / "replication/result.json")
        fit_us = load(area / "training-1/ranker-result.json")["elapsed_us"]
        assert saved["charged_us"] == 2 * len(roster) * 3 * Role.last_wall_us + fit_us
        assert reopen(area / "replication", design, roster, witnesses) == (
            saved["models"],
            saved["cells"],
        )
    path = tmp_path / report["archive"]
    assert verify_transport(path) == report["receipt"]
    with zipfile.ZipFile(path) as z:
        parts = {n: z.read(n) for n in z.namelist()}
    with zipfile.ZipFile(io.BytesIO(parts["development.zip"])) as z:
        decoded = z.read("selector.json") + parts["replication.json"]
    for value in (secret, sequence, body):
        assert value.encode() not in decoded and base64.b64encode(value.encode()) not in decoded
    for modified in (
        dict(
            parts, **{"development.zip": parts["development.zip"] + gzip.compress(secret.encode())}
        ),
        dict(parts, **{"replication.json": encode(seal(dict(saved, charged_us=10**12)))}),
    ):
        with pytest.raises(ValueError):
            verify_transport(io.BytesIO(archive_bytes(modified)))
    with pytest.raises(ValueError):
        broken = seal(dict(saved, design=seal(dict(design, comparisons=7))))
        verify_transport(
            io.BytesIO(archive_bytes(dict(parts, **{"replication.json": encode(broken)})))
        )
