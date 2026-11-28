"""Disposable CPU controls use separate search and scoring stages on a synthetic panel."""

from copy import deepcopy
import json
from pathlib import Path
import runpy
import shutil

import pytest

from oeis_learn.fresh.bigint_config import resolve
from oeis_learn.generalization.progressive.cohorts import STRATA
from oeis_learn.generalization.progressive.contracts import digest, load, reference, seal
from oeis_learn.generalization.progressive.graph import velocity_look
from oeis_learn.pipeline_validation.progressive_evaluation import velocity_probe
from oeis_learn.synthesis.contracts import ResourceProfile

OPERATOR = (
    Path(__file__).resolve().parents[2]
    / "specs/016-measure-learning-velocity/validation/qualify_continuation.py"
)
api = runpy.run_path(str(OPERATOR))


class EmptyClient:
    limits = ResourceProfile()

    def fit(self, source, visible, **kwargs):
        assert len(visible.indices) == len(visible.values) == 20
        return dict(complete=True, outcome="ok", admitted=[])

    def runtime_identity(self, profile):
        return digest(profile)


class SelectedClient(EmptyClient):
    def fit(self, source, visible, **kwargs):
        from oeis_learn.synthesis.admission import source_identity

        source_hash = source_identity(source)
        admission = dict(
            source=source,
            transformed_source=source,
            source_sha256=source_hash,
            original_source_sha256=source_hash,
            assignment={},
            verification=dict(prompt_sha256=visible.sha256),
        )
        return dict(complete=True, outcome="ok", admitted=[admission])

    def execute(self, source, indices, *, deadline):
        from oeis_learn.synthesis.reference import evaluate

        result = evaluate(source, indices, deadline=deadline)
        return dict(values=list(result.values), outcome=result.outcome)


def panel():
    members = [
        dict(task=f"private/{s}{i}", stratum=s, group=f"private-group/{s}{i}")
        for s in STRATA
        for i in range(2)
    ]
    truth = [
        dict(
            id=m["task"],
            indices=list(map(str, range(100))),
            values=[str(i + 2) for i in range(100)],
        )
        for m in members
    ]
    visible = [dict(id=t["id"], indices=t["indices"][:20], values=t["values"][:20]) for t in truth]
    return members, visible, truth


@pytest.mark.parametrize(
    "method,selected",
    [
        ("template", False),
        ("polynomial", False),
        ("linear_recurrence", False),
        ("polynomial", True),
    ],
)
def test_real_sealed_search_and_score_are_joined_by_original_candidate(tmp_path, method, selected):
    client = SelectedClient() if selected else EmptyClient()
    members, visible, truth = panel()
    look = velocity_look(
        registration=digest("fixture"),
        method=method,
        seed=29,
        look=4_000_000,
        profile="full",
        development=members,
        retention=[],
    )
    probe = velocity_probe(look, visible, mode="true", inference_seed=57)
    recipe = resolve({"version": "bigint-bootstrap/v1"})
    request = dict(probe=probe, method=method)
    work = tmp_path / "search"
    (work / "output").mkdir(parents=True)
    search = api["control_role"](
        work, dict(request, phase="search"), dict(rows=visible, recipe=recipe), client
    )
    assert search["artifact"]["path"] == "generated/candidates.json"
    gen = load(work / "output" / search["artifact"]["path"])
    scorework = tmp_path / "score"
    (scorework / "output").mkdir(parents=True)
    (scorework / "inputs").mkdir()
    dest = scorework / "inputs/candidates.json"
    shutil.copyfile(work / "output" / search["artifact"]["path"], dest)
    ref = reference(scorework, dest)
    scoring = api["control_role"](
        scorework, dict(request, phase="score", candidates=ref), dict(rows=truth), client
    )
    scores = load(scorework / "output" / scoring["artifact"]["path"])
    evidence = api["control_case_costs"](probe, visible, gen, ref["sha256"], scores)
    assert all(
        c["cases"] == 2
        and c["empty"] == (0 if selected else 2)
        and c["max_us"] >= c["mean_us"] >= 0
        for c in evidence.values()
    )
    assert "private/" not in json.dumps(evidence)
    with pytest.raises(ValueError):
        api["control_case_costs"](probe, visible, gen, digest("wrong parent"), scores)
    bad = deepcopy(gen)
    bad["clock"]["clocks"][0]["started"] = 0
    bad["clock"] = seal(bad["clock"])
    with pytest.raises(ValueError):
        api["control_case_costs"](probe, visible, seal(bad), ref["sha256"], scores)
    with pytest.raises(ValueError):
        api["control_role"](
            tmp_path, dict(request, phase="search"), dict(rows=truth, recipe=recipe), client
        )


def test_prices_each_endpoint_once_and_keeps_unmeasured_work_explicit():
    cells = {s: dict(cases=2, mean_us=i + 1, max_us=2 * (i + 1)) for i, s in enumerate(STRATA)}
    results = {
        m: dict(cells=cells, overhead_us=dict(search=11, score=13))
        for m in ("template", "polynomial", "linear_recurrence")
    }
    ops = {"D.full.search." + s: 3 for s in STRATA}
    ops.update(
        {"D.full.search.retention": 2, "D.full.search.startup": 1, "conventional_cpu_controls": 1}
    )
    wave = dict(
        operations=ops,
        missing=["conventional_cpu_controls", "matched_full_bank_scoring"],
        known_expected_us=7,
        known_tail_us=10,
    )
    cost = api["control_cost_forecast"](results, wave)
    # 3 cases from each of four strata plus two retention cases at the slowest stratum.
    assert cost["control_expected_us"] == 3 * (24 + 30 + 8)
    assert cost["control_tail_us"] == 3 * (24 + 60 + 16)
    assert cost["known_expected_us"] == 7 + cost["control_expected_us"] and cost["missing"] == [
        "matched_full_bank_scoring"
    ]
    assert cost["expected_us"] is None and cost["tail_us"] is None
    assert api["control_cost_forecast"]({"template": results["template"]}, wave) is None


def test_exposed_roster_fallback_reopens_only_sealed_visible_and_truth_channels(
    tmp_path, monkeypatch
):
    from oeis_learn.generalization.progressive.contracts import atomic_write
    from oeis_learn.pipeline_validation import progressive_velocity_qualification as qual

    members = {f"case/{s}/{i}": (s, f"family/{s}/{i}") for s in STRATA for i in range(32)}
    refs = []
    for ordinal, key in enumerate(members):
        for channel, role, kind, n in (
            ("visible", "generator", "visible_prompt", 20),
            ("truth", "scorer", "development_truth", 100),
        ):
            values = [str(ordinal + i) for i in range(n)]
            data = dict(id=key, indices=list(map(str, range(n))), values=values)
            path = tmp_path / f"{ordinal}-{channel}"
            atomic_write(
                path, seal(dict(version="discovery-channel/v1", kind=kind, role=role, data=data))
            )
            refs.append(
                dict(
                    name=f"private/channels/{channel}-{digest(key)[7:]}",
                    role=role,
                    kind="prepared",
                    artifact=reference(tmp_path, path),
                )
            )
    scope = api["exposed_control_inputs"].__globals__
    campaign = seal(dict(payload=dict(references=refs)))
    monkeypatch.setitem(scope, "frozen_control_study", lambda root: ({}, campaign, members))
    monkeypatch.setitem(
        scope, "read_history", lambda root: dict(status="COMPLETE", coverage={"two_cohorts": 128})
    )
    monkeypatch.setattr(qual, "_original_profile", lambda root: {"fixture": True})
    rows, prompts, truth, _, origin = api["exposed_control_inputs"](tmp_path)
    assert len(rows) == len({m["group"] for m in rows}) == 8 and set(
        m["stratum"] for m in rows
    ) == set(STRATA)
    assert len(origin) == 71 and [p["id"] for p in prompts] == [t["id"] for t in truth]
    key = rows[0]["task"]
    target = next(r for r in refs if r["name"] == "private/channels/truth-" + digest(key)[7:])
    (tmp_path / target["artifact"]["path"]).write_bytes(b"{}")
    with pytest.raises(ValueError):
        api["exposed_control_inputs"](tmp_path)


def test_missing_verified_checkpoint_blocks_the_cpu_grant(tmp_path):
    def action(name, **kwargs):
        assert name == "preflight"
        return dict(inventory=dict(gpu_dispatches=0, prices={}))

    with pytest.raises(ValueError, match="before issuing"):
        api["donor_isolated_controls"](action, tmp_path, tmp_path, "a" * 40, "15")
    assert list(tmp_path.iterdir()) == []
