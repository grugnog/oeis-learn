"""A single grant prices paired neural inputs before exposing the scorer to the suffix."""

from copy import deepcopy

import pytest

from oeis_learn.generalization.progressive.contracts import digest, seal
from tests.integration.test_round029_isolated_control_prices import api, panel


def inputs():
    rows, visible, truth = panel()
    for i, (p, t) in enumerate(zip(visible, truth, strict=True)):
        t["values"] = [str(int(x) + i) for x in t["indices"]]
        p["values"] = t["values"][:20]
    models = {
        m: dict(
            method=m,
            seed=29,
            origin=digest(["origin", m]),
            recipe=dict(method=m),
            weight=dict(path="inputs/" + m + ".pt", bytes=4, sha256=digest(["weights", m])),
        )
        for m in ("00", "D")
    }
    tensors = {m: digest(["tensors", m]) for m in models}
    return rows, visible, truth, models, tensors


def test_all_six_probes_are_fixed_before_search_and_use_matching_controls():
    rows, visible, _, models, tensors = inputs()
    pairs = api["conditioning_probes"](digest("plan"), rows, visible, models, tensors)
    assert list(pairs) == ["ordinary-common", "ordinary-full", "D-full"]
    for name, modes in pairs.items():
        a, b = modes["true"], modes["shuffled"]
        assert a["model"] == b["model"] and a["look"] == b["look"]
        assert a["look"]["attempts"] == (1 if name.endswith("common") else 8)
        assert a["inference_seed"] == b["inference_seed"] and a["identity"] != b["identity"]
        assert all(a["donors"][r["task"]] == r["task"] != b["donors"][r["task"]] for r in rows)
    visible[1]["values"] = visible[0]["values"]
    with pytest.raises(ValueError, match="differing"):
        api["conditioning_probes"](digest("plan"), rows, visible, models, tensors)


def test_conditioning_cli_requires_exact_bank_receipt_and_fresh_zip(tmp_path):
    with pytest.raises(SystemExit):
        api["cli"](["a" * 40, "75", "conditioning", "--output", str(tmp_path / "test.zip")])
    assert not list(tmp_path.iterdir())


def test_paired_installed_paths_bind_all_candidates_scores_and_admissions(tmp_path, monkeypatch):
    import os
    import torch
    from oeis_learn.decoder import bigint_generation
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.baselines import load_baseline
    from oeis_learn.generalization.progressive.contracts import load, reference
    from tests.integration.test_round029_isolated_control_prices import EmptyClient

    rows, visible, truth, models, _ = inputs()
    recipe = resolve({"version": "bigint-bootstrap/v1"})
    enc, dec, _ = load_baseline("random", recipe, seed=29)
    work = tmp_path / "generator"
    (work / "inputs").mkdir(parents=True)
    (work / "output").mkdir()
    path = work / "inputs/00.pt"
    torch.save(dict(encoder=enc.state_dict(), decoder=dec.state_dict()), path)
    models["00"].update(recipe=recipe, weight=reference(work, path))
    os.link(path, work / "inputs/D.pt")
    models["D"].update(recipe=recipe, weight=reference(work, work / "inputs/D.pt"))
    model_req = seal(dict(phase="models"))
    raw = api["conditioning_role"](
        work, model_req, dict(models=models), EmptyClient(), device="cpu"
    )
    tensors = load(work / "output" / raw["artifact"]["path"])["tensors"]
    assert len(tensors) == 2 and tensors["D"] == tensors["00"]
    # A separate writable output represents the next isolated role.
    search_work = tmp_path / "search"
    (search_work / "inputs").mkdir(parents=True)
    (search_work / "output").mkdir()
    os.link(path, search_work / "inputs/00.pt")
    probes = api["conditioning_probes"](digest("frozen"), rows, visible, models, tensors)[
        "ordinary-common"
    ]
    monkeypatch.setattr(
        bigint_generation,
        "sample_batch",
        lambda decoder, reqs: [
            dict(
                tokens=[],
                source="",
                complete=False,
                generation_batch_size=1,
                generation_seconds=0.0,
            )
            for _ in reqs
        ],
    )
    request = seal(dict(phase="search", probes=probes, order=["true", "shuffled"]))
    payload = dict(prompts=visible, model=models["00"])
    artifact = api["conditioning_role"](search_work, request, payload, EmptyClient(), device="cpu")[
        "artifact"
    ]
    emitted = load(search_work / "output" / artifact["path"])
    evidence = api["check_paired_search"](emitted, search_work / "output", request, payload)
    assert evidence["bank_programs"] == 0
    score_work = tmp_path / "score"
    (score_work / "inputs").mkdir(parents=True)
    (score_work / "output").mkdir()
    children, generations = {}, {}
    for mode, item in evidence["modes"].items():
        raw = search_work / "output" / item["artifact"]["path"]
        dest = score_work / "inputs" / (mode + ".json")
        os.link(raw, dest)
        generations[mode] = load(raw)
        children[mode] = dict(artifact=reference(score_work, dest), case_us=item["case_us"])
    closed = seal(dict(phase="score", probes=probes, order=["true", "shuffled"], search=children))
    payload = dict(truth=truth)
    ref = api["conditioning_role"](score_work, closed, payload, EmptyClient(), device="cpu")[
        "artifact"
    ]
    scored = load(score_work / "output" / ref["path"])
    reduced = api["check_paired_scores"](
        scored, score_work / "output", closed, payload, generations
    )
    assert sum(reduced["paired_selected"].values()) == 8
    assert all(
        sum(c["empty"] for c in m["bank_cells"].values()) == 8 for m in reduced["modes"].values()
    )
    wrong = deepcopy(scored)
    wrong["modes"]["true"]["selected"] = scored["modes"]["shuffled"]["selected"]
    with pytest.raises(ValueError, match="unexpected paired scoring"):
        api["check_paired_scores"](wrong, score_work / "output", closed, payload, generations)
