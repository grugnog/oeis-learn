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


def test_device_evidence_requires_corroboration_and_distinguishes_clocks(tmp_path, monkeypatch):
    import json
    from oeis_learn.generalization.progressive.contracts import atomic_write
    from oeis_learn.pipeline_validation.progressive_generator_probe import usage_summary
    from oeis_learn.tracking import pipeline_profile

    render = "/dev/dri/renderD129"
    sensor = tmp_path / "pci-device"
    monkeypatch.setattr(
        pipeline_profile, "gpu_sensor_path", lambda node: sensor if node == render else None
    )
    exe = api["conditioning_device_evidence"]
    area = tmp_path / "telemetry"
    area.mkdir()
    (area / "samples.jsonl").write_text("")
    atomic_write(area / "capability.json", dict(gpu_sensor=str(sensor)))
    atomic_write(tmp_path / "gpu-lease.json", dict(elapsed_us=40))
    image = digest("image")
    argv = [
        "/usr/bin/podman",
        "run",
        "--device",
        "/dev/kfd",
        "--device",
        render,
        "--entrypoint",
        "env",
        image,
    ]

    def command(args):
        (tmp_path / "command.json").write_text(json.dumps(dict(argv=args)))

    command(argv)
    record = dict(Image=image, HostConfig=dict(Devices=[]))
    producer = dict(
        device="hip",
        measured_us=20,
        worker_us=35,
        peak_device_bytes=10,
        telemetry=usage_summary(area, 20),
    )
    assert exe(tmp_path, record, producer, 50, render)["lease_us"] == 40
    command(argv[:5] + ["/dev/dri/renderD130"] + argv[6:])
    with pytest.raises(ValueError, match="device_argv"):
        exe(tmp_path, record, producer, 50, render)
    command(argv)
    (area / "capability.json").write_text(json.dumps(dict(gpu_sensor="/somewhere-else")))
    with pytest.raises(ValueError, match="device_observation"):
        exe(tmp_path, record, producer, 50, render)
    paths = [dict(PathOnHost=p, PathInContainer=p) for p in ("/dev/kfd", render)]
    record["HostConfig"]["Devices"] = paths
    with pytest.raises(ValueError, match="lease_clock"):
        exe(tmp_path, record, dict(producer, worker_us=41), 50, render)
    record["HostConfig"]["Devices"] = paths + paths[:1]
    with pytest.raises(ValueError, match="device_mapping"):
        exe(tmp_path, record, producer, 50, render)


def test_conditioning_retains_paid_evidence_if_final_verification_fails(tmp_path, monkeypatch):
    report = dict(
        scope=api["CONDITIONING"],
        status="COMPLETE",
        research_run_requested=False,
        scientific_verdict=None,
        grant=digest("grant"),
        elapsed_us=1_000_000,
        errors={},
        files={"models.json": {"receipt": digest("model")}},
    )
    cli = api["cli"]
    monkeypatch.setitem(cli.__globals__, "main", lambda *args: report)
    target = tmp_path / "result.zip"
    assert (
        cli(
            [
                "a" * 40,
                "75",
                "conditioning",
                "--bank-receipt",
                digest("bank"),
                "--output",
                str(target),
            ]
        )
        == 1
    )
    summary, files = api["diagnostic_zip_members"](target, api["CONDITIONING"])
    assert summary["status"] == "UNVERIFIED" and summary["errors"] == {"publication": "KeyError"}
    assert summary["grant"] == digest("grant") and summary["elapsed_us"] == 1_000_000
    assert summary["receipt"] is None and files == report["files"]


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


@pytest.mark.parametrize("recover_paid", [False, True])
def test_one_grant_resumes_one_search_and_seals_all_groups_before_scoring(
    tmp_path, monkeypatch, recover_paid
):
    import os
    import time
    import zipfile
    import torch
    from types import SimpleNamespace
    from oeis_learn.decoder import bigint_generation
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.baselines import load_baseline
    from oeis_learn.generalization.progressive.contracts import atomic_write, load, reference
    from oeis_learn.pipeline_validation.progressive_generator_probe import usage_summary
    from oeis_learn.pipeline_validation import progressive_round028_rehearsal
    from tests.integration.test_round029_isolated_control_prices import EmptyClient, OPERATOR

    fn = api["conditioning_matrix"]
    ns = fn.__globals__
    folder = tmp_path / "state"
    folder.mkdir()
    source = OPERATOR.parents[3]
    wheel = tmp_path / "runtime.whl"
    with zipfile.ZipFile(wheel, "w") as z:
        z.writestr("fixture.py", "pass")
    rt = dict(
        image=digest("image"),
        wheel=digest("wheel"),
        source=digest("src"),
        physical=digest("hip"),
        path=wheel,
    )
    host = SimpleNamespace(policy=dict(free_disk_floor_bytes=0, render_node="/dev/dri/renderD128"))
    monkeypatch.setattr(
        progressive_round028_rehearsal, "physical_scope", lambda h, phase, **kw: digest(phase)
    )
    rows, visible, truth, models, _ = inputs()
    recipe = resolve({"version": "bigint-bootstrap/v1"})
    enc, dec, _ = load_baseline("random", recipe, seed=29)
    weight = tmp_path / "weights.pt"
    torch.save(dict(encoder=enc.state_dict(), decoder=dec.state_dict()), weight)
    for method, m in models.items():
        m.update(
            recipe=recipe, weight=dict(reference(tmp_path, weight), path="inputs/" + method + ".pt")
        )
    monkeypatch.setitem(
        ns, "paired_model_inputs", lambda *args: (models, {m: weight for m in models})
    )
    monkeypatch.setitem(ns, "previous_bank_prices", lambda *args: dict(receipt=digest("paid-bank")))
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
    monkeypatch.setitem(
        ns, "control_capacity", lambda *a, **kw: [dict(deficit_bytes=0, deficit_inodes=0)]
    )
    calls = []
    fail = set() if recover_paid else {"ordinary-full"}
    from oeis_learn.tracking import pipeline_profile

    sensor = tmp_path / "fixture-pci-device"
    monkeypatch.setattr(pipeline_profile, "gpu_sensor_path", lambda _: sensor)

    def stage(_host, tree, image, out, argv, *, gpu, seconds, readonly_inputs, record_lease=False):
        began = time.monotonic_ns()
        local = tmp_path / ("local-" + str(len(calls)))
        (local / "inputs").mkdir(parents=True)
        mounted = {dest: path for path, dest in readonly_inputs}
        req = load(mounted["/work/inputs/request.json"])
        data = load(mounted["/work/inputs/data.json"])
        group, phase = req["method"], req["phase"]
        calls.append((group, phase))
        assert (phase == "search") == gpu == record_lease
        assert not any(path.is_dir() for path in mounted.values())
        for dest, path in mounted.items():
            if dest.endswith((".pt", "true.json", "shuffled.json")):
                os.link(path, local / dest.removeprefix("/work/"))
        atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
        if gpu and group in fail:
            fail.clear()
            atomic_write(out / "gpu-lease.json", dict(elapsed_us=1))
            raise ValueError("synthetic interruption with a closed lease")
        if phase == "search":
            assert set(data) == {"prompts", "model"} and all(
                len(r["values"]) == 20 for r in data["prompts"]
            )
        if phase == "score":
            assert (
                req["barrier"]
                and len(list(folder.glob("preparation/paired-conditioning/*/sealed-searches.json")))
                == 1
            )
            assert set(data) == {"truth"} and all(len(r["values"]) == 100 for r in data["truth"])
            assert not any(k.endswith(".pt") for k in mounted)
        (local / "output").mkdir()
        start = time.monotonic_ns()
        result = api["conditioning_role"](local, req, data, EmptyClient(), device="cpu")
        import shutil

        for p in (local / "output").iterdir():
            shutil.move(p, out / p.name)
        us = max(1, (time.monotonic_ns() - start) // 1000)
        if gpu:
            (out / "telemetry").mkdir()
            (out / "telemetry/samples.jsonl").write_text("")
            atomic_write(out / "telemetry/capability.json", dict(gpu_sensor=str(sensor)))
            atomic_write(
                out / "command.json",
                dict(
                    argv=[
                        "/usr/bin/podman",
                        "run",
                        "--device",
                        "/dev/kfd",
                        "--device",
                        host.policy["render_node"],
                        "--entrypoint",
                        "env",
                        image,
                        "python",
                    ]
                ),
            )
            result.update(
                device="hip",
                measured_us=us,
                peak_device_bytes=1,
                telemetry=usage_summary(out / "telemetry", us),
            )
        atomic_write(
            out / "producer.json",
            seal(dict(request=req["identity"], native=digest("native"), worker_us=us, **result)),
        )
        mounts = [
            dict(Type="bind", Source=str(p), Destination=k, RW=False) for k, p in mounted.items()
        ]
        mounts.append(dict(Type="bind", Source=str(out), Destination="/work/output", RW=True))
        devices = (
            [
                dict(PathOnHost=k, PathInContainer=k)
                for k in ("/dev/kfd", host.policy["render_node"])
            ]
            if gpu and not recover_paid
            else []
        )
        atomic_write(
            out / "container-inspect.json",
            [
                dict(
                    Image=image,
                    Mounts=mounts,
                    HostConfig=dict(
                        NetworkMode="none", ReadonlyRootfs=True, Privileged=False, Devices=devices
                    ),
                )
            ],
        )
        atomic_write(
            out / "process.json", dict(outcome="COMPLETE", returncode=0, cleanup_failure=False)
        )
        if gpu:
            atomic_write(
                out / "gpu-lease.json", dict(elapsed_us=(time.monotonic_ns() - began) // 1000)
            )

    args = (
        tmp_path,
        source,
        folder,
        "a" * 40,
        75,
        rt,
        (rows, visible, truth, {}, digest("panel")),
        dict(checkpoint_receipt=digest("checkpoint")),
        [],
        host,
        stage,
        digest("bank"),
    )
    if recover_paid:

        def legacy_check(*args):
            raise ValueError("original HIP devices, lease, samples, or measured clocks differ")

        with monkeypatch.context() as legacy:
            legacy.setitem(ns, "conditioning_device_evidence", legacy_check)
            partial = fn(*args)
        assert partial["status"] == "PARTIAL" and len(partial["errors"]) == 3
        assert partial["hip_attempts"] == 3 and all(p != "score" for _, p in calls)
        recovery = dict(grant=partial["grant"], reader="b" * 40)
        # Prove a host with no authority for a new grant can finish the original one.
        from oeis_learn.pipeline_validation import (
            progressive_velocity_qualification as qualification,
        )

        monkeypatch.setattr(
            qualification,
            "authorize_qualification",
            lambda *a: pytest.fail("recovery attempted a new grant"),
        )
        assert fn(*args, recovery=dict(recovery, grant=digest("other")))["status"] == "BLOCKED"
        area = folder / "preparation/paired-conditioning" / partial["grant"][7:]
        old = {p: p.read_bytes() for p in area.glob("operations/*/first-error.json")}
        assert len(old) == 3
        sensor_file = next(iter(old)).parent / "attempt-0000/telemetry/capability.json"
        saved_sensor = sensor_file.read_bytes()
        sensor_file.write_text('{"gpu_sensor":"another-device"}')
        refused = fn(*args, recovery=recovery)
        assert refused["status"] == "PARTIAL" and len(refused["errors"]) == 1
        (problem,) = refused["errors"]
        diagnostics = refused["files"]["recovery.json"]
        assert diagnostics["failure_checks"][problem] == ["device_observation"]
        assert diagnostics["observations"][problem]["reported_devices"] == 0
        assert not diagnostics["observations"][problem]["sole_render_sensor_matches"]
        assert refused["hip_attempts"] == 3 and len(calls) == 4
        sensor_file.write_bytes(saved_sensor)
        full = fn(*args, recovery=recovery)
        assert full["status"] == "COMPLETE", full
        assert full["grant"] == partial["grant"] and full["elapsed_us"] > partial["elapsed_us"]
        assert full["hip_attempts"] == full["gpu_dispatches"] == 3
        assert full["recorded_lease_us"] == partial["recorded_lease_us"]
        assert all(p.read_bytes() == v for p, v in old.items())
        assert [p for _, p in calls] == [
            "models",
            "search",
            "search",
            "search",
            "score",
            "score",
            "score",
        ]
        details = full["files"]["recovery.json"]
        assert details["additional_hip_attempts"] == 0 and len(details["groups"]) == 3
        dest = tmp_path / "recovered.zip"
        api["write_readiness_zip"](dest, full)
        assert api["verify_conditioning_zip"](dest)["status"] == "COMPLETE"
        wrong = deepcopy(full)
        wrong["files"]["recovery.json"]["original_hip_attempts"] += 1
        api["write_readiness_zip"](tmp_path / "wrong-recovery.zip", wrong)
        with pytest.raises(ValueError, match="recovery grant"):
            api["verify_conditioning_zip"](tmp_path / "wrong-recovery.zip")
        again = fn(*args, recovery=recovery)
        assert again["receipt"] == full["receipt"] and again["charged_work"] == full["charged_work"]
        assert len(calls) == 7
        return
    partial = fn(*args)
    assert partial["status"] == "PARTIAL" and partial["errors"] == {
        "search-ordinary-full": "ValueError"
    }, partial
    assert len(calls) == 4 and all(phase != "score" for _, phase in calls)
    full = fn(*args)
    assert full["status"] == "COMPLETE" and not full["errors"], full
    assert (
        full["grant"] == partial["grant"]
        and full["hip_attempts"] == full["gpu_dispatches"] == 4
        and len(calls) == 8
    )
    assert (
        full["charged_work"]["tokens"] == 0
        and full["elapsed_us"] + full["remaining_us"] == 75 * 60_000_000
    )
    report = deepcopy(full)
    dest = tmp_path / "conditioning.zip"
    api["write_readiness_zip"](dest, report)
    assert api["verify_conditioning_zip"](dest)["status"] == "COMPLETE"
    again = fn(*args)
    assert (
        again["receipt"] == full["receipt"]
        and again["charged_work"] == full["charged_work"]
        and len(calls) == 8
    )
    import json

    assert "private/" not in json.dumps(again)
    bad = deepcopy(full)
    bad["files"]["forecast.json"]["successful_role_us"] += 1
    api["write_readiness_zip"](tmp_path / "wrong.zip", bad)
    with pytest.raises(ValueError, match="projection"):
        api["verify_conditioning_zip"](tmp_path / "wrong.zip")
    swapped = deepcopy(full)
    swapped["receipt"] = digest("foreign")
    api["write_readiness_zip"](tmp_path / "swapped.zip", swapped)
    with pytest.raises(ValueError, match="paired receipt"):
        api["verify_conditioning_zip"](tmp_path / "swapped.zip")


def test_original_ordinary_and_retained_model_are_bound_to_their_own_profiles(
    tmp_path, monkeypatch
):
    import json
    from oeis_learn.generalization.progressive.contracts import reference
    from oeis_learn.pipeline_validation import (
        progressive_velocity_census,
        progressive_velocity_donor,
    )

    _, _, _, models, _ = inputs()
    base = dict(recipe="D")
    profile = dict(recipe="ordinary")
    for name in ("D", "00"):
        (tmp_path / (name + ".pt")).write_text(name)
    recipe_file = tmp_path / "recipe.json"
    recipe_file.write_text(json.dumps(base))
    refs = {m: reference(tmp_path, tmp_path / (m + ".pt")) for m in models}
    freeze = dict(seeds=dict(initialization=13), reference=refs["D"])
    stream = dict(identity=digest("stream"))
    monkeypatch.setattr(
        progressive_velocity_census,
        "training_source",
        lambda _: (freeze, dict(profile=profile, stream=stream), []),
    )
    parent = dict(weights=refs["00"], checkpoint=dict(sha256=digest("full-state")))
    monkeypatch.setattr(
        progressive_velocity_donor,
        "donated_ordinary",
        lambda root: (dict(parent=parent), [(root / "00.pt", "/work/inputs/previous/00.pt")]),
    )
    fn = api["paired_model_inputs"]
    monkeypatch.setitem(
        fn.__globals__,
        "frozen_control_study",
        lambda _: (
            {},
            dict(
                payload=dict(
                    references=[
                        dict(
                            name="recipe",
                            role="custodian",
                            kind="recipe",
                            artifact=reference(tmp_path, recipe_file),
                        )
                    ]
                )
            ),
            {},
        ),
    )
    ordinary = dict(
        method="00",
        look=4_000_000,
        inference=True,
        weights=refs["00"]["sha256"],
        seeds=freeze["seeds"],
        lineage=stream["identity"],
        full_state=digest("full-state"),
        registration=digest("study"),
    )
    incumbent = dict(
        method="D",
        look=None,
        inference=True,
        weights=refs["D"]["sha256"],
        lineage=digest("D"),
        recipe=digest(base),
        registration=digest("study"),
    )
    registry, paths = fn(tmp_path, [ordinary, incumbent])
    assert (
        registry["00"]["recipe"] == profile
        and registry["D"]["recipe"] == base
        and registry["00"]["seed"] == 13
    )
    assert all(paths[k] == tmp_path / (k + ".pt") for k in paths)
    with pytest.raises(ValueError, match="qualified original"):
        fn(tmp_path, [ordinary, dict(incumbent, recipe=digest("substitute"))])


def test_differing_donor_replacement_uses_only_originally_exposed_families(tmp_path, monkeypatch):
    from oeis_learn.generalization.progressive.contracts import atomic_write, reference

    rows, prompts, truth, _, _ = inputs()
    prompts[1]["values"] = prompts[0]["values"]
    truth[1]["values"] = truth[0]["values"]
    extra = dict(task="replacement", stratum=rows[0]["stratum"], group="additional-family")
    roster = {r["task"]: (r["stratum"], r["group"]) for r in [*rows, extra]}
    targets = {r["id"]: r for r in truth}
    targets["replacement"] = dict(
        id="replacement",
        indices=[str(i) for i in range(100)],
        values=[str(23 * i) for i in range(100)],
    )
    refs = []
    for ident, row in targets.items():
        for kind, role, label, size in (
            ("visible", "generator", "visible_prompt", 20),
            ("truth", "scorer", "development_truth", 100),
        ):
            data = dict(id=ident, indices=row["indices"][:size], values=row["values"][:size])
            path = tmp_path / f"{kind}-{digest(ident)[7:]}"
            atomic_write(
                path, seal(dict(version="discovery-channel/v1", role=role, kind=label, data=data))
            )
            refs.append(
                dict(
                    name="private/channels/" + path.name,
                    role=role,
                    kind="prepared",
                    artifact=reference(tmp_path, path),
                )
            )
    fn = api["conditioning_panel"]
    monkeypatch.setitem(
        fn.__globals__,
        "frozen_control_study",
        lambda _: ({}, dict(payload=dict(references=refs)), roster),
    )
    monkeypatch.setitem(fn.__globals__, "read_history", lambda _: dict(status="PARTIAL"))
    with pytest.raises(ValueError, match="original exposure"):
        fn(tmp_path, (rows, prompts, truth))
    monkeypatch.setitem(fn.__globals__, "read_history", lambda _: dict(status="COMPLETE"))
    selected, visible, full = fn(tmp_path, (rows, prompts, truth))
    assert selected[0] == rows[0] and selected[1] == extra and selected[2:] == rows[2:]
    assert (
        visible[1]["values"] == full[1]["values"][:20]
        and visible[1]["values"] != visible[0]["values"]
    )


def test_partial_zip_keeps_paid_searches_out_of_verified_results(tmp_path):
    files = {
        "models.json": dict(receipt=digest("model"), elapsed_us=3),
        "verification.json": dict(plan=digest("plan"), research_dispatches=0,
                                  sealed_searches=None, verified={}),
    }
    files["forecast.json"] = api["conditioning_prices"](files)
    charged = dict(tokens=0, draws=272, attempts=272, rejections=272, native_work=3072,
                   elapsed_us=180_000_000, disk_bytes=0, archive_bytes=0)
    report = dict(scope=api["CONDITIONING"], status="PARTIAL", candidate="a" * 40,
                  grant=digest("original"), grant_minutes=75, charged_work=charged,
                  elapsed_us=charged["elapsed_us"], remaining_us=4_320_000_000,
                  hip_attempts=3, verified_leases=3, gpu_dispatches=3, recorded_lease_us=175_000_000,
                  research_run_requested=False, scientific_verdict=None,
                  errors={"search-" + group: "ValueError" for group in api["CONDITIONING_GROUPS"]},
                  files=files)
    archive = tmp_path / "partial.zip"
    api["write_readiness_zip"](archive, report)
    assert api["verify_conditioning_zip"](archive)["status"] == "PARTIAL"
    for key, value in (("remaining_us", 4_500_000_000), ("hip_attempts", 0),
                       ("elapsed_us", True), ("receipt", digest("invented"))):
        api["write_readiness_zip"](archive, dict(report, **{key: value}))
        with pytest.raises(ValueError):
            api["verify_conditioning_zip"](archive)
    for name, key, value in (("forecast.json", "proxy_four_treatment_paired_sample_us", 0),
                             ("verification.json", "verified", {"D-full": {}}),
                             ("verification.json", "sealed_searches", digest("unearned"))):
        altered = dict(files, **{name: dict(files[name], **{key: value})})
        api["write_readiness_zip"](archive, dict(report, files=altered))
        with pytest.raises(ValueError):
            api["verify_conditioning_zip"](archive)


def test_launch_projection_prices_the_matched_banks_once_and_requires_verified_pairs():
    previous = dict(control_expected_us=10, control_tail_us=15, known_expected_us=110,
                    known_tail_us=215, missing=["matched_full_bank_scoring",
                    "paired_conditioning_sample", "trajectory_startup_extra_checkpoints_and_drain",
                    "verification_and_publication"])
    files = {
        "previous-prices.json": dict(projection=dict(known_wave=previous,
                matched_sample_expected_us=17, matched_sample_tail_us=25)),
        "preflight.json": dict(supported_wave=dict(
            methods=["00","04","05","06"], known_expected_us=100, known_tail_us=200,
            checkpoint_receipt=digest("checkpoint"), missing=["conventional_cpu_controls", *previous["missing"]]),
            followup_reserve_us=7),
        "checkpoint-verification.json": dict(report_ref=digest("checkpoint"), mechanical_status="VERIFIED"),
        "models.json": dict(elapsed_us=5),
    }
    project = api["launch_estimate"]
    partial = project(files)
    assert (partial["known_expected_us"], partial["known_tail_us"]) == (127, 240)
    assert partial["expected_us"] is partial["tail_with_reserve_us"] is None
    assert "paired_conditioning_sample" in partial["missing"]
    cell = dict(cases=2, search_us=5, max_search_us=4, score_us=4, max_score_us=3)
    bank = dict(cases=2, scoring_us=4, max_us=3)
    strata = ("parameter","composition","unseen_structure","oeis")
    for g in api["CONDITIONING_GROUPS"]:
        files["search-" + g + ".json"] = dict(elapsed_us=60,
                modes={m: dict(elapsed_us=25) for m in ("true","shuffled")})
        files["paired-" + g + ".json"] = dict(elapsed_us=80, modes={m: dict(
                selected_us=18, bank_us=18, selected_cells=dict.fromkeys(strata, cell),
                bank_cells=dict.fromkeys(strata, bank)) for m in ("true","shuffled")})
    result = project(files)
    # Nine supervised pairs and one model setup. Each pair's tail adds the slowest
    # case twice per stratum in each of the two modes (3+2+2 microseconds per cell).
    assert result["known_expected_us"] == 110 + 5 + 9 * 140
    assert result["known_tail_us"] == 215 + 5 + 9 * (140 + 2 * 4 * 7)
    assert result["missing"] == ["trajectory_startup_extra_checkpoints_and_drain", "verification_and_publication"]
    files["preflight.json"]["supported_wave"]["fresh_receipt"] = digest("fresh")
    files["preflight.json"]["launch_preparation"] = dict(
        checkpoint_receipt=digest("checkpoint"), fresh_receipt=digest("fresh"),
        trajectory=dict(known_expected_us=50, known_tail_us=70, missing=[]),
        publication_proxy=dict(expected_us=11, tail_us=12),
        verification_and_publication=dict(expected_us=14, tail_us=25))
    priced = project(files)
    assert priced["expected_us"] == result["known_expected_us"] + 50 + 14
    assert priced["tail_with_reserve_us"] == result["known_tail_us"] + 70 + 25 + 7
    assert not priced["missing"] and priced["research_authority"] is None
    assert "historical_scale_publication_and_read" not in priced["items"]
    files["preflight.json"]["launch_preparation"]["verification_and_publication"] = None
    fallback = project(files)
    assert fallback["known_expected_us"] == result["known_expected_us"] + 50 + 11
    assert fallback["missing"] == ["round029_private_verification_and_report_scale"]
    assert fallback["tail_with_reserve_us"] is None
    wrong = deepcopy(files)
    wrong["preflight.json"]["supported_wave"]["known_expected_us"] += 1
    with pytest.raises(ValueError, match="control"):
        project(wrong)
