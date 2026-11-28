import pytest

from oeis_learn.generalization.progressive.campaign import velocity_registration
from oeis_learn.generalization.progressive.contracts import digest, seal
from oeis_learn.generalization.progressive.graph import velocity_study_looks
from oeis_learn.generalization.progressive.velocity_records import validate_grant, physical_root_id
from oeis_learn.pipeline_validation import progressive_velocity_wave as wave


def fixture_wave(methods, *, paired=False, strata=None):
    from oeis_learn.generalization.progressive.cohorts import STRATA

    strata = strata or (STRATA if len(methods) == 7 else ("oeis",))
    registration = velocity_registration(
        source="a" * 40,
        design=digest("design"),
        initialization=9,
        stream_seed=42,
        development=[
            dict(task=f"case-{s}-{i}", group=f"group-{s}-{i}", stratum=s)
            for s in strata
            for i in range(2 if paired else 1)
        ],
        retention=[dict(task="other", group="family", stratum="retention")],
        ordinary=digest("stream"),
        broad=digest("broader"),
    )
    graph = velocity_study_looks(registration)
    packages = {}
    for m in methods:
        packages[m] = seal(
            dict(
                method=m,
                policy=dict(effective_batch=16),
                order=["s"],
                rows={"s": dict(tokens=list(range(400)))},
            )
        )
    schedule = [
        dict(method=m, look=l["look"], profile=p, members=len(l["members"]), roster=l["identity"])
        for m, profiles in graph.items()
        if m in methods or m not in {f"0{i}" for i in range(7)}
        for p, looks in profiles.items()
        for l in looks
    ]
    accepted = seal(
        dict(
            registration=registration["identity"],
            supported=methods,
            omitted={m: digest(["omitted", m]) for m in graph if m.isdigit() and m not in methods},
            trials={m: digest(m) for m in graph if m.isdigit()},
            schedule=schedule,
            training=[
                dict(method=m, target=t, learner=packages[m]["identity"])
                for m in methods
                for t in (0, 1_000_000, 4_000_000)
            ],
            followup_us=5_000_000,
            proposed_total_us=75_000_000,
        )
    )
    return accepted, registration, packages


@pytest.mark.parametrize("methods", [["00", "04", "05", "06"], [f"0{i}" for i in range(7)]])
def test_frozen_wave_has_separate_endpoints_exact_crossings_and_one_allowance(tmp_path, methods):
    accepted, reg, packages = fixture_wave(methods)
    graph, looks, cuts, ops = wave.plan(accepted, reg, packages)
    assert len(graph) == 11
    assert len(ops) == 11 * len(methods) + 8
    assert cuts["00"][1_000_000]["useful_tokens"] == 1_004_800
    train = [o for o in ops if o["role"] == "train"]
    assert [looks[o["roster"]]["look"] for o in train] == sorted(
        looks[o["roster"]]["look"] for o in train
    )
    assert sum(o["cost"]["tokens"] for o in train) == cuts["00"][4_000_000]["useful_tokens"] * len(
        methods
    )
    assert all(o["roster"] in looks for o in ops)
    zero = [o for o in ops if o["role"] in {"search", "score"} and looks[o["roster"]]["look"] == 0]
    assert len([o for o in zero if o["cost"]]) == 2
    assert all(looks[o["roster"]]["method"] == "00" for o in zero if o["cost"])
    grant, approval, marker = wave.authority(
        tmp_path, "b" * 40, 2, accepted, ops, digest("driver"), digest("labels")
    )
    assert marker["grant"] == grant["identity"]
    assert grant["holdback_us"] == accepted["followup_us"]
    validate_grant(
        grant, approval, root=physical_root_id(tmp_path), parent=accepted["identity"], action="run"
    )
    with pytest.raises(ValueError, match="tail"):
        wave.authority(tmp_path, "b" * 40, 1, accepted, ops, digest("driver"), digest("labels"))
    assert (
        wave.authority(tmp_path, "b" * 40, 2, accepted, ops, digest("other"), digest("labels"))[2]
        != marker
    )


def accepted_donor(root, source, methods, monkeypatch, *, strata=None):
    """Qualify stored, measured CPU fixtures through the real preparation and archive reader."""
    from types import SimpleNamespace
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive import velocity_donation_report as public
    from oeis_learn.generalization.progressive.contracts import atomic_write, load, reference
    from oeis_learn.generalization.progressive.generator import broader_stream, structure_id
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.generalization.progressive.training_rows import pair_seeds
    from oeis_learn.generalization.progressive.velocity_records import SESSION
    from oeis_learn.pipeline_validation import progressive_round029 as command
    from oeis_learn.pipeline_validation import progressive_velocity_broad as broad
    from oeis_learn.pipeline_validation import progressive_velocity_checkpoint as checkpoint
    from oeis_learn.pipeline_validation import progressive_velocity_trial_preparation as preparation
    from oeis_learn.pipeline_validation.progressive_preparation import prepare_velocity_start
    from oeis_learn.generalization.progressive.velocity_trials import select_start
    from oeis_learn.pipeline_validation.progressive_study import open_velocity_study
    from oeis_learn.pipeline_validation.progressive_velocity_broad_host import FRESH_MARKER
    from oeis_learn.pipeline_validation.progressive_velocity_worker import measure_calibration

    _, template, _ = fixture_wave(methods, paired=True, strata=strata)
    extended = "01" in methods
    authority = digest("independent-training-seed")
    seeds = pair_seeds(authority, 0)
    profile = resolve_profile(
        resolve({"version": "bigint-bootstrap/v1"}),
        shape="smaller_width",
        seed=seeds["initialization"],
    )
    rows, slots = {}, {}
    for i in range(6 if extended else 1):
        op = "add" if i < 4 else "mul"
        body = f"(program (return ({op} $index $index)))"
        indices = list(range(67 + i, 167 + i))
        values = [str(n + n if op == "add" else n * n) for n in indices]
        key = digest(["draw", i])
        row = seal(
            dict(
                version="progressive-training-row/v1",
                ordinal=i,
                draw_identity=key,
                source_policy=authority,
                source=body,
                structure=structure_id(body),
                behavior=digest(values),
                indices=list(map(str, indices)),
                values=values,
                target_source=body,
                tokens=encode_body(body),
                target_disposition="fixed",
                membership="training",
                native_verified=True,
                control_events={},
                branch_events={},
                operations=[],
                detector_limit="fixture",
            )
        )
        rows[key] = row
        slots[key] = dict(
            draw=key,
            composition=row["structure"],
            tokens=len(row["tokens"]),
            band="middle",
            familiar=False,
        )
    seed = seeds["training_families"]
    order = sorted(list(rows)[:4], key=lambda k: (digest([seed, k]), k))
    ordinary = seal(
        dict(
            version="round028-matched-stream/v1",
            registration=authority,
            pair=0,
            seeds=seeds,
            order=order,
            row_identities=digest([rows[k]["identity"] for k in order]),
        )
    )
    stream = broader_stream(
        [slots[k] for k in order],
        list(slots.values()),
        seed=seed,
        ordinary_ref=ordinary["identity"],
    )
    audit = seal(
        dict(
            plan=digest("native-plan"),
            selection=stream["identity"],
            slots=slots,
            rejections=[],
            witnesses={k: r["identity"] for k, r in rows.items()},
        )
    )
    stream = seal(dict(stream, audit=audit["identity"]))
    assert stream["available"] == extended
    start = prepare_velocity_start(profile, ordinary, microbatch=8, broad=stream, device="cpu")
    folder = root / "run-state/discovery" / SESSION
    folder.mkdir(parents=True)
    reg = velocity_registration(
        source="a" * 40,
        design=digest("design"),
        initialization=start["initialization"],
        stream_seed=seed,
        ordinary=start["ordinary"],
        broad=start["broad"],
        training_start=select_start(dict(cuts=[]), None, start),
        development=template["development"],
        retention=template["retention"],
    )
    journal = open_velocity_study(folder, reg)
    fresh_grant = digest("original fresh grant")
    fresh_area = folder / "qualification" / fresh_grant[7:]
    native = fresh_area / "prepared/native"
    selected = {k: rows[k] for k in stream["order"]}
    ordinary_rows = [rows[k] for k in order]
    view, pairs = broad.matching_views(
        dict(
            stream=ordinary,
            ordinary_rows=ordinary_rows,
            start=start,
            frame=True,
            pair_seed=seed,
            initial_weights=start["initial_weights"],
            profile=profile,
        ),
        stream,
        selected,
    )
    observation = measure_calibration(view["streams"]["ordinary"], device="cpu")
    assert (observation["weight"] is not None) == extended
    fresh = seal(
        dict(
            start=start,
            supply=dict(matching=view["support"]),
            device=dict(calibration=observation, methods=dict.fromkeys(methods, {"tail_us": 500})),
        )
    )
    atomic_write(folder / FRESH_MARKER, seal(dict(grant=fresh_grant, receipt=fresh["identity"])))
    atomic_write(fresh_area / "receipt.json", fresh)
    atomic_write(
        native / "fresh-ordinary.json",
        seal(dict(stream=ordinary, ordinary_rows=ordinary_rows, start=start)),
    )
    atomic_write(native / "pairs.json", pairs)
    path = native / "projection.json"
    atomic_write(path, view)
    area = native / "run-state/discovery" / SESSION / "preparation/composition" / audit["plan"][7:]

    def save(name, value):
        atomic_write(area / name, value)
        return reference(area, area / name)

    save(
        "receipt.json",
        seal(
            dict(
                stream=save("stream.json", stream),
                audit=save("audit.json", audit),
                rows={k: save(k[7:] + ".json", r) for k, r in selected.items()},
            )
        ),
    )
    matrix = checkpoint.design(
        view, microbatch=8, origin=fresh["identity"], eligible=methods, calibration=observation
    )
    looks = {
        m: {
            p: [
                dict(look=r["look"], members=len(r["members"]), receipt=r["identity"])
                for r in series
            ]
            for p, series in profiles.items()
        }
        for m, profiles in velocity_study_looks(reg).items()
    }
    reasons = [
        "ordinary already has a distinct composition in every matched slot",
        "complete census found zero eligible ordinary matching slots",
    ]
    omitted = {} if extended else {"01": reasons[:1], "02": reasons[1:], "03": reasons}
    sources = {k: digest(k + "-price") for k in preparation.SOURCES}
    sources["fresh"] = fresh["identity"]
    quote = seal(
        dict(
            sources=sources,
            expected_us=50_000_000,
            tail_us=70_000_000,
            followup_reserve_us=5_000_000,
        )
    )
    q = seal(
        dict(
            candidate="a" * 40,
            **{k: digest(k) for k in ("source", "image", "wheel", "physical", "role_policy")},
            quote=quote,
            registration=reg["identity"],
            supported=methods,
            omitted=omitted,
            seed=seeds["initialization"],
            stream_seed=seed,
            graph=digest(looks),
            looks=looks,
            trials={m: journal.velocity_trial(m)["identity"] for m in looks if m.isdigit()},
        )
    )
    cut = dict(
        receipt=digest("initial cut"), state=digest("state"), checkpoint=digest("checkpoint")
    )
    atomic_write(
        folder / preparation.INITIAL_MARKER,
        seal(dict(grant=digest("zero grant"), receipt=cut["receipt"])),
    )
    grant = load(preparation.authorize(root, "b" * 40, 2))
    mount = dict(
        device=root.stat().st_dev,
        **{
            kind + "_" + unit: value
            for unit in ("bytes", "inodes")
            for kind, value in (("required", 1), ("reserved", 1), ("available", 9), ("deficit", 0))
        },
    )
    prices = dict(
        launch_estimate=dict(missing=[], sources=sources),
        launch_options=dict(
            research_authority=None, scopes=dict(wave=dict(fits=True, mounts=[mount]))
        ),
    )
    with monkeypatch.context() as hardware:
        hardware.setattr(
            command,
            "preflight",
            lambda *a, **kw: dict(
                inventory=dict(prices=prices, gpu_dispatches=0, assessment="sealed")
            ),
        )
        hardware.setattr(
            preparation, "verified_initialization", lambda *args: (q, cut, matrix, path)
        )
        result, code = preparation.qualify(SimpleNamespace(), root, folder, source, "b" * 40, grant)
        assert code == 0, result
    accepted = public.read_member(
        preparation.retained_qualification(root, grant), preparation.MEMBER
    )
    assert accepted["omitted"] == {m: digest(reason) for m, reason in omitted.items()}
    assert preparation.published(root, folder)["accepted"] == accepted["identity"]
    reopened, packages = wave.inputs(root, folder, accepted)
    assert reg == reopened
    if extended:
        assert (
            packages["00"]["rows"]
            == packages["02"]["rows"]
            != packages["01"]["rows"]
            == packages["03"]["rows"]
        )
        for m, name in (("02", "ordinary"), ("03", "broader")):
            assert packages[m]["pair_proof"] == pairs[name]["identity"]
            assert packages[m]["policy"]["matching_weight"] == observation["weight"]
            assert packages[m]["donors"] == view["streams"][name]["donors"] != {}
        from oeis_learn.pipeline_validation.progressive_velocity_wave_role import perform
        for method in ("01", "02", "03"):
            learner = packages[method]
            request = dict(role="train", learner=learner["identity"], parent_files=[], target=1,
                           cap=matrix["methods"][method]["first"])
            result = perform(root / ("cpu-update-" + method), request, dict(package=learner), device="cpu")
            assert result["cut"]["updates"] == 1
            assert (result["cut"]["metrics"].get("matching_eligible", 0) > 0) == (method != "01")
        original = (native / "pairs.json").read_bytes()
        import json
        (native / "pairs.json").write_text(json.dumps(dict(pairs, broader=pairs["ordinary"])))
        with pytest.raises(ValueError):
            wave.inputs(root, folder, accepted)
        (native / "pairs.json").write_bytes(original)
    return folder, accepted, reopened, packages


def test_changed_effective_worker_requires_new_evidence_and_matching_starts_reuse_it(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    import shutil
    from oeis_learn.generalization.progressive.contracts import atomic_write, load, reference
    from oeis_learn.pipeline_validation import progressive_round029 as command
    from oeis_learn.pipeline_validation import progressive_velocity_wave_qualification as rehearsal
    from oeis_learn.pipeline_validation import progressive_velocity_wave_host as host

    source = Path(__file__).resolve().parents[2]
    folder, accepted, _, _ = accepted_donor(tmp_path, source, ["00","04","05","06"], monkeypatch)
    mirror = tmp_path / "candidate"
    for name in (wave.DRIVER, wave.EVALUATOR, wave.CONTROLS):
        path = mirror / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, path)
    wheel = tmp_path / "fixture.whl"
    wheel.write_bytes(b"disposable")
    monkeypatch.setattr(host, "runtime", lambda *_: wheel)
    monkeypatch.setattr(command, "preflight", lambda *_: dict(inventory=dict(prices=dict(
        launch_preparation=dict(wave_space=dict(dispatch_prefixes={}))))))
    launches = []
    class Supervisor:
        def __init__(self, *args):
            pass
        def supervise(self, request, data, extra, proof):
            def execute(out, allowance):
                role = request["role"]
                launches.append(role)
                if role == "search":
                    candidate = out / "choices.json"
                    atomic_write(candidate, seal(dict(choice=digest("selected"))))
                atomic_write(out / "producer.json", seal(dict(request=request["identity"])))
                atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
            def verify(out):
                selected = (dict(artifact=reference(out, out / "choices.json"), probe=dict(identity=digest("probe")))
                            if request["role"] == "search" else {})
                return seal(dict(producer=load(out / "producer.json")["identity"], **selected))
            return execute, verify, lambda: None
    monkeypatch.setattr(host, "Donor", Supervisor)
    for _ in range(2):
        path = rehearsal.authorize(tmp_path, mirror, 30, accepted["identity"])
        if not launches:
            with pytest.raises(ValueError, match="qualification"):
                rehearsal.required(tmp_path, folder, mirror, accepted)
        value, code = rehearsal.qualify(SimpleNamespace(), tmp_path, folder, mirror, "a"*40, load(path), None)
        assert code == 0 and value["last_committed_ref"] == rehearsal.required(tmp_path, folder, mirror, accepted)
        assert launches == ["train", "search", "score"]
    adapter = mirror / wave.CONTROLS
    with adapter.open("a") as modified:
        modified.write("\n# changed mounted control\n")
    with pytest.raises(ValueError, match="qualification"):
        rehearsal.required(tmp_path, folder, mirror, accepted)
    assert not (folder / wave.MARKER).exists()
