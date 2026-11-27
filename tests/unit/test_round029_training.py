import math
import pytest
from oeis_learn.generalization.progressive.campaign import velocity_policy
from oeis_learn.generalization.progressive.contracts import digest
from oeis_learn.generalization.progressive.velocity_trials import recipe_bindings, exposure


def test_grid_and_useful_crossing_after_discard():
    rows = velocity_policy()["recipes"]
    assert [(r["batch"], r["decay"]) for r in rows] == [(32, 0)] * 4 + [
        (16, 0),
        (32, 0.01),
        (16, 0.01),
    ]
    bindings = recipe_bindings(
        digest("new-registration"),
        initialization=13,
        stream_seed=72,
        ordinary=digest("old stream"),
        broad=digest("broad"),
    )
    assert {b["initialization"] for b in bindings.values()} == {13}
    for recipe in ("00", "02", "04", "05", "06"):
        assert bindings[recipe]["stream_ref"] == digest("old stream")
    assert bindings["01"]["stream_ref"] == bindings["03"]["stream_ref"]
    stream = {"order": ["a", "b", "c"]}
    data = {"a": {"tokens": [1, 2]}, "b": {"tokens": [1]}, "c": {"tokens": [2]}}
    for size in (16, 32):
        cut = exposure(stream, data, updates=2, batch=size, discarded=100)
        assert cut["cursor"] == 2 * size
        assert cut["useful_tokens"] == 2 * size + math.ceil(2 * size / 3)
        assert cut["charged_tokens"] == cut["useful_tokens"] + 100


def test_masked_mean_scores_and_batch_normalization():
    import torch
    from oeis_learn.rl.sft_trainer import prefix_matching_loss, calibrate_matching

    true = torch.tensor(
        [[[0.0, 1.0, 2.0], [0.0, 2.0, 1.0]], [[0.0, 1.0, 2.0], [0.0, 2.0, 1.0]]], requires_grad=True
    )
    donor = torch.zeros((1, 2, 3), requires_grad=True)
    target = torch.tensor([[2, 0], [1, 2]])
    aux, wins = prefix_matching_loss(true, donor, target, [0], pad_id=0)
    d = -math.log(3) - (2 - math.log(1 + math.e + math.exp(2)))
    assert aux.item() == pytest.approx(math.log1p(math.exp(d)) / 2)
    assert wins == 1
    zero, wins_zero = prefix_matching_loss(true, donor[:0], target, [], pad_id=0)
    assert zero.item() == 0 and wins_zero == 0
    aux.backward()
    assert true.grad[1].abs().sum() == 0 and torch.isfinite(donor.grad).all()
    assert calibrate_matching([2.0, 4.0, 6.0], [1.0, 2.0, 3.0]) == 0.5
    with pytest.raises(ValueError):
        calibrate_matching([2.0], [])
    for positives, auxiliary in (([1e308], [1e-308]), ([1e-308], [1e308])):
        with pytest.raises(ValueError):
            calibrate_matching(positives, auxiliary)


def test_composition_preserves_familiar_slots_and_closes_only_shortage():
    from oeis_learn.generalization.progressive.generator import broader_stream

    def member(i, family, familiar=False, length=4):
        return dict(draw=i, composition=family, tokens=length, band="middle", familiar=familiar)

    slots = [member("a", "A", True), member("b", "B"), member("b", "B"), member("b", "B")]
    supply = [member(str(k), c) for k, c in enumerate("BCDE")]
    result = broader_stream(slots, supply, seed=12)
    assert result["available"] and result["order"][0] == "a"
    assert result["ordinary"]["tokens"] == result["broad"]["tokens"] == 16
    assert result["broad"]["distinct"] == 4
    assert not broader_stream(slots, supply[:1], seed=12)["available"]


def test_frozen_study_registers_and_reopens_looks_without_sampler_feedback(tmp_path):
    from oeis_learn.generalization.progressive.campaign import velocity_registration
    from oeis_learn.generalization.progressive.graph import velocity_study_looks
    from oeis_learn.pipeline_validation.progressive_study import (
        open_velocity_study,
        close_velocity_study,
    )

    development = [
        dict(task="a", group="f", stratum="oeis"),
        dict(task="b", group="f", stratum="oeis"),
    ]
    retention = [dict(task="r", group="known", stratum="retention")]
    args = dict(
        source="a" * 40,
        design=digest("frozen hypotheses, workers and prices"),
        development=development,
        retention=retention,
        initialization=53,
        stream_seed=67,
        ordinary=digest("ordinary stream"),
        broad=digest("broad stream"),
    )
    reg = velocity_registration(**args)
    journal = open_velocity_study(tmp_path, reg)
    assert len(journal.trials) == 7
    assert [journal.velocity_trial(f"{i:02d}")["binding"]["recipe"]["id"] for i in range(7)] == [
        "00",
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
    ]
    graph = velocity_study_looks(reg)
    cheap = graph["00"]["common"]
    assert [r["look"] for r in cheap] == [0, 1_000_000, 4_000_000]
    assert graph["00"]["full"][0]["members"] == development + retention
    assert all(r["members"] == development[:1] for r in cheap)
    with pytest.raises(ValueError, match="previous"):
        journal.observe_velocity(reg, cheap[1], evidence=digest("one million"))
    first = journal.observe_velocity(reg, cheap[0], evidence=digest("zero"))
    assert journal.observe_velocity(reg, cheap[0], evidence=digest("zero")) == first
    with pytest.raises(ValueError, match="differs"):
        journal.observe_velocity(reg, cheap[0], evidence=digest("other"))
    with pytest.raises(ValueError):
        journal.observe_velocity(reg, dict(cheap[1], members=[]), evidence=digest("one million"))
    journal.observe_velocity(reg, cheap[1], evidence=digest("one million"))
    journal.observe_velocity(reg, cheap[2], reason="physical budget exhausted")
    assert all(row["status"] == "PROPOSED" for row in journal.trials.values())
    assert (
        open_velocity_study(tmp_path, reg).observe_velocity(reg, cheap[0], evidence=digest("zero"))
        == first
    )
    from oeis_learn.generalization.progressive.budgets import KEYS
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.generalization.progressive.velocity_records import SESSION, physical_root_id

    accepted = seal(dict(registration=reg["identity"], scope="exploration"))
    authority = physical_root_id(tmp_path)
    terms = dict(
        version="round029-grant/v1",
        study_id=SESSION,
        scope="cpu",
        state_root_id=authority,
        registration_or_parent_ref=accepted["identity"],
        total_us=100,
        holdback_us=10,
        allowed_operations=["cpu", "structure_search"],
        resources=dict(dict.fromkeys(KEYS, 0), elapsed_us=100),
    )
    approval = seal(dict(version="round029-authorization/v1", terms=digest(terms)))
    grant = seal(dict(terms, authorizing_record_ref=approval["identity"]))
    funded = dict(
        decision=digest("observed deficit"),
        reason="price a fixed-order comparison",
        grant=grant,
        approval=approval,
        accepted=accepted,
        root_id=authority,
    )
    assert (
        journal.record_velocity_branch("structure_search", **funded)["grant"] == grant["identity"]
    )
    assert (
        journal.record_velocity_branch("paired_extension", decision=first, reason="unfunded")[
            "state"
        ]
        == "unused"
    )
    with pytest.raises(ValueError):
        journal.record_velocity_branch("confirmation", **funded)
    boundary = close_velocity_study(tmp_path, reg, state="resumable", evidence=first)
    with pytest.raises(ValueError, match="authenticated"):
        velocity_registration(**args, scope="confirmation", parent=boundary, finalist="03")
    with pytest.raises(ValueError):
        velocity_registration(
            **args, scope="confirmation", parent=dict(boundary, state="running"), finalist="03"
        )


def test_whole_update_crossing_after_discard_and_seed_provenance():
    from oeis_learn.generalization.progressive.velocity_trials import crossed

    stream = {"order": ["a", "b"]}
    rows = {"a": {"tokens": [1, 2, 3]}, "b": {"tokens": [4]}}
    first = exposure(stream, rows, updates=1, batch=16, discarded=64)
    second = exposure(stream, rows, updates=2, batch=16, discarded=64)
    assert (first["useful_tokens"], first["charged_tokens"]) == (32, 96)
    assert not crossed(first, 48) and crossed(second, 48) and not crossed(second, 32)
    args = dict(
        initialization=9, stream_seed=21, ordinary=digest("ordinary"), broad=digest("broad")
    )
    before = recipe_bindings(digest("study"), **args)
    for changed in (
        dict(initialization=10),
        dict(stream_seed=22),
        dict(ordinary=digest("other ordinary")),
    ):
        other = recipe_bindings(digest("study"), **dict(args, **changed))
        assert before["00"]["identity"] != other["00"]["identity"]
    for invalid in (-1, 2**32, True):
        with pytest.raises(ValueError):
            recipe_bindings(digest("study"), **dict(args, initialization=invalid))


def test_screen_freezes_verified_shared_start_and_reruns_control_only_on_mismatch(
    tmp_path, monkeypatch
):
    from oeis_learn.generalization.progressive import training_rows
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.generalization.progressive.generator import broader_stream
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.pipeline_validation import progressive_velocity_donor as donor
    from oeis_learn.pipeline_validation.progressive_preparation import (
        freeze_velocity_screen,
        prepare_velocity_start,
    )
    from oeis_learn.pipeline_validation.progressive_study import open_velocity_study
    from oeis_learn.fresh.bigint_config import resolve

    def prepared(anchor):
        origin = digest(anchor)
        seeds = training_rows.pair_seeds(origin, 0)
        draws = list(map(digest, range(3)))
        order = sorted(draws, key=lambda k: (digest([seeds["training_families"], k]), k))
        stream = seal(
            dict(
                version="round028-matched-stream/v1",
                registration=origin,
                pair=0,
                seeds=seeds,
                order=order,
                row_identities=digest(order),
            )
        )
        p = resolve_profile(
            resolve({"version": "bigint-bootstrap/v1"}),
            shape="smaller_width",
            seed=seeds["initialization"],
        )

        def member(key, composition):
            return dict(draw=key, composition=composition, tokens=4, band="middle", familiar=False)

        broad = broader_stream(
            [member(key, "A") for key in order],
            [member(key, str(i)) for i, key in enumerate(draws)],
            seed=seeds["training_families"],
            ordinary_ref=stream["identity"],
        )
        start = prepare_velocity_start(p, stream, microbatch=8, broad=broad, device="cpu")
        return start, stream, p, broad

    preferred, stream, model, broad = prepared("original seed authority")
    replacement, _, _, _ = prepared("separately chosen fresh authority")
    cut = dict(
        method="00",
        look=0,
        updates=0,
        cursor=0,
        useful_tokens=0,
        weights=digest("zero weight blob"),
        initial=preferred["initial_weights"],
        lineage=stream["identity"],
        seeds=stream["seeds"],
        policy=preferred["common_policy"],
        codec=preferred["codec"],
    )
    inventory = dict(cuts=[cut], stream=dict(identity=stream["identity"], seeds=stream["seeds"]))
    monkeypatch.setattr(donor, "inventory", lambda root: inventory)
    kwargs = dict(
        source="c" * 40,
        design=digest("initial freeze"),
        development=[dict(group="f", task="d", stratum="oeis")],
        retention=[dict(group="r", task="r", stratum="retention")],
    )
    reg = freeze_velocity_screen(tmp_path, preferred=preferred, fallback=replacement, **kwargs)
    assert reg["training_start"]["mode"] == "reuse_ordinary_start"
    (tmp_path / "studies").mkdir()
    journal = open_velocity_study(tmp_path / "studies", reg)
    trials = {
        method: journal.velocity_trial(method)["binding"]
        for method in ("00", "01", "02", "03", "04", "05", "06")
    }
    assert {item["initial_weights"] for item in trials.values()} == {preferred["initial_weights"]}
    assert {trials[k]["stream_ref"] for k in ("00", "02", "04", "05", "06")} == {stream["identity"]}
    assert {trials[k]["stream_ref"] for k in ("01", "03")} == {broad["identity"]}
    reg2 = freeze_velocity_screen(
        tmp_path,
        preferred=preferred,
        fallback=replacement,
        **dict(kwargs, design=digest("another design")),
    )
    assert reg2["identity"] != reg["identity"] and reg2["training_start"] == reg["training_start"]
    for change in (
        dict(initial=digest("changed weights")),
        dict(seeds={**stream["seeds"], "initialization": 17}),
        dict(lineage=digest("other order")),
        dict(policy=digest("changed policy")),
    ):
        inventory["cuts"] = [{**cut, **change}]
        fresh = freeze_velocity_screen(
            tmp_path, preferred=preferred, fallback=replacement, **kwargs
        )
        assert fresh["training_start"]["mode"] == "fresh_shared_control"
        assert (
            fresh["ordinary"] == replacement["ordinary"]
            and fresh["initialization"] == replacement["initialization"]
        )
        assert (
            open_velocity_study(tmp_path / "studies", fresh).velocity_trial("00")["binding"][
                "initial_weights"
            ]
            == replacement["initial_weights"]
        )
    inventory["cuts"] = [
        dict(cut, weights=None),
        dict(
            cut,
            look=4_000_000,
            weights=digest("trained weights"),
            updates=3010,
            cursor=96320,
            useful_tokens=4_000_210,
        ),
    ]
    reproduced = freeze_velocity_screen(tmp_path, preferred=preferred, fallback=None, **kwargs)
    assert reproduced["training_start"]["mode"] == "reconstruct_ordinary_start"
    assert reproduced["training_start"]["control_weights"] is None
    assert reproduced["training_start"]["origin"]["initial_weights"] == preferred["initial_weights"]
    with pytest.raises(ValueError):
        prepare_velocity_start(
            model,
            stream,
            microbatch=8,
            broad=seal(dict(broad, ordinary_ref=digest("foreign"))),
            device="cpu",
        )
    with pytest.raises(ValueError):
        prepare_velocity_start(
            model,
            seal(dict(stream, order=list(reversed(stream["order"])))),
            microbatch=8,
            device="cpu",
        )


@pytest.fixture
def matched_composition_inputs(tmp_path):
    from oeis_learn.generalization.progressive import generator, training_rows
    from oeis_learn.generalization.progressive.complexity import measure
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.generalization.progressive.program_search_panels import descriptor, bin_key
    from oeis_learn.pipeline_validation.progressive_preparation import (
        verify_draw,
    )
    from oeis_learn.synthesis.reference import evaluate

    authority = digest("composition inputs")
    seeds = training_rows.pair_seeds(authority, 0)
    law = generator.default_policy(seed=seeds["training_families"])

    class ExactClient:
        calls = 0
        reference_sha256 = digest("reference fixture")

        def runtime_identity(self, profile):
            return digest(["fixture", profile])

        def fit(self, source, prompt, **_):
            return dict(outcome="timeout")

        def execute(self, source, indices, **_):
            self.calls += 1
            return dict(outcome="ok", values=list(evaluate(source, indices).values))

    client = ExactClient()
    buckets, witnesses = {}, {}
    for ordinal in (0, 11, 14, 46, 25, 43):
        measurement = {}
        row, reason = verify_draw(
            generator.draw(law, ordinal), client=client, measurements=measurement
        )
        assert reason is None
        witnesses[ordinal] = row
        item = dict(
            row,
            stratum="training",
            difficulty=measure(row["source"], reference_steps=measurement["reference_steps"]),
        )
        buckets[bin_key(descriptor(item))] = "middle"
    calibration = seal(
        dict(
            version="round028-difficulty-map/v1",
            bins=buckets,
            families=len(buckets),
            observations=digest("reserved calibration observations"),
        )
    )
    ordinary = [witnesses[n] for n in (0, 11, 14, 46)]
    keys = {r["draw_identity"]: r for r in ordinary}
    order = sorted(keys, key=lambda k: (digest([seeds["training_families"], k]), k))
    stream = seal(
        dict(
            version="round028-matched-stream/v1",
            registration=authority,
            pair=0,
            seeds=seeds,
            order=order,
            row_identities=digest([keys[k]["identity"] for k in order]),
        )
    )
    args = dict(
        root=tmp_path,
        stream=stream,
        shards=[ordinary],
        law=law,
        ordinals=[25, 43],
        familiar=[witnesses[0]["structure"]],
        calibration=calibration,
        protected=[],
        indexes=[],
        client=client,
    )
    return args, witnesses


def test_broad_preparation_rechecks_witnesses_preserves_slots_and_excludes_protected(
    tmp_path, matched_composition_inputs
):
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.pipeline_validation.progressive_preparation import prepare_velocity_broad

    args, witnesses = matched_composition_inputs
    client = args["client"]
    stream = args["stream"]
    order = stream["order"]
    keys = {r["draw_identity"]: r for r in args["shards"][0]}
    broad, rows, audit = prepare_velocity_broad(**args)
    assert broad["available"] and broad["ordinary"]["tokens"] == broad["broad"]["tokens"]
    assert broad["broad"]["per_million"] > broad["ordinary"]["per_million"]
    known = witnesses[0]["draw_identity"]
    assert broad["order"][order.index(known)] == known and set(rows) == set(broad["order"])
    assert broad["audit"] == audit["identity"] and audit["ordinary"] == stream["identity"]
    first_calls = client.calls
    assert prepare_velocity_broad(**args)[0] == broad and client.calls == first_calls
    for old, new in zip(order, broad["order"], strict=True):
        assert audit["slots"][old]["band"] == audit["slots"][new]["band"]
        assert len(keys[old]["tokens"]) == len(rows[new]["tokens"])
    # A protected original blocks a prospective composition even through a shifted window.
    excluded = {
        "id": "reserved",
        "stratum": "oeis",
        "group": "sealed",
        "indices": list(map(str, range(100))),
        "values": [str(1 << 200)] * 100,
    }
    excluded["values"][:20] = witnesses[43]["values"][80:]
    blocked, _, audited = prepare_velocity_broad(**dict(args, protected=[excluded]))
    assert witnesses[43]["draw_identity"] not in blocked["order"]
    assert any(
        x["reason"] == "protected_family_or_unverified_history" for x in audited["rejections"]
    )
    no_supply, _, _ = prepare_velocity_broad(**dict(args, ordinals=[]))
    assert (
        not no_supply["available"]
        and no_supply["ordinary"]["tokens"] == broad["ordinary"]["tokens"]
    )
    with pytest.raises(ValueError):
        prepare_velocity_broad(
            **dict(args, stream=seal(dict(stream, row_identities=digest("changed"))))
        )
    counted = [p for p in tmp_path.rglob("audit.json") if audit["identity"] in p.read_text()]
    assert len(counted) == 1
    counted[0].write_text("{}")
    with pytest.raises(ValueError):
        prepare_velocity_broad(**args)


def test_matching_weight_frozen_from_ordinary_rows_for_both_treatments(
    tmp_path, velocity_supply_rows
):
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.generalization.progressive.graph import velocity_study_looks
    from oeis_learn.generalization.progressive.models import resolve_profile
    from oeis_learn.generalization.progressive.training_rows import pair_seeds
    from oeis_learn.pipeline_validation.progressive_preparation import (
        freeze_velocity_screen,
        prepare_velocity_start,
    )
    from oeis_learn.pipeline_validation.progressive_study import open_velocity_study
    from oeis_learn.pipeline_validation.progressive_velocity_matching import (
        prepare_calibration,
        commit_calibration,
        registered_update_policy,
    )
    from oeis_learn.pipeline_validation.progressive_velocity_worker import measure_calibration

    authority = digest("calibration seed")
    seeds = pair_seeds(authority, 0)
    model = resolve_profile(
        resolve({"version": "bigint-bootstrap/v1"}),
        seed=seeds["initialization"],
        shape="smaller_width",
    )

    def study(rows):
        by_key = {r["draw_identity"]: r for r in rows}
        order = sorted(by_key, key=lambda k: (digest([seeds["training_families"], k]), k))
        stream = seal(
            dict(
                version="round028-matched-stream/v1",
                registration=authority,
                pair=0,
                seeds=seeds,
                order=order,
                row_identities=digest([by_key[k]["identity"] for k in order]),
            )
        )
        start = prepare_velocity_start(model, stream, microbatch=8, device="cpu")
        reg = freeze_velocity_screen(
            tmp_path,
            source="3" * 40,
            design=digest("matching design"),
            preferred=None,
            fallback=start,
            development=[dict(group="a", task="x", stratum="oeis")],
            retention=[dict(group="b", task="y", stratum="retention")],
        )
        return stream, reg, open_velocity_study(tmp_path, reg)

    rows = velocity_supply_rows
    stream, reg, journal = study(rows)
    req = prepare_calibration(journal, reg, model, stream, [rows], batches=1)
    assert all(
        set(row) == {"values", "tokens"} and len(row["values"]) == 20
        for row in req["rows"].values()
    )
    value = measure_calibration(req, device="cpu")
    assert value["positive_rows"] == 32 and 0 < value["eligible_rows"] < 32
    assert value["weight"] * value["median_auxiliary"] == pytest.approx(
        value["median_positive"] / 4
    )
    assert value["elapsed_us"] > 0 and value["extra_passes"] == 1
    saved = commit_calibration(journal, req, value)
    assert commit_calibration(journal, req, value) == saved
    from oeis_learn.pipeline_validation.progressive_preparation import prepare_velocity_learner
    from oeis_learn.pipeline_validation.progressive_velocity_worker import train_to

    package, sidecar = prepare_velocity_learner(journal, "02", model, stream, [rows], microbatch=8)
    assert sidecar["package"] == package["identity"] and all(
        set(r) == {"tokens", "values"} for r in package["rows"].values()
    )
    assert package["pair_proof"] == req["pair_proof"] == sidecar["matching_pairs"]["identity"]
    with pytest.raises(ValueError, match="selections differ"):
        prepare_velocity_learner(
            journal,
            "02",
            model,
            stream,
            [rows],
            microbatch=8,
            pair_manifest=seal(dict(sidecar["matching_pairs"], seed=29)),
        )
    assert package["initial_weights"] == sidecar["initial_weights"] == value["initial_weights"]
    mismatched = seal(dict(package, initial_weights=digest("other weights at zero")))
    with pytest.raises(ValueError, match="initial learner weights"):
        train_to(tmp_path / "mismatched-start", mismatched, target=1, cap=2000, device="cpu")
    assert not (tmp_path / "mismatched-start/checkpoint/generalization-latest.json").exists()
    trained = train_to(tmp_path / "registered-training", package, target=1, cap=2000, device="cpu")
    assert trained["updates"] == 1 and trained["metrics"]["matching_eligible"] > 0
    assert (
        math.isfinite(trained["metrics"]["gradient_norm"])
        and trained["metrics"]["matching_forward_us"] > 0
    )
    with pytest.raises(ValueError):
        prepare_velocity_learner(journal, "03", model, stream, [rows], microbatch=8)
    assert (
        registered_update_policy(journal, model, "02", microbatch=8)["matching_weight"]
        == value["weight"]
    )
    assert (
        registered_update_policy(journal, model, "03", microbatch=8)["matching_weight"]
        == value["weight"]
    )
    with pytest.raises(ValueError):
        commit_calibration(journal, req, seal(dict(value, weight=2 * value["weight"])))
    with pytest.raises(ValueError):
        prepare_calibration(
            journal,
            reg,
            model,
            seal(dict(stream, order=list(reversed(stream["order"])))),
            [rows],
            batches=1,
        )
    with pytest.raises(ValueError):
        prepare_calibration(journal, reg, model, stream, [rows], batches=2)
    journal.observe_velocity(
        reg, velocity_study_looks(reg)["00"]["common"][0], evidence=digest("zero look")
    )
    assert prepare_calibration(journal, reg, model, stream, [rows], batches=1) == req
    zero, reg0, j0 = study(rows[:2])
    req0 = prepare_calibration(j0, reg0, model, zero, [rows[:2]], batches=1)
    unavailable = measure_calibration(req0, device="cpu")
    assert unavailable["weight"] is None and unavailable["eligible_rows"] == 0
    commit_calibration(j0, req0, unavailable)
    assert registered_update_policy(j0, model, "00", microbatch=8)["matching_weight"] is None
    with pytest.raises(ValueError, match="eligible"):
        registered_update_policy(j0, model, "02", microbatch=8)


def test_donor_timing_covers_cyclic_positions_and_weights_each_batch_once():
    import math
    from oeis_learn.pipeline_validation.progressive_velocity_timing import batch_plan, unit_price

    keys = [str(i) for i in range(37)]
    rows = {k: {"tokens": [0] * (1 + int(k) % 7)} for k in keys}
    plan = batch_plan(keys, rows, {keys[0]: ["7"]}, 16, target=8000)
    assert sum(cell["batches"] for cell in plan) == len(keys) // math.gcd(len(keys), 16)
    assert {cell["has_pair"] for cell in plan} == {False, True}
    seen = {cursor for c in plan for cursor in (c["typical"], c["slow"])}
    clocks = {
        i: 11 + sum(len(rows[keys[(i + j) % len(keys)]]["tokens"]) for j in range(16)) for i in seen
    }
    expect = unit_price(plan, clocks)
    total = sum(cell["positive_tokens"] for cell in plan)
    assert total == 16 * sum(len(r["tokens"]) for r in rows.values())
    tokens = updates = 0
    while tokens < 8000:
        tokens += sum(len(rows[keys[(16 * updates + j) % len(keys)]]["tokens"]) for j in range(16))
        updates += 1
    assert sum(c["tokens_at_look"] for c in plan) == tokens
    assert sum(c["updates_at_look"] for c in plan) == updates
    assert expect["expected_us"] == sum(
        cell["updates_at_look"] * clocks[cell["typical"]] for cell in plan
    )
    assert expect["tail_us"] >= expect["expected_us"] > 0
    with pytest.raises(ValueError):
        unit_price(plan, dict(clocks, unused=4))


def test_timing_projection_charges_all_updates_and_omits_hidden_terms(velocity_supply_rows):
    from oeis_learn.generalization.progressive.contracts import digest, seal
    from oeis_learn.generalization.progressive.training_rows import matching_pairs
    from oeis_learn.pipeline_validation.progressive_velocity_timing import (
        timing_projection,
        timing_design,
    )

    rows = velocity_supply_rows
    order = [r["draw_identity"] for r in rows]
    pairs = matching_pairs([rows], order, seed=29)
    stream = seal(dict(order=order, row_identities=digest([r["identity"] for r in rows])))
    package = seal(dict(profile={}, microbatch=1, stream=stream, rows=rows))
    p = timing_projection(package, pairs, digest("weights"), digest("receipt"))
    assert all(
        set(r) == {"values", "tokens"} and len(r["values"]) == 20 for r in p["rows"].values()
    )
    assert len(p["order"]) == 32 and len(p["donors"]) == pairs["eligible_rows"]
    design = timing_design(p, 3, 1)
    assert set(design["methods"]) == {"00", "02", "04", "05", "06"}
    assert design["cost"]["tokens"] > 0 and design["cost"]["attempts"] >= 5
    with pytest.raises(ValueError, match="ordinary"):
        timing_projection(
            package,
            seal(dict(pairs, order=digest(list(reversed(order))))),
            digest("weights"),
            digest("receipt"),
        )


def test_cpu_replays_the_measured_optimizer_cells_and_rejects_changed_charges(velocity_supply_rows):
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.generalization.progressive.contracts import digest, seal
    from oeis_learn.generalization.progressive.models import construct, resolve_profile
    from oeis_learn.generalization.progressive.training_rows import matching_pairs
    from oeis_learn.pipeline_validation.progressive_velocity_timing import (
        timing_projection,
        timing_design,
        run_matrix,
        matrix_prices,
    )

    model = resolve_profile(
        resolve({"version": "bigint-bootstrap/v1"}), seed=29, shape="smaller_width"
    )
    encoder, decoder, _ = construct(model, device="cpu")
    initial = model_snapshot(encoder, decoder)
    del encoder, decoder
    rows = velocity_supply_rows
    keys = [r["draw_identity"] for r in rows]
    stream = seal(dict(order=keys, row_identities=digest([r["identity"] for r in rows])))
    pairs = matching_pairs([rows], keys, seed=29)
    p = timing_projection(
        seal(dict(profile=model, microbatch=8, stream=stream, rows=rows)),
        pairs,
        initial,
        digest("paid"),
    )
    d = timing_design(p, len(rows), 8)
    observed = run_matrix(p, d, device="cpu")
    prices = matrix_prices(observed, p, d, device="cpu")
    assert prices["calibration_eligible"] > 0 and prices["update_count"] == d["cost"]["attempts"]
    assert all(r["tail_us"] >= r["expected_us"] > 0 for r in prices["methods"].values())
    with pytest.raises(ValueError, match="device"):
        matrix_prices(observed, p, d, device="cuda")
    observed["methods"]["02"]["samples"]["0"]["eligible"] += 1
    with pytest.raises(ValueError, match="charge"):
        matrix_prices(seal(observed), p, d, device="cpu")


def test_historical_optimizer_prices_survive_reader_changes_and_reject_workload_drift(tmp_path):
    import shutil
    import subprocess
    from pathlib import Path

    from oeis_learn.experiments.source_snapshot import collect_snapshot
    from oeis_learn.pipeline_validation.progressive_velocity_timing_host import checked_price_source

    root = Path(__file__).resolve().parents[2]
    source = tmp_path / "reader"
    source.mkdir()
    helper = Path("specs/015-learned-program-search/validation/candidate_source.py")
    (source / helper).parent.mkdir(parents=True)
    shutil.copyfile(root / helper, source / helper)

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], text=True).strip()

    git("init", "-q")
    original = tmp_path / "tests/clock.py"
    original.parent.mkdir()
    original.write_text("value = 1")
    assets = [
        ".specify/memory/constitution.md",
        "specs/007-experiment-foundation/validation/adoption.md",
        "specs/007-experiment-foundation/validation/review-inputs.json",
        "specs/007-experiment-foundation/tasks.md",
    ]
    for asset in assets:
        for tree in (source, tmp_path):
            (tree / asset).parent.mkdir(parents=True, exist_ok=True)
            (tree / asset).write_text("fixture")
    git("add", "tests/clock.py", *assets)
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "timed")
    revision = git("rev-parse", "HEAD")
    cache = tmp_path / "run-state/discovery-tools/round028-source" / revision
    (cache / "tests").mkdir(parents=True)
    shutil.copyfile(original, cache / "tests/clock.py")
    for asset in assets:
        (cache / asset).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(tmp_path / asset, cache / asset)
    binding = dict(candidate=revision, source=digest(collect_snapshot(cache)))
    (source / "tests").mkdir()
    (source / "tests/clock.py").write_text("value = 2")
    checked_price_source(tmp_path, source, binding)
    model = source / "src/oeis_learn/generalization/progressive/models.py"
    model.parent.mkdir(parents=True)
    model.write_text("changed = True")
    with pytest.raises(ValueError, match="workload changed"):
        checked_price_source(tmp_path, source, binding)
    (cache / "tests/clock.py").write_text("tampered")
    with pytest.raises(ValueError, match="cached candidate differs"):
        checked_price_source(tmp_path, source, binding)


def test_broader_timings_share_ordinary_weight_count_repeated_slots_and_charge_both_recipes(
    velocity_supply_rows,
):
    from copy import deepcopy
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.generalization.progressive.models import construct, resolve_profile
    from oeis_learn.generalization.progressive.training_rows import matching_pairs
    from oeis_learn.pipeline_validation import progressive_velocity_broad as broad

    rows = {r["draw_identity"]: r for r in velocity_supply_rows}
    keys = list(rows)
    pairs = matching_pairs([list(rows.values())], keys, seed=29)
    assert pairs["pairs"]
    repeated = next(iter(pairs["pairs"]))
    stream = seal(dict(order=[*keys, repeated]))
    model = resolve_profile(
        resolve({"version": "bigint-bootstrap/v1"}), seed=29, shape="smaller_width"
    )
    encoder, decoder, _ = construct(model, device="cpu")
    origin = dict(profile=model, initial_weights=model_snapshot(encoder, decoder))
    frozen = dict(
        weight=0.125, observation=digest("ordinary calibration"), measurement=digest("paid timing")
    )
    view = broad.projection(stream, rows, pairs, origin)
    assert len(view["rows"]) == len(keys) + 1
    assert len(view["donors"]) == pairs["eligible_rows"] + 1
    assert all(
        set(row) == {"tokens", "values"} and len(row["values"]) == 20
        for row in view["rows"].values()
    )
    design = broad.design(view, 8, frozen)
    observed = broad.run(view, design, device="cpu")
    prices = broad.prices(view, design, observed)
    assert set(prices) == {"01", "03"}
    assert all(r["tail_us"] >= r["expected_us"] > 0 for r in prices.values())
    assert design["cost"]["tokens"] == sum(
        s["positive_tokens"] for v in observed["methods"].values() for s in v["samples"].values()
    )
    assert set(broad.design(view, 8, dict(frozen, weight=None))["methods"]) == {"01"}
    forged = deepcopy(observed)
    forged["methods"]["03"]["samples"]["0"]["eligible"] += 1
    with pytest.raises(ValueError, match="passes"):
        broad.prices(view, design, seal(forged))


@pytest.mark.parametrize("prospects", [True, False])
def test_broader_native_preparation_replays_slots_and_prices_only_viable_recipes(
    tmp_path, monkeypatch, matched_composition_inputs, prospects
):
    from pathlib import Path
    from types import SimpleNamespace
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.generalization.progressive.contracts import (
        atomic_write,
        digest,
        load,
        reference,
        seal,
    )
    from oeis_learn.generalization.progressive.models import construct, resolve_profile
    from oeis_learn.generalization.progressive.velocity_records import SESSION
    from oeis_learn.pipeline_validation import (
        progressive_velocity_broad as broad,
        progressive_velocity_broad_host as host,
    )
    from oeis_learn.pipeline_validation import supervision
    from oeis_learn.pipeline_validation.progressive_generator_probe import usage_summary
    from oeis_learn.pipeline_validation.progressive_velocity_supply import _area

    x, _ = matched_composition_inputs
    source = Path(__file__).resolve().parents[2]
    folder = tmp_path / "run-state/discovery" / SESSION
    image, machine = digest("test image"), digest("test hardware")
    pointer = seal(dict(grant=digest("ordinary-grant"), receipt=digest("ordinary-measurement")))
    atomic_write(folder / "preparation/update-prices.json", pointer)
    atomic_write(
        _area(tmp_path, "families") / "funded.json",
        seal(dict(kind="families", grant=digest("families-grant"), receipt=digest("families"))),
    )
    revision = "a" * 40
    monkeypatch.setattr(host, "archive_size", lambda *_: 8192)
    stream = seal(dict(x["stream"], law=x["law"]))
    plan = dict(stream=stream["identity"], start=25, draws=2, slots=4, input_bytes=8192)
    monkeypatch.setattr(host, "preparation_plan", lambda root: plan)
    grant = load(host.authorize(tmp_path, 5, revision))
    model = resolve_profile(
        resolve({"version": "bigint-bootstrap/v1"}),
        seed=x["stream"]["seeds"]["initialization"],
        shape="smaller_width",
    )
    encoder, decoder, _ = construct(model, device="cpu")
    origin = dict(profile=model, initial_weights=model_snapshot(encoder, decoder))
    calls = []

    def custody(argv, out, **kwargs):
        calls.append("custodian")
        cal = dict(
            weight=0.25, observation=digest("fixed ordinary weight"), measurement=pointer["receipt"]
        )
        plan = seal(
            dict(
                **origin,
                stream=seal(dict(x["stream"], law=x["law"])),
                ordinary_rows=x["shards"][0],
                familiar=x["familiar"],
                difficulty=x["calibration"],
                eligible_ordinals=[25, 26] if prospects else [],
                pair_seed=x["stream"]["seeds"]["training_families"],
                microbatch=8,
                calibration=cal,
            )
        )
        atomic_write(out / "training.json", plan)
        atomic_write(
            out / "custody.json",
            seal(
                dict(
                    origins=load(Path(argv[-1]))["origins"],
                    ordinary_slots=4, stream=stream["identity"],
                    drawn=2,
                    physical=machine,
                    calibration=cal,
                    counts={},
                    disposition="eligible",
                    sources=[],
                    training=reference(out, out / "training.json"),
                )
            ),
        )
        atomic_write(
            out / "process.json", dict(outcome="COMPLETE", returncode=0, cleanup_failure=None)
        )

    monkeypatch.setattr(supervision, "launch", custody)
    base = seal(dict(image=image))
    monkeypatch.setattr(host, "cached_runtime", lambda *_: base)

    def package(root, source, area, out, revision, parent, src, worker, stage, seconds):
        paths = [out / "bootstrap.py", out / "candidate.whl"]
        for path in paths:
            path.write_bytes(b"fixture")
        return seal(
            dict(parent=base, bootstrap=reference(area, paths[0]), wheel=reference(area, paths[1]))
        )

    monkeypatch.setattr(host, "package_worker", package)
    monkeypatch.setattr(host, "package_evidence", lambda *_: None)
    monkeypatch.setattr(host, "physical_scope", lambda *_args, **_kwargs: machine)

    def engine_stage(
        physical, source, image, out, argv, *, gpu, record_lease, readonly_inputs, seconds
    ):
        requests = {dest: p for p, dest in readonly_inputs}
        req = load(requests["/work/inputs/request.json"])
        binds = [
            dict(Type="bind", Source=str(p), Destination=d, RW=False) for p, d in readonly_inputs
        ]
        binds.append(dict(Type="bind", Source=str(out), Destination="/work/output", RW=True))
        atomic_write(
            out / "container-inspect.json",
            [
                dict(
                    Image=image,
                    HostConfig=dict(NetworkMode="none", ReadonlyRootfs=True, Privileged=False),
                    Mounts=binds,
                )
            ],
        )
        atomic_write(
            out / "process.json", dict(outcome="COMPLETE", returncode=0, cleanup_failure=None)
        )
        atomic_write(out / "cleanup.json", dict(acknowledged=True, within_bound=True))
        calls.append(req["role"])
        assert record_lease is gpu and req["role"] == argv[4]
        if not gpu:
            assert "protected" not in req
            produced = broad.prepare(req, out, x["client"])
        else:
            assert set(requests) == {
                "/work/inputs/request.json",
                "/work/inputs/projection.json",
                "/work/inputs/candidate-wheel.py",
                "/work/inputs/candidate.whl",
            }
            view = load(requests["/work/inputs/projection.json"])
            observed = broad.run(view, req["design"], device="cpu")
            # This fixture emulates the HIP producer. The actual backend is CPU.
            observed = seal(
                dict(observed, calibration=seal(dict(observed["calibration"], device="cuda")))
            )
            elapsed = (
                1
                + observed["calibration"]["elapsed_us"]
                + sum(
                    m["startup_us"] + sum(s["elapsed_us"] for s in m["samples"].values())
                    for m in observed["methods"].values()
                    if "samples" in m
                )
            )
            (out / "telemetry").mkdir()
            (out / "telemetry/samples.jsonl").write_text("")
            atomic_write(out / "gpu-lease.json", dict(elapsed_us=elapsed + 1))
            produced = seal(
                dict(
                    device="hip",
                    request=req["identity"],
                    matrix=observed,
                    elapsed_us=elapsed,
                    worker_cpu_us=1,
                    peak_device_bytes=1,
                    telemetry=usage_summary(out / "telemetry", elapsed),
                )
            )
        atomic_write(out / "producer.json", produced)

    monkeypatch.setattr(host, "stage", engine_stage)
    engine = SimpleNamespace(image=lambda k: k)

    def qualify():
        return host.qualify(
            None,
            tmp_path,
            folder,
            source,
            revision,
            grant,
            lambda *a, **kw: SimpleNamespace(engine=engine),
        )

    # A grant one draw short must leave the custodian and native service untouched.
    small = seal(dict(grant, resources=dict(grant["resources"], draws=1)))
    refused, code = host.qualify(None, tmp_path, folder, source, revision, small,
                                 lambda *a, **kw: pytest.fail("unfunded worker"))
    assert code == 2 and not calls and refused["elapsed_us"] == 0

    from oeis_learn.pipeline_validation import progressive_research_resources as resources
    check = resources.velocity_capacity
    paused = True
    def native_pressure(root, *, owners, floors):
        value = check(root, owners=owners, floors=floors)
        if paused and owners[0]["owner"] == "broader prepare":
            value["mounts"][0]["deficit_bytes"] = 1024
        return value
    monkeypatch.setattr(resources, "velocity_capacity", native_pressure)
    stopped, code = qualify()
    assert code == 2 and stopped["stage"] == "prepare" and stopped["deficit_bytes"] == 1024
    assert calls == ["custodian"]
    paused = False
    first, code = qualify()
    assert code == 0, first
    assert first["receipt"]["disposition"] == (
        "prepared" if prospects else "no_breadth_at_matched_exposure"
    )
    assert (first["charged_tokens"] > 0) is prospects
    assert qualify() == (first, 0) and host.inventory(tmp_path, source) == first["receipt"]
    assert calls == ["custodian", "prepare-broad"] + (["time-broad"] if prospects else [])
    debits = first["receipt"]["preparation_work"]
    assert debits["draws"] == 2
    assert debits["native_work"] >= 4 + len(x["ordinals"] if prospects else [])
    assert first["receipt"]["drawn"] == 2
    # Interrupt after nested producer installation, before the host publishes its boundary.
    from oeis_learn.generalization.progressive.budgets import BudgetJournal
    from oeis_learn.pipeline_validation.progressive_velocity_operations import run_operation
    recovery = tmp_path / "separate-recovery"
    ledger = BudgetJournal(recovery / "ledger", grant["identity"], grant["resources"])
    from oeis_learn.generalization.progressive.velocity_records import operation_key
    from oeis_learn.generalization.progressive.budgets import KEYS
    original_area = folder / "qualification" / grant["identity"][7:]
    binding = load(original_area / "binding.json")
    coords = host.coordinates(grant, "prepare")
    logical = operation_key(coords, 0)["logical"]
    block = original_area / "operations" / logical[7:]
    for name in ("boundary-0000.json", "receipt.json"):
        (block / name).unlink()
    work = load(block / "input.json")["work"]
    ledger.reserve(logical, "dispatch", 0, dict(work, elapsed_us=grant["total_us"]), boot="lost", monotonic_us=0)
    args = dict(area=original_area, journal=ledger, grant=grant, coordinates=coords,
                request=load(block / "input.json")["request"], cost={k: work[k] for k in KEYS if work[k]}, execute=lambda *_: pytest.fail("redispatch"),
                installed=lambda out: (out / "native/producer.json").is_file(), verify=lambda out: host.cpu_evidence(tmp_path, original_area, binding, out))
    restored = run_operation(**args)
    assert run_operation(**args) == restored and restored["evidence"]["disposition"] == first["receipt"]["disposition"]
    assert ledger.snapshot()["remaining"]["elapsed_us"] == 0


@pytest.mark.parametrize("collision", [False, True])
def test_broader_custodian_checks_reserved_shift_before_native_projection(
    tmp_path, monkeypatch, matched_composition_inputs, collision
):
    from oeis_learn.generalization.progressive.contracts import atomic_write, load, reference, seal
    from oeis_learn.generalization.progressive.velocity_records import SESSION
    from oeis_learn.pipeline_validation import progressive_velocity_broad_host as host
    from oeis_learn.pipeline_validation import progressive_panel_sources
    from oeis_learn.pipeline_validation.progressive_round028_audit import ORIGINAL

    args, witnesses = matched_composition_inputs
    (tmp_path / "run-state/discovery" / SESSION).mkdir(parents=True)
    panel = tmp_path / "panel"
    panel_id = digest("original panel")
    difficulty = args["calibration"]
    stock = seal(
        dict(
            version="round028-private-panel-supply/v1",
            registration=panel_id,
            calibrated=True,
            allocations=dict(calibration=difficulty["identity"]),
            witnesses={},
        )
    )
    atomic_write(panel / "supply.json", stock)
    atomic_write(panel / "difficulty-map.json", difficulty)
    held = panel / "calibration-reservation.json"
    atomic_write(held, seal(dict(registration=panel_id, witnesses={})))
    original = seal(dict(payload={"references": []}))
    atomic_write(tmp_path / ORIGINAL / "prepare-research/frozen/campaign.json", original)
    monkeypatch.setattr(host, "learned_search_policy", lambda: dict(campaign=original["identity"]))
    census, data = digest("exposures"), digest("donor data")
    stream = seal(dict(args["stream"], law=args["law"]))
    stock_ref = reference(tmp_path, panel / "supply.json")
    package = dict(
        stream=stream,
        rows=args["shards"][0],
        native_draws=42,
        profile={},
        microbatch=8,
        exposure_cut=digest([census, data, stock["identity"], reference(panel, held)]),
    )
    monkeypatch.setattr(
        host, "training_source", lambda _: (dict(roster_source=stock_ref), package, [stock_ref])
    )
    monkeypatch.setattr(
        host,
        "paid_calibration",
        lambda *_: (
            dict(weight=0.2, observation=digest("cal"), measurement=digest("paid")),
            digest("weights"),
            digest("device"),
        ),
    )
    families = dict(panels={"assessment": [{"task": "reserved"}]})
    monkeypatch.setattr(host, "funded_inventory", lambda *_a, **_kw: ({}, families))
    monkeypatch.setattr(host, "DRAW_BLOCK", 2)
    reserved = dict(
        id="reserved",
        stratum="oeis",
        indices=list(map(str, range(100))),
        values=[str((1 << 200) + i) for i in range(100)],
    )
    reserved["group"] = "replication-family"
    sibling = dict(reserved, id="unchosen", values=[str((1 << 180) + i) for i in range(100)])
    if collision:
        sibling["values"][:20] = witnesses[0]["values"][80:]
    monkeypatch.setattr(
        progressive_panel_sources,
        "read_inputs",
        lambda *_a, **_kw: dict(
            census=census,
            data=dict(identity=data),
            candidates=[reserved, sibling],
            historical=[],
            originals_used=[],
            support={"structures": args["familiar"]},
        ),
    )
    monkeypatch.setattr(host, "practice_prompts", lambda: [])
    out = tmp_path / "custody"
    out.mkdir()
    host.custodian(tmp_path, tmp_path, out, {"families": {}})
    receipt = load(out / "custody.json")
    if collision:
        assert receipt["disposition"] == "requires_fresh_disjoint_start"
        assert (
            receipt["counts"]["excluded"] > 0
            and receipt["drawn"] == 0
            and receipt["training"] is None
        )
    else:
        assert receipt["disposition"] == "eligible" and receipt["drawn"] == 2
        training = load(out / receipt["training"]["path"])
        assert set(training["eligible_ordinals"]) <= {42, 43}
        assert "reserved" not in str(training) and "protected" not in training
        from oeis_learn.pipeline_validation.progressive_velocity_broad import checked_request

        request = host.native_request(
            dict(identity=digest("grant"), source=digest("source")),
            dict(resources={"native_work": 3}),
            dict(parent={"image": digest("image")}),
            training,
        )
        witnesses = len(training["stream"]["order"]) + len(training["eligible_ordinals"])
        assert request["native_pair_limit"] == max(0, 3 - witnesses) // 2
        with pytest.raises(ValueError, match="closed training-only"):
            checked_request(seal(dict(request, hidden_suffix=reserved["values"])), "prepare-broad")


@pytest.mark.parametrize("indexed", [False, True])
def test_broad_audit_preserves_declared_parameter_exception(tmp_path, matched_composition_inputs, indexed):
    import re
    from oeis_learn.generalization.progressive import generator
    from oeis_learn.generalization.progressive.contracts import reference, seal
    from oeis_learn.pipeline_validation.progressive_preparation import prepare_velocity_broad, verify_draw
    from oeis_learn.pipeline_validation.progressive_research_inputs import ExposureIndex
    from oeis_learn.synthesis.reference import evaluate

    args, _ = matched_composition_inputs
    draw = next(d for i in range(256) if re.fullmatch(r"\(program \(return \(int -?[0-9]+\)\)\)",
                (d := generator.draw(args["law"], i))["source"]))
    row, reason = verify_draw(draw, client=args["client"])
    assert reason is None
    stream = seal(dict(args["stream"], order=[row["draw_identity"]], row_identities=digest([row["identity"]])))
    source = "(program (return (int 4321)))"
    old = dict(id="prior", source=source, structure=generator.structure_id(source),
               indices=row["indices"], values=list(evaluate(source, row["indices"]).values))
    for stratum in ("parameter", "composition"):
        prior = dict(old, stratum=stratum)
        history, indexes = [prior], []
        if indexed:
            path = tmp_path / (stratum + ".sqlite")
            with ExposureIndex(path) as db:
                db.add(prior, origin="prior", role="assessment")
            history, indexes = [], [reference(tmp_path, path)]
        settings = dict(args, stream=stream, shards=[[row]], ordinals=[], protected=history,
                        indexes=indexes, familiar=[row["structure"]])
        if stratum == "parameter":
            broader, _, _ = prepare_velocity_broad(**settings)
            assert broader["order"] == stream["order"]
        else:
            with pytest.raises(ValueError, match="overlaps protected"):
                prepare_velocity_broad(**settings)
