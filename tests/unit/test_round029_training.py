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
    from oeis_learn.pipeline_validation.progressive_study import open_velocity_study, close_velocity_study

    development = [dict(task="a", group="f", stratum="oeis"), dict(task="b", group="f", stratum="oeis")]
    retention = [dict(task="r", group="known", stratum="retention")]
    args = dict(source="a" * 40, design=digest("frozen hypotheses, workers and prices"),
                development=development, retention=retention, initialization=53, stream_seed=67,
                ordinary=digest("ordinary stream"), broad=digest("broad stream"))
    reg = velocity_registration(**args)
    journal = open_velocity_study(tmp_path, reg)
    assert len(journal.trials) == 7
    assert [journal.velocity_trial(f"{i:02d}")["binding"]["recipe"]["id"] for i in range(7)] == [
        "00", "01", "02", "03", "04", "05", "06"]
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
    assert open_velocity_study(tmp_path, reg).observe_velocity(reg, cheap[0], evidence=digest("zero")) == first
    from oeis_learn.generalization.progressive.budgets import KEYS
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.generalization.progressive.velocity_records import SESSION, physical_root_id
    accepted = seal(dict(registration=reg["identity"], scope="exploration"))
    authority = physical_root_id(tmp_path)
    terms = dict(version="round029-grant/v1", study_id=SESSION, scope="cpu", state_root_id=authority,
                 registration_or_parent_ref=accepted["identity"], total_us=100, holdback_us=10,
                 allowed_operations=["cpu", "structure_search"], resources=dict(dict.fromkeys(KEYS, 0), elapsed_us=100))
    approval = seal(dict(version="round029-authorization/v1", terms=digest(terms)))
    grant = seal(dict(terms, authorizing_record_ref=approval["identity"]))
    funded = dict(decision=digest("observed deficit"), reason="price a fixed-order comparison",
                  grant=grant, approval=approval, accepted=accepted, root_id=authority)
    assert journal.record_velocity_branch("structure_search", **funded)["grant"] == grant["identity"]
    assert journal.record_velocity_branch("paired_extension", decision=first, reason="unfunded")["state"] == "unused"
    with pytest.raises(ValueError):
        journal.record_velocity_branch("confirmation", **funded)
    boundary = close_velocity_study(tmp_path, reg, state="resumable", evidence=first)
    child = velocity_registration(**dict(args, initialization=71, stream_seed=99, ordinary=digest("new stream"),
                                         development=[dict(task="new", group="fresh", stratum="oeis")],
                                         retention=[dict(task="new-r", group="fresh-r", stratum="retention")]),
                                  scope="confirmation", parent=boundary, finalist="03")
    assert set(velocity_study_looks(child)) == {"00", "03", "D"}
    assert len(open_velocity_study(tmp_path, child).trials) == 2
    with pytest.raises((FileNotFoundError, ValueError)):
        open_velocity_study(tmp_path / "foreign", child)
    with pytest.raises(ValueError):
        velocity_registration(**args, scope="confirmation", parent=dict(boundary, state="running"), finalist="03")


def test_whole_update_crossing_after_discard_and_seed_provenance():
    from oeis_learn.generalization.progressive.velocity_trials import crossed

    stream = {"order": ["a", "b"]}
    rows = {"a": {"tokens": [1, 2, 3]}, "b": {"tokens": [4]}}
    first = exposure(stream, rows, updates=1, batch=16, discarded=64)
    second = exposure(stream, rows, updates=2, batch=16, discarded=64)
    assert (first["useful_tokens"], first["charged_tokens"]) == (32, 96)
    assert not crossed(first, 48) and crossed(second, 48) and not crossed(second, 32)
    args = dict(initialization=9, stream_seed=21, ordinary=digest("ordinary"), broad=digest("broad"))
    before = recipe_bindings(digest("study"), **args)
    for changed in (dict(initialization=10), dict(stream_seed=22), dict(ordinary=digest("other ordinary"))):
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


def test_broad_preparation_rechecks_witnesses_preserves_slots_and_excludes_protected(tmp_path):
    from oeis_learn.generalization.progressive import generator, training_rows
    from oeis_learn.generalization.progressive.complexity import measure
    from oeis_learn.generalization.progressive.contracts import seal
    from oeis_learn.generalization.progressive.program_search_panels import descriptor, bin_key
    from oeis_learn.pipeline_validation.progressive_preparation import (
        prepare_velocity_broad,
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
