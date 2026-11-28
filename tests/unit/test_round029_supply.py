import pytest

from oeis_learn.generalization.progressive.program_search_panels import reserve_velocity_families


def case(i, group, stratum="oeis", **extra):
    return dict(
        id=str(i),
        indices=[str(j - 7) for j in range(100)],
        values=[str(i * 500 + j) for j in range(100)],
        group=group,
        stratum=stratum,
        **extra,
    )


def test_shift_old_reservation_and_composition_bridges_are_excluded():
    old = case(1, "old", composition="b")
    bridge = case(2, "new", composition="b")
    duplicate = case(3, "other")
    duplicate["values"][:20] = old["values"][75:95]
    clean = [case(i, str(i)) for i in range(4, 10)]
    roles = {"replication": {"oeis": 2}, "assessment": {"oeis": 2}}
    receipt = reserve_velocity_families(
        [duplicate, bridge, *clean], [old], indexes=[], quotas=roles, seed=42
    )
    assert receipt["excluded"] == 2
    assert {v["difficulty"] for rows in receipt["panels"].values() for v in rows} == {"unknown"}
    panels = list(receipt["panels"].values())
    assert not {v["family"] for v in panels[0]} & {v["family"] for v in panels[1]}
    assert all(len({r["family"] for r in rows}) == 2 for rows in panels)


def test_live_target_must_exclude_every_grounding_on_donor():
    from types import SimpleNamespace
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.generalization.progressive.training_rows import matching_proof
    from oeis_learn.synthesis.admission import VisiblePrompt

    prompt = VisiblePrompt(tuple(map(str, range(20))), tuple("7" for _ in range(20)))

    def check(src, outcome, bits, replay=True):
        native = {
            "outcome": outcome,
            "completed_bounds": [bits],
            "coefficient_bits": bits,
            "evidence": {"test": True},
        }
        client = SimpleNamespace(
            fit=lambda *a, **kw: dict(outcome=outcome, native=native),
            runtime_identity=lambda _: "test",
            reference_sha256="test",
        )
        return matching_proof(encode_body(src), prompt, client=client, replay=lambda _: replay)[
            "status"
        ]

    assert check("(program (return $index))", "unknown", 16) == "eligible"
    literal = "(program (return (int 1)))"
    assert check(literal, "verified_solution", 16) == "regroundable"
    assert check(literal, "resource_exhausted", 16) == "unfinished"
    assert check(literal, "timeout", 16) == "timed_out"
    def expired(*args, **kwargs):
        raise TimeoutError
    assert matching_proof(encode_body(literal), prompt, client=SimpleNamespace(fit=expired))["status"] == "timed_out"
    assert check(literal, "unsat_in_scope", 16) == "partial_scope"
    assert check(literal, "unsat_in_scope", 4096, False) == "unverified"
    assert check(literal, "unsat_in_scope", 4096) == "eligible"


def test_installed_literal_can_refit_and_full_scope_negative_replays(fitting_client):
    from oeis_learn.pipeline_validation.progressive_velocity_worker import literal_probes

    outcomes = literal_probes(fitting_client)
    assert outcomes["constant"]["status"] == "regroundable"
    varying = outcomes["varying"]
    assert varying["status"] == "eligible"
    native = varying["proof"]["result"]["native"]
    assert native["coefficient_bits"] == 4096 and 4096 in native["completed_bounds"]


def test_reopened_index_preserves_compositions_and_both_sides_of_exception(tmp_path):
    from oeis_learn.generalization.progressive.generator import structure_id
    from oeis_learn.pipeline_validation.progressive_research_inputs import ExposureIndex
    from oeis_learn.synthesis.reference import evaluate

    def witness(i, body, stratum):
        source = f"(program (return {body}))"
        row = case(i, f"group-{i}", stratum, source=source, structure=structure_id(source))
        row["values"] = list(evaluate(source, row["indices"]).values)
        return row

    old = witness(1, "(add $index (int 500))", "composition")
    flipped = witness(2, "(add (int 1000) $index)", "composition")
    variant = witness(3, "(add $index (int 1000))", "parameter")
    cases = [(old, flipped, 0), (old, variant, 0), (dict(old, stratum="parameter"), variant, 1)]
    for i, (historical, candidate, expected) in enumerate(cases):
        path = tmp_path / f"prior-{i}.sqlite"
        with ExposureIndex(path) as index:
            index.add(historical, origin="old panel", role="assessment")
        args = dict(quotas={"replication": {candidate["stratum"]: 1}}, seed=29)
        direct = reserve_velocity_families([candidate], [historical], indexes=[], **args)
        with ExposureIndex(path, readonly=True) as index:
            reopened = reserve_velocity_families([candidate], [], indexes=[index], **args)
        assert direct["eligible"] == reopened["eligible"] == expected
        assert direct["panels"] == reopened["panels"]


def test_legacy_exposure_stays_readable_and_unknown_provenance_is_reported(tmp_path):
    import sqlite3
    from oeis_learn.pipeline_validation.progressive_research_inputs import ExposureIndex

    path = tmp_path / "legacy.sqlite"
    old = case(1, "reserved", stratum="composition", composition="shared")
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE keys(kind TEXT, value TEXT, PRIMARY KEY(kind,value)) WITHOUT ROWID")
        db.execute("CREATE TABLE origins(name TEXT PRIMARY KEY, rows INTEGER NOT NULL)")
        db.execute("INSERT INTO origins VALUES ('prior',1)")
        db.executemany("INSERT INTO keys VALUES (?,?)", ExposureIndex.keys(old))
    candidate = case(2, "new", stratum="composition", composition="shared")
    before = path.read_bytes()
    with ExposureIndex(path, readonly=True) as index:
        assert "window" in index.matches(old)
        result = reserve_velocity_families([candidate], [], indexes=[index],
                                          quotas={"replication": {"composition": 1}}, seed=29)
    assert result["eligible"] == 0 and result["provenance_unavailable"] == 1
    assert "provenance" in result["provenance_reason"]
    assert path.read_bytes() == before


def test_verified_pair_rows_families_and_exclusions(velocity_supply_rows):
    from oeis_learn.generalization.progressive.training_rows import matching_pairs

    rows = velocity_supply_rows
    receipt = matching_pairs([rows], [r["draw_identity"] for r in rows], seed=29)
    assert (receipt["positive_rows"], receipt["eligible_rows"], receipt["eligible_families"]) == (3, 2, 2)
    assert receipt["dispositions"] == {"eligible": 2, "no_differing_donor": 1}
    assert {p["family"] for p in receipt["pairs"].values()} == {rows[0]["structure"], rows[2]["structure"]}
    zero = matching_pairs([rows], [r["draw_identity"] for r in rows[:2]], seed=29)
    assert zero["eligible_rows"] == zero["eligible_families"] == 0
    assert zero["positive_rows"] == zero["dispositions"]["no_differing_donor"] == 2


def test_all_protected_roles_get_distinct_cases_and_families_before_second_slots():
    import pytest

    roles = ("development", "calibration", "teacher", "practice", "replication",
             "development_retention", "replication_retention", "assessment")
    cases = [case(i, str(i // 2), stratum="parameter") for i in range(1, 49)]
    quotas = {role: {"parameter": 2} for role in roles}
    frozen = reserve_velocity_families(cases, [], indexes=[], quotas=quotas, seed=613)
    allocated = [r for panel in frozen["panels"].values() for r in panel]
    assert len(allocated) == len({r["task"] for r in allocated}) == len({r["family"] for r in allocated}) == 16
    assert frozen["deficits"] == {}
    assert frozen == reserve_velocity_families(list(reversed(cases)), [], indexes=[], quotas=quotas, seed=613)
    with pytest.raises(ValueError, match="synthetic"):
        reserve_velocity_families(cases, [], indexes=[], quotas={"calibration": {"oeis": 1}}, seed=613)


def test_retention_claims_familiar_families_alongside_development():
    rows = [case(i, str(i), "parameter", structure=str(i)) for i in range(8)]
    quotas = {"development": {"parameter": 8}, "development_retention": {"retention": 8}}
    result = reserve_velocity_families(rows, [], indexes=[], quotas=quotas, seed=23)
    panels = result["panels"]
    assert [len(panels[role]) for role in quotas] == [4, 4]
    assert {r["stratum"] for r in panels["development_retention"]} == {"retention"}
    assert len({r["family"] for rows in panels.values() for r in rows}) == 8


def test_prior_short_practice_prompt_blocks_a_shift_and_its_linked_family():
    first, bridge, independent = case(1, "shared"), case(2, "shared"), case(3, "free")
    prompt = dict(values=first["values"][12:32], stratum="practice")
    result = reserve_velocity_families(
        [first, bridge, independent],
        [],
        indexes=[],
        quotas={"development": {"oeis": 3}},
        seed=19,
        exposed_prompts=[prompt],
    )
    assert result["excluded"] == 2
    assert [r["task"] for r in result["panels"]["development"]] == ["3"]


def test_original_relation_provenance_requires_every_index_origin(tmp_path):
    import pytest
    from oeis_learn.pipeline_validation.progressive_panel_sources import verify_origins
    from oeis_learn.pipeline_validation.progressive_research_inputs import ExposureIndex

    index = tmp_path / "exposures.sqlite"
    with ExposureIndex(index) as db:
        db.add(case(1, "old"), origin="one", role="training")
        db.add(case(2, "other"), origin="two", role="training")
    verify_origins(index, {"one": 1, "two": 1})
    with pytest.raises(ValueError, match="origin"):
        verify_origins(index, {"one": 2})


def test_prior_calibration_and_every_pilot_training_member_are_protected():
    import pytest
    from oeis_learn.generalization.progressive.contracts import digest, seal
    from oeis_learn.pipeline_validation.progressive_velocity_census import prior_exposures

    prior = case(1, "old", declared_groups=["linked"])
    held = case(2, "held")
    training = dict(case(3, "training"), draw_identity="draw", source=None)
    calibration = seal(dict(registration="prior-reg", witnesses={"reserved": digest(held)}))
    supply = dict(registration="prior-reg", witnesses={"member": prior})
    inputs = dict(candidates=[held], historical=[], support={"structures": {}})
    history = prior_exposures(inputs, supply, calibration, [training])
    assert [(r["id"], r["group"]) for r in history] == [
        ("1", "old"),
        ("2", "held"),
        ("3", "training"),
    ]
    assert history[0]["declared_groups"] == ["linked"]
    with pytest.raises(ValueError, match="calibration"):
        prior_exposures(dict(inputs, candidates=[]), supply, calibration, [training])


@pytest.fixture
def adjustable_matching_rows(velocity_supply_rows):
    from oeis_learn.decoder.bigint_codec import encode_body
    from oeis_learn.generalization.progressive.contracts import digest, seal
    from oeis_learn.generalization.progressive.generator import structure_id

    rows = list(velocity_supply_rows)
    source = "(program (return (int 180)))"
    values = ["180"] * 100
    rows[1] = seal(
        dict(
            rows[1],
            source=source,
            target_source=source,
            tokens=encode_body(source),
            values=values,
            behavior=digest(values),
            structure=structure_id(source),
        )
    )
    return rows


def test_matching_selection_reopens_trial_seed_and_full_scope_native_proof(
    adjustable_matching_rows, monkeypatch
):
    from types import SimpleNamespace
    from oeis_learn.generalization.progressive.contracts import digest, seal
    from oeis_learn.generalization.progressive.training_rows import matching_pairs
    from oeis_learn.synthesis import replay

    rows = adjustable_matching_rows
    order = [r["draw_identity"] for r in rows]
    native = dict(
        outcome="unsat_in_scope",
        completed_bounds=[16, 64, 256, 1024, 4096],
        coefficient_bits=4096,
        evidence={"fixture": "only a constant can be grounded"},
    )
    client = SimpleNamespace(
        fit=lambda *a, **kw: dict(outcome="unsat_in_scope", native=native),
        runtime_identity=lambda _: digest("native"),
        reference_sha256=digest("reference"),
    )
    monkeypatch.setattr(replay, "replay_native", lambda r: r["result"]["native"] == native)
    results = [matching_pairs([rows], order, seed=k, client=client) for k in (1, 2, 3, 29)]
    assert len({digest(p["donor_map"]) for p in results}) == 2
    receipt = results[0]
    assert receipt["coverage"]["full"] and receipt["pairs"][order[1]]["proof"]["proof"][
        "runtime"
    ] == digest("native")
    assert matching_pairs([rows], order, seed=1, recorded=receipt) == receipt
    with pytest.raises(ValueError):
        matching_pairs([rows], order, seed=29, recorded=receipt)
    from copy import deepcopy

    forged = deepcopy(receipt)
    forged["donor_map"][order[1]] = order[1]
    with pytest.raises(ValueError, match="donor map"):
        matching_pairs([rows], order, seed=1, recorded=seal(forged))

    partial = matching_pairs([rows], order, seed=1)
    assert not partial["coverage"]["full"] and partial["coverage"]["unmeasured_native"] == 1

    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.fresh.bigint_diagnostics import model_snapshot
    from oeis_learn.generalization.progressive.models import construct, resolve_profile
    from oeis_learn.pipeline_validation import progressive_velocity_broad as broad

    model = resolve_profile(
        resolve({"version": "bigint-bootstrap/v1"}), seed=1, shape="smaller_width"
    )
    encoder, decoder, _ = construct(model, device="cpu")
    ordinary = seal(dict(order=order, seeds={"training_families": 1}))
    stream = seal(dict(order=[*order, order[1]], seed=1))
    data = dict(
        profile=model,
        initial_weights=model_snapshot(encoder, decoder),
        pair_seed=1,
        stream=ordinary,
        ordinary_rows=rows,
    )
    by_key = {r["draw_identity"]: r for r in rows}
    old = dict(
        weight=None,
        observation=digest("original calibration"),
        measurement=digest("original paid matrix"),
    )
    view, manifests = broad.matching_views(data, stream, by_key, client=client)
    assert broad.matching_views(data, stream, by_key, recorded=manifests) == (view, manifests)
    assert all(m == receipt for m in manifests.values())
    assert all(v["pair_proof"] == receipt["identity"] for v in view["streams"].values())
    for name, keys in (("ordinary", order), ("broader", stream["order"])):
        projected = view["streams"][name]
        assert [projected["donors"].get(k) for k in projected["order"][: len(keys)]] == [
            by_key[receipt["pairs"][k]["donor"]]["values"][:20] if k in receipt["pairs"] else None
            for k in keys
        ]
    plan = broad.design(view, 8, old)
    observed = broad.run(view, plan, device="cpu")
    assert observed["calibration"]["weight"] > 0
    assert set(broad.prices(view, plan, observed)) == {"01", "02", "03"}
    subset, _ = broad.matching_views(
        data, stream, by_key, recorded=dict(ordinary=partial, broader=receipt)
    )
    assert subset["support"]["ordinary"]["coverage"]["unmeasured_native"] == 1
    assert set(broad.design(subset, 8, old)["methods"]) == {"01"}

    limited = dict(data, native_pair_limit=1)
    funded, evidence = broad.matching_views(limited, stream, by_key, client=client)
    assert funded["support"]["ordinary"]["coverage"]["full"]
    assert funded["support"]["broader"]["coverage"]["unmeasured_native"] == 1
    assert broad.matching_views(limited, stream, by_key, recorded=evidence) == (funded, evidence)
    assert set(broad.design(funded, 8, old)["methods"]) == {"01", "02"}
    with pytest.raises(ValueError, match="funded native scope"):
        broad.matching_views(limited, stream, by_key, recorded=manifests)


def test_native_adjustable_selection_reopens_its_learner_manifest(
    adjustable_matching_rows, fitting_client
):
    from oeis_learn.generalization.progressive.training_rows import matching_pairs
    from oeis_learn.synthesis.replay import replay_native

    rows = adjustable_matching_rows
    keys = [r["draw_identity"] for r in rows]
    pairs = matching_pairs([rows], keys, seed=1, client=fitting_client)
    assert replay_native(pairs["pairs"][keys[1]]["proof"]["proof"])
    assert matching_pairs([rows], keys, seed=1, recorded=pairs) == pairs
