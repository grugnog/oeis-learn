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
