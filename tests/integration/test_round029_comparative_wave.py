import pytest

from oeis_learn.generalization.progressive.campaign import velocity_registration
from oeis_learn.generalization.progressive.contracts import digest, seal
from oeis_learn.generalization.progressive.graph import velocity_study_looks
from oeis_learn.generalization.progressive.velocity_records import validate_grant, physical_root_id
from oeis_learn.pipeline_validation import progressive_velocity_wave as wave


def fixture_wave(methods):
    registration = velocity_registration(source="a" * 40, design=digest("design"), initialization=9,
        stream_seed=42, development=[dict(task="case", group="group", stratum="oeis")],
        retention=[dict(task="other", group="family", stratum="retention")],
        ordinary=digest("stream"), broad=digest("broader"))
    graph = velocity_study_looks(registration)
    packages = {}
    for m in methods:
        packages[m] = seal(dict(method=m, policy=dict(effective_batch=16), order=["s"],
                                rows={"s": dict(tokens=list(range(400)))}))
    schedule = [dict(method=m, look=l["look"], profile=p, members=len(l["members"]),
                     roster=l["identity"]) for m, profiles in graph.items()
                if m in methods or m not in {f"0{i}" for i in range(7)}
                for p, looks in profiles.items() for l in looks]
    accepted = seal(dict(registration=registration["identity"], supported=methods,
                    omitted={m: digest(["omitted", m]) for m in graph if m.isdigit() and m not in methods},
                    trials={m: digest(m) for m in graph if m.isdigit()}, schedule=schedule,
                    training=[dict(method=m, target=t, learner=packages[m]["identity"])
                              for m in methods for t in (0, 1_000_000, 4_000_000)],
                    followup_us=5_000_000, proposed_total_us=75_000_000))
    return accepted, registration, packages


@pytest.mark.parametrize("methods", [["00", "04", "05", "06"], [f"0{i}" for i in range(7)]])
def test_frozen_wave_has_separate_endpoints_exact_crossings_and_one_allowance(tmp_path, methods):
    accepted, reg, packages = fixture_wave(methods)
    graph, looks, cuts, ops = wave.plan(accepted, reg, packages)
    assert len(graph) == 11
    assert len(ops) == 11 * len(methods) + 8
    assert cuts["00"][1_000_000]["useful_tokens"] == 1_004_800
    train = [o for o in ops if o["role"] == "train"]
    assert [looks[o["roster"]]["look"] for o in train] == sorted(looks[o["roster"]]["look"] for o in train)
    assert sum(o["cost"]["tokens"] for o in train) == cuts["00"][4_000_000]["useful_tokens"] * len(methods)
    assert all(o["roster"] in looks for o in ops)
    grant, approval, marker = wave.authority(tmp_path, "b"*40, 2, accepted, ops, digest("driver"))
    assert marker["grant"] == grant["identity"]
    assert grant["holdback_us"] == accepted["followup_us"]
    validate_grant(grant, approval, root=physical_root_id(tmp_path), parent=accepted["identity"], action="run")
    with pytest.raises(ValueError, match="tail"):
        wave.authority(tmp_path, "b"*40, 1, accepted, ops, digest("driver"))
    assert wave.authority(tmp_path, "b"*40, 2, accepted, ops, digest("other"))[2] != marker

