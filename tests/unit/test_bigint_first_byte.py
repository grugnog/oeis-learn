"""First-call attribution cannot invent growth, completed bytes or double-count time."""

from copy import deepcopy

import pytest
from tests.unit.test_bigint_benchmark import benchmark
from tests.unit.test_bigint_input_load_profile import input_report


def counts(requests=0, increase=0):
    return dict(
        requests=requests,
        allowed=requests,
        denied=0,
        grow_failed_callbacks=0,
        requested_increase_bytes=increase,
    )


def byte_report():
    data = input_report()
    for sample in data["samples"]:
        detail = sample["timings"]["stages"]["input_load"]
        detail["policy"] = "bigint-input-load-profile/v2"
        detail["first_byte_probe"] = dict(
            policy="bigint-first-byte-probe/v1",
            first_byte=dict(calls=1, seconds=0.00008),
            first_byte_calls_returned=1,
            remaining_transfer_seconds=0.00002,
            terms_observed=1,
            memory_probe_terms=1,
            memory_before_first=counts(1, 65536),
            memory_in_first=counts(1, 65536),
            memory_after_first=counts(),
            memory_events=[
                dict(
                    term=1,
                    scope="before_first",
                    ordinal=1,
                    current_bytes=0,
                    desired_bytes=65536,
                    allowed=True,
                ),
                dict(
                    term=1,
                    scope="first_byte",
                    ordinal=2,
                    current_bytes=65536,
                    desired_bytes=131072,
                    allowed=True,
                ),
            ],
            memory_events_omitted=0,
        )
    return data


def test_first_time_and_growth_remain_observations_inside_the_byte_parent():
    result = benchmark().input_load_summary(byte_report(), 1)
    assert result["policy"] == "bigint-input-load-profile/v2"
    probe = result["first_byte_probe"]
    assert probe["first_byte_warm"]["median_seconds"] == pytest.approx(0.00008)
    assert probe["remaining_transfer_warm"]["median_seconds"] == pytest.approx(0.00002)
    assert probe["memory_in_first"]["requests"] == [1, 1, 1]
    assert probe["later_byte_calls_attempted"] == [2, 2, 2]


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "policy",
        "mixed",
        "calls",
        "returned",
        "time",
        "negative",
        "probe_terms",
        "balance",
        "event_scope",
        "event_bytes",
        "omission",
        "too_many",
        "scope_order",
    ],
)
def test_forged_first_byte_or_memory_evidence_is_rejected(kind):
    data = byte_report()
    detail = data["samples"][0]["timings"]["stages"]["input_load"]
    p = detail["first_byte_probe"]
    if kind == "missing":
        del detail["first_byte_probe"]
    elif kind == "policy":
        p["policy"] = "unknown"
    elif kind == "mixed":
        detail["policy"] = "bigint-input-load-profile/v1"
    elif kind == "calls":
        p["first_byte"]["calls"] = 2
    elif kind == "returned":
        p["first_byte_calls_returned"] = 2
    elif kind == "time":
        p["first_byte"]["seconds"] = 1
    elif kind == "negative":
        p["remaining_transfer_seconds"] = -1
    elif kind == "probe_terms":
        p["memory_probe_terms"] = 0
    elif kind == "balance":
        p["memory_in_first"]["denied"] = 1
    elif kind == "event_scope":
        p["memory_events"][1]["scope"] = "compute"
    elif kind == "event_bytes":
        p["memory_events"][1]["current_bytes"] = 200000
    elif kind == "omission":
        p["memory_events_omitted"] = 1
    elif kind == "too_many":
        p["memory_events"] *= 9
    else:
        p["memory_events"][1]["scope"] = "after_first"
    with pytest.raises((ValueError, KeyError)):
        benchmark().input_load_summary(data, 1)


def test_v1_remains_readable_without_manufacturing_first_byte_data():
    old = input_report()
    before = deepcopy(old)
    summary = benchmark().input_load_summary(old, 1)
    assert summary["policy"] == "bigint-input-load-profile/v1" and "first_byte_probe" not in summary
    assert old == before


def test_trapping_first_byte_retains_growth_without_inventing_returned_bytes():
    data = byte_report()
    for sample in data["samples"]:
        d = sample["timings"]["stages"]["input_load"]
        d.update(
            literal_values_started=1,
            index_values_started=0,
            values_finished=0,
            byte_calls_attempted=1,
            byte_calls_returned=0,
        )
        for name, phase in d["phases"].items():
            phase["calls"] = int(name in ("begin_load", "magnitude_encode", "byte_transfer"))
            phase["seconds"] = 0.0001 if phase["calls"] else 0
        d["unattributed_seconds"] = 0.0007
        d["first_byte_probe"]["first_byte_calls_returned"] = 0
        d["first_byte_probe"]["memory_in_first"].update(allowed=0, denied=1)
        d["first_byte_probe"]["memory_events"][1]["allowed"] = False
    p = benchmark().input_load_summary(data, 1)["first_byte_probe"]
    assert p["first_byte_calls_returned"] == [0, 0, 0]
    assert p["later_byte_calls_attempted"] == p["later_byte_calls_returned"] == [0, 0, 0]
    assert p["memory_in_first"]["denied"] == [1, 1, 1]


def test_zero_growth_and_bounded_omissions_are_readable():
    for omitted in [0, 17]:
        data = byte_report()
        for sample in data["samples"]:
            p = sample["timings"]["stages"]["input_load"]["first_byte_probe"]
            p.update(
                memory_before_first=counts(21 if omitted else 0, (21 if omitted else 0) * 65536),
                memory_in_first=counts(),
                memory_events=[
                    dict(
                        term=1,
                        scope="before_first",
                        ordinal=k + 1,
                        current_bytes=k * 65536,
                        desired_bytes=(k + 1) * 65536,
                        allowed=True,
                    )
                    for k in range(4)
                ]
                if omitted
                else [],
                memory_events_omitted=omitted,
            )
        p = benchmark().input_load_summary(data, 1)["first_byte_probe"]
        assert p["memory_events_omitted"] == [omitted] * 3
        assert p["memory_in_first"]["requests"] == [0] * 3


@pytest.mark.parametrize(
    "kind",
    ["no_first", "missing_trace", "boolean", "nan", "event_ordinal", "event_term", "later_returns"],
)
def test_impossible_first_call_and_trace_locations_are_rejected(kind):
    data = byte_report()
    d = data["samples"][0]["timings"]["stages"]["input_load"]
    p = d["first_byte_probe"]
    if kind == "no_first":
        p["first_byte"].update(calls=0, seconds=0)
        p["first_byte_calls_returned"] = 0
        p["remaining_transfer_seconds"] = 0.0001
    elif kind == "missing_trace":
        p["memory_events_omitted"] = 2
        p["memory_events"] = []
    elif kind == "boolean":
        p["memory_in_first"]["requests"] = True
    elif kind == "nan":
        p["first_byte"]["seconds"] = float("nan")
    elif kind == "event_ordinal":
        p["memory_events"][1]["ordinal"] = 3
    elif kind == "event_term":
        p["memory_events"][1]["term"] = 0
    else:
        p["first_byte_calls_returned"] = 0
    with pytest.raises((ValueError, KeyError)):
        benchmark().input_load_summary(data, 1)


def test_current_native_cannot_downgrade_to_archived_v1(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    module = benchmark()
    raw = json.dumps(input_report()).encode()
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout=raw, stderr=b""),
    )
    failure = tmp_path / "failed-native.log"
    with pytest.raises(ValueError, match="v2 evidence"):
        module.runtime_measurements(
            dict(source="fixture", indices=["0"]),
            1,
            3,
            profile=True,
            profile_input_load=True,
            failure_output=failure,
        )
    assert failure.read_bytes() == raw


def test_growth_failure_callbacks_do_not_invent_a_request_or_prove_oom():
    data = byte_report()
    for sample in data["samples"]:
        p = sample["timings"]["stages"]["input_load"]["first_byte_probe"]
        p.update(memory_before_first=counts(), memory_in_first=counts(), memory_events=[])
        p["memory_in_first"]["grow_failed_callbacks"] = 1
    p = benchmark().input_load_summary(data, 1)["first_byte_probe"]
    assert p["memory_in_first"]["requests"] == [0] * 3
    assert p["memory_in_first"]["grow_failed_callbacks"] == [1] * 3
