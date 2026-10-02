"""Generic sampling and deterministic pool construction contracts."""

from concurrent.futures import Future
import pytest


def test_generic_sampler_reproducible_complete_and_disclosed():
    from oeis_learn.data.generic_programs import default_config, GenericSampler
    from oeis_learn.decoder.program_codec import decode_body, encode_body

    sampler = GenericSampler(default_config())
    samples = [sampler.sample(i) for i in range(128)]
    assert samples == [sampler.sample(i) for i in range(128)]
    valid = [s for s in samples if s["sampling_failure"] is None]
    assert valid
    for sample in valid:
        assert sample["body_tokens"] == encode_body(sample["canonical_source"])
        assert decode_body(sample["body_tokens"]) == sample["canonical_source"]
        assert sample["origin"] == "generic_sample"
        assert 1 <= sample["statistics"]["statements"] <= 32
        assert sample["statistics"]["loop_depth"] <= 3
        assert sample["statistics"]["structured_depth"] <= 8
    kinds = {k for s in samples for k in s["statistics"]["statement_kinds"]}
    assert kinds == {"assignment", "conditional", "counted_loop"}
    assert sampler.config["priors"]["statement_weights"] == [70, 15, 15]
    with pytest.raises(ValueError):
        sampler.sample(True)


def test_sampler_unknown_options_and_changed_priors_fail():
    from oeis_learn.data.generic_programs import default_config, GenericSampler

    config = default_config()
    config["teacher"] = "Fibonacci"
    with pytest.raises(ValueError):
        GenericSampler(config)
    config = default_config()
    config["priors"]["statement_weights"] = [100, 0, 0]
    with pytest.raises(ValueError):
        GenericSampler(config)


def test_completion_order_does_not_choose_publication_order():
    from oeis_learn.data.program_pool import ordered_results

    futures = {i: Future() for i in (2, 0, 1)}
    for i in (2, 1, 0):
        futures[i].set_result(str(i))
    assert list(ordered_results(futures)) == [(0, "0"), (1, "1"), (2, "2")]


def test_dedup_uses_outputs_then_shortest_and_source_hash():
    from oeis_learn.data.program_pool import representative_key

    a = {"body_tokens": [1, 2, 3], "source_sha256": "a", "sample_counter": 0}
    b = {"body_tokens": [1, 2], "source_sha256": "z", "sample_counter": 1}
    c = {"body_tokens": [1, 2], "source_sha256": "b", "sample_counter": 2}
    assert min((a, b, c), key=representative_key) is c


def test_clock_charges_crash_gap_across_boots_and_rejects_backward_time(monkeypatch):
    from oeis_learn.data import program_pool

    start = {"boot_id": "boot-a", "monotonic_ns": 100, "utc_ns": 1000}
    monkeypatch.setattr(
        program_pool, "_clock", lambda: dict(boot_id="boot-a", monotonic_ns=400, utc_ns=1500)
    )
    assert program_pool._elapsed(start) == 300
    monkeypatch.setattr(
        program_pool, "_clock", lambda: dict(boot_id="boot-b", monotonic_ns=1, utc_ns=1900)
    )
    assert program_pool._elapsed(start) == 900
    monkeypatch.setattr(
        program_pool, "_clock", lambda: dict(boot_id="boot-b", monotonic_ns=1, utc_ns=900)
    )
    with pytest.raises(ValueError, match="backward"):
        program_pool._elapsed(start)


def test_sampler_seed_is_independent_of_execution_and_budget_settings():
    from oeis_learn.data.generic_programs import default_config, GenericSampler

    first = default_config()
    second = dict(first, workers=1, target_count=1, max_attempts=128, budget_ns=1000)
    a, b = GenericSampler(first), GenericSampler(second)
    for i in range(32):
        left, right = a.sample(i), b.sample(i)
        assert left["canonical_source"] == right["canonical_source"]
        assert left["sample_seed"] == right["sample_seed"]


def test_sampler_leaves_and_constants_cover_the_frozen_prior():
    from collections import Counter
    from oeis_learn.data.generic_programs import default_config, GenericSampler

    sampler = GenericSampler(default_config())
    constants, leaves = Counter(), Counter()
    for i in range(256):
        sample = sampler.sample(i)
        constants.update(sample["statistics"]["constants"])
        leaves.update(
            key.split(":")[1]
            for key, count in sample["statistics"]["leaves"].items()
            for _ in range(count)
        )
    assert 0.82 < constants["small"] / constants.total() < 0.93
    assert {"8", "16", "32", "64", "128", "256"} <= constants.keys()
    assert all(0.28 < value / leaves.total() < 0.38 for value in leaves.values())
