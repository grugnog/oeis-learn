"""Independent candidate semantics and shared portfolio limits for T067."""

import time
import pytest
from tests.unit.test_progressive_components import definition, library
from oeis_learn.generalization.progressive.component_search import Proposals


def test_typed_search_expands_into_original_codec_and_charges_every_trial():
    from oeis_learn.decoder.bigint_codec import decode_body

    charges = []
    search = Proposals(library(definition()), charge=lambda **v: charges.append(v))
    result = search(4, time.monotonic() + 60)
    assert result["complete"]
    assert decode_body(result["tokens"]) == result["source"]
    assert "sub" in result["source"] and "call" not in result["source"]
    assert len(search.report["proposals"]) == 2
    assert len(charges) == 2
    assert search.report["expanded_nodes"] > 2
    assert search.report["scoring_tokens"] == 0
    assert search.report["library"] == search.library["identity"]


def test_only_four_component_slots_and_expired_work_is_never_selected():
    search = Proposals(library(definition()))
    for position in (0, 3, 8, True):
        with pytest.raises(ValueError):
            search(position, time.monotonic() + 60)
    value = search(4, time.monotonic() - 1)
    assert not value["complete"] and value["source"] == ""
    with pytest.raises(ValueError):
        search(4, time.monotonic() + 60)


def test_real_decoder_scoring_is_finite_and_charged():
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.generalization.progressive.baselines import load_random
    import torch

    e, d, _ = load_random(resolve({"version": "bigint-bootstrap/v1"}), seed=13)
    with torch.no_grad():
        memory = e.forward_from_sequences([list(range(20))], device="cpu")
    search = Proposals(library(definition()), decoder=d, memory=memory)
    result = search(4, time.monotonic() + 60)
    assert result["complete"]
    assert search.report["scoring_tokens"] > 0
    assert all(isinstance(r["model_score"], float) for r in search.report["proposals"])


def _drive(library_value, client):
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.fresh.bigint_diagnostics import _portfolio
    from oeis_learn.decoder.bigint_generation import GenerationRequest

    recipe = resolve({"version": "bigint-bootstrap/v1"})
    recipe.pop("pipeline", None)
    prompt = dict(
        id="heldout", indices=list(map(str, range(20))), values=[str(2 * i + 1) for i in range(20)]
    )
    proposals = Proposals(library_value) if library_value else None
    stream = _portfolio(
        None,
        None,
        prompt,
        recipe,
        client,
        deadline=time.monotonic() + 60,
        component_proposals=proposals,
    )
    ordinary = 0
    response = None
    while True:
        try:
            request = stream.send(response)
        except StopIteration as stopped:
            return ordinary, proposals, stopped.value
        assert isinstance(request, GenerationRequest)
        ordinary += 1
        response = dict(
            source="",
            complete=False,
            tokens=[],
            generation_seconds=0.0,
            generation_batch_size=1,
            failure="diagnostic incomplete draw",
        )


def test_existing_driver_spends_four_ordinary_and_four_component_slots():
    class Client:
        def __init__(self):
            self.calls = []

        def fit(self, source, prompt, **kwargs):
            assert "call" not in source
            self.calls.append((source, kwargs["deadline"]))
            return dict(admitted=[], complete=True, outcome="unsat")

    client = Client()
    ordinary, proposals, _ = _drive(library(definition()), client)
    assert ordinary == 4 and len(client.calls) == 4
    assert proposals.used == {4, 5, 6, 7}
    assert max(d for _, d in client.calls) <= time.monotonic() + 4
    ordinary, proposals, _ = _drive(None, Client())
    assert ordinary == 8 and proposals is None


def test_mined_library_input_preserves_original_cost_and_rejects_tampering(tmp_path):
    import copy
    from tests.unit.test_progressive_component_mining import sources
    from oeis_learn.generalization.progressive.component_mining import LibraryStore
    from oeis_learn.generalization.progressive.component_search import search_input, validate_input
    from oeis_learn.generalization.progressive.contracts import seal

    store = LibraryStore(tmp_path, "library", purpose="acceptance", track="synthetic_strict")
    value = store.publish(sources(tmp_path))
    bundle = search_input(store, value)
    assert sum(r["units"] for r in bundle["discovery_work"]) == store.spent()
    assert bundle["proofs"]
    bad = copy.deepcopy(bundle)
    bad["library"]["payload"]["components"] = []
    bad["library"] = seal(bad["library"])
    with pytest.raises(ValueError):
        validate_input(seal(bad))
    bad = copy.deepcopy(bundle)
    bad["proofs"] = []
    with pytest.raises(ValueError):
        validate_input(seal(bad))


def test_native_mined_component_reaches_original_fitter(tmp_path, fitting_client):
    from tests.unit.test_progressive_component_mining import sources
    from oeis_learn.generalization.progressive.component_mining import LibraryStore

    store = LibraryStore(tmp_path, "library", purpose="acceptance", track="synthetic_strict")
    value = store.publish(sources(tmp_path, fitting_client), client=fitting_client)
    ordinary, proposals, result = _drive(value, fitting_client)
    assert ordinary == 4 and proposals.used == {4, 5, 6, 7}
    assert result["equal_budget"] is not None


def test_slow_neural_decode_preserves_component_slots_inside_same_portfolio(monkeypatch):
    from oeis_learn.fresh.bigint_config import resolve
    from oeis_learn.fresh.bigint_diagnostics import _portfolio
    from oeis_learn.decoder.bigint_generation import GenerationRequest

    clock = [1000.0]
    monkeypatch.setattr(time, "monotonic", lambda: clock[0])
    recipe = resolve({"version": "bigint-bootstrap/v1"})
    recipe.pop("pipeline", None)
    prompt = dict(
        id="slow-prefix",
        indices=list(map(str, range(20))),
        values=[str(2 * i + 1) for i in range(20)],
    )
    proposals = Proposals(library(definition()))

    class Client:
        def fit(self, *args, **kwargs):
            return dict(admitted=[], complete=True, outcome="unsat")

    stream = _portfolio(
        None, None, prompt, recipe, Client(), deadline=1060, component_proposals=proposals
    )
    requests = []
    response = None
    while True:
        try:
            request = stream.send(response)
        except StopIteration:
            break
        assert isinstance(request, GenerationRequest)
        requests.append(request)
        # A neural attempt consumes its whole allowance; later component work
        # must remain reachable without increasing the 60-second ceiling.
        clock[0] = max(clock[0], request.deadline + 0.1)
        response = dict(
            source="",
            complete=False,
            tokens=[],
            generation_seconds=30.0,
            generation_batch_size=1,
            failure="deadline",
        )
    assert len(requests) == 4
    assert all(request.deadline == 1030 for request in requests)
    assert proposals.used == {4, 5, 6, 7}
    assert clock[0] < 1060
